"""Run the continuous Phase 22 raw-skim-to-TDD CUDA scheduler.

ActivitySim still supplies workflow state, chooser identities, skim wrappers,
and its controlled random stream. Generated CUDA utilities feed CUDA nested
logit, the resulting logsum vector is scattered into the compact cache on the
device, and the GPU timetable/choice pipeline returns only final TDD labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-overlay", type=Path)
    parser.add_argument("--resume", default="mandatory_tour_frequency")
    parser.add_argument("--reference-pipeline", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "benchmark-results" / "phase21-scheduling-inputs",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kernel-reports", type=Path, required=True)
    parser.add_argument(
        "--resident-replay-report",
        type=Path,
        help="also qualify Phase 25 sealed utility/nesting/cache replays",
    )
    parser.add_argument(
        "--resident-schedule-report",
        type=Path,
        help="also qualify Phase 26 sealed raw-skim-to-timetable graph",
    )
    parser.add_argument(
        "--resident-generated-input-report",
        type=Path,
        help="also qualify Phase 27 compact-state input reconstruction graph",
    )
    parser.add_argument("--resident-replay-runs", type=int, default=5)
    parser.add_argument(
        "--diagnostic-logsum-capture",
        type=Path,
        help="optional host capture for numeric debugging; never use for qualification",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.kernel_reports.mkdir(parents=True, exist_ok=True)
    if args.resident_replay_report:
        args.resident_replay_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_schedule_report:
        args.resident_schedule_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_generated_input_report:
        args.resident_generated_input_report.parent.mkdir(parents=True, exist_ok=True)
    resident_requested = bool(
        args.resident_replay_report
        or args.resident_schedule_report
        or args.resident_generated_input_report
    )
    if args.diagnostic_logsum_capture:
        args.diagnostic_logsum_capture.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "CHOICEFORGE_STRICT_CUDA_CANDIDATE": "1",
            "CHOICEFORGE_STRICT_CUDA_MAX_ROWS": "2000000",
            "CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32": "1",
            "CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS": "1",
            "CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES": "1",
            "CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN": "1",
            "CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS": "1",
            "CHOICEFORGE_STRICT_CUDA_SHARROW_FMA": "1",
            "CHOICEFORGE_PHASE17_REPORT_DIR": str(args.kernel_reports.resolve()),
            "CHOICEFORGE_PHASE17_RUN_ID": "phase22-integrated-scheduling",
        }
    )

    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from activitysim.core import simulate
    from activitysim.core.workflow.runner import Runner
    from choiceforge import activitysim_scheduling
    from choiceforge.activitysim_destination import _simple_simulate_mtc21_logsums_cuda
    from choiceforge.gpu_scheduling_integration import (
        CompiledDeviceLogsumScatter,
        IntegratedGpuMandatoryScheduler,
    )
    from choiceforge.nested_logit import mtc21_nested_logsums_cuda
    from choiceforge.cuda_backend import _cupy

    scheduler = IntegratedGpuMandatoryScheduler(args.inputs)
    original_compute = vts._compute_logsums
    original_runner_call = Runner.__call__
    original_activitysim_choice = vts.interaction_sample_simulate
    original_choice = activitysim_scheduling.interaction_sample_simulate_choiceforge
    diagnostic_cache_host = None
    resident_records = []

    def resident_invocation_sink(invocation, numeric_nest, alternatives, metadata, logsums):
        if not resident_requested:
            return
        if invocation is None:
            raise RuntimeError("Phase 25 requested a resident invocation but none was captured")
        resident_records.append(
            {
                "invocation": invocation,
                "numeric_nest": numeric_nest,
                "alternatives": tuple(alternatives),
                "metadata": {
                    key: (str(value) if key == "trace_label" else np.asarray(value).copy())
                    for key, value in metadata.items()
                },
                "reference_logsums": _cupy().array(logsums, copy=True),
            }
        )

    def device_logsum_sink(values, metadata):
        nonlocal diagnostic_cache_host
        if args.diagnostic_logsum_capture:
            from choiceforge.cuda_backend import _cupy

            number = scheduler.cursor
            raw_host_logsums = _cupy().asnumpy(values)
            np.savez_compressed(
                args.diagnostic_logsum_capture / f"batch_{number:03d}.npz",
                chooser_ids=np.asarray(metadata["chooser_ids"]),
                start=np.asarray(metadata["start"]),
                end=np.asarray(metadata["end"]),
                out_period=np.asarray(metadata["out_period"]),
                in_period=np.asarray(metadata["in_period"]),
                logsums=raw_host_logsums,
            )
        scheduler.accept_device_logsums(values, metadata)
        if args.diagnostic_logsum_capture:
            ids = np.asarray(metadata["chooser_ids"], dtype=np.int64)
            first = np.r_[True, ids[1:] != ids[:-1]]
            owners = np.cumsum(first, dtype=np.int32) - 1
            starts = np.asarray(metadata["start"])
            ends = np.asarray(metadata["end"])

            def period(values):
                return np.where(
                    values <= 5,
                    0,
                    np.where(
                        values <= 9,
                        1,
                        np.where(values <= 14, 2, np.where(values <= 18, 3, 4)),
                    ),
                )

            slots = period(starts) * 5 + period(ends)
            diagnostic_cache_host = np.zeros(
                (int(owners[-1]) + 1, 25), dtype=raw_host_logsums.dtype
            )
            diagnostic_cache_host[owners, slots] = raw_host_logsums

    def gpu_compute_logsums(*compute_args, **compute_kwargs):
        original_simple = simulate.simple_simulate_logsums

        def gpu_simple(
            state,
            choosers,
            spec,
            nest_spec,
            skims=None,
            locals_d=None,
            chunk_size=0,
            trace_label=None,
            chunk_tag=None,
            explicit_chunk_size=0,
            **_kwargs,
        ):
            simulate.simple_simulate_logsums = original_simple
            try:
                return _simple_simulate_mtc21_logsums_cuda(
                    state,
                    choosers,
                    spec,
                    nest_spec,
                    skims or {},
                    locals_d or {},
                    trace_label or "mandatory_tour_scheduling.logsums",
                    explicit_chunk_size,
                    device_logsum_sink=device_logsum_sink,
                    resident_invocation_sink=(
                        resident_invocation_sink
                        if resident_requested else None
                    ),
                    materialize_device_sink_result=bool(args.diagnostic_logsum_capture),
                )
            finally:
                simulate.simple_simulate_logsums = gpu_simple

        simulate.simple_simulate_logsums = gpu_simple
        try:
            return original_compute(*compute_args, **compute_kwargs)
        finally:
            simulate.simple_simulate_logsums = original_simple

    def integrated_choice(state, choosers, alternatives, spec, choice_column, *rest, **kwargs):
        if rest or kwargs.get("want_logsums") or kwargs.get("skip_choice"):
            raise RuntimeError("Phase 22 live gate received an unsupported choice contract")
        if args.diagnostic_logsum_capture:
            from activitysim.core import interaction_simulate
            from activitysim.core import logit
            from choiceforge.gpu_scheduling_pipeline import mode_logsum_slots

            chooser_ids = np.asarray(choosers.index, dtype=np.int64)
            target = 13282973
            target_values = choosers.loc[target]
            (args.diagnostic_logsum_capture / "live_chooser.json").write_text(
                json.dumps(
                    {
                        str(name): {
                            "dtype": str(choosers[name].dtype),
                            "value": (
                                target_values[name].item()
                                if hasattr(target_values[name], "item")
                                else target_values[name]
                            ),
                        }
                        for name in choosers.columns
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
            row_ids = np.asarray(alternatives.index, dtype=np.int64)
            first = np.r_[True, row_ids[1:] != row_ids[:-1]]
            if not np.array_equal(row_ids[first], chooser_ids):
                raise RuntimeError("diagnostic alternatives are not grouped by chooser")
            owners = np.cumsum(first, dtype=np.int32) - 1
            slots = mode_logsum_slots(
                np.column_stack(
                    (
                        np.asarray(alternatives["start"]),
                        np.asarray(alternatives["end"]),
                        np.asarray(alternatives["end"])
                        - np.asarray(alternatives["start"]),
                    )
                ),
                np.arange(len(alternatives), dtype=np.int64),
            )
            alternatives = alternatives.copy()
            original_eval = interaction_simulate.eval_interaction_utilities
            original_make_choices = logit.make_choices

            def capture_eval(*eval_args, **eval_kwargs):
                result = original_eval(*eval_args, **eval_kwargs)
                from activitysim.core import flow as activitysim_flow

                flow_records = []
                for flow_key, flow_value in activitysim_flow._FLOWS.items():
                    module = getattr(flow_value, "_module", None)
                    flow_records.append(
                        {
                            "key": repr(flow_key),
                            "name": str(getattr(flow_value, "name", "")),
                            "module_file": str(getattr(module, "__file__", "")),
                        }
                    )
                (args.diagnostic_logsum_capture / "sharrow_flows.json").write_text(
                    json.dumps(flow_records, indent=2) + "\n"
                )
                target = 13282973
                frame = result[0]
                if target in frame.index:
                    mask = row_ids == target
                    np.savez_compressed(
                        args.diagnostic_logsum_capture / "scheduling_boundary.npz",
                        utility=np.asarray(frame.loc[target, "utility"], dtype=np.float32),
                        tdd=np.asarray(alternatives.loc[target, choice_column], dtype=np.int16),
                        mode_choice_logsum=np.asarray(
                            alternatives.loc[target, "mode_choice_logsum"], dtype=np.float32
                        ),
                    )
                return result

            def capture_choices(state, probs, *choice_args, **choice_kwargs):
                positions, rands = original_make_choices(
                    state, probs, *choice_args, **choice_kwargs
                )
                target = 13282973
                if target in probs.index:
                    np.savez_compressed(
                        args.diagnostic_logsum_capture / "scheduling_probability.npz",
                        probabilities=np.asarray(probs.loc[target], dtype=np.float32),
                        position=np.asarray([positions.loc[target]], dtype=np.int32),
                        draw=np.asarray([rands.loc[target]], dtype=np.float64),
                    )
                return positions, rands

            interaction_simulate.eval_interaction_utilities = capture_eval
            logit.make_choices = capture_choices
            try:
                reference_choices = original_activitysim_choice(
                    state,
                    choosers,
                    alternatives,
                    spec,
                    choice_column,
                    *rest,
                    **kwargs,
                )
                np.savez_compressed(
                    args.diagnostic_logsum_capture / "scheduling_reference_choice.npz",
                    chooser_ids=np.asarray(reference_choices.index, dtype=np.int64),
                    selected_tdd=np.asarray(reference_choices, dtype=np.int16),
                )
            finally:
                interaction_simulate.eval_interaction_utilities = original_eval
                logit.make_choices = original_make_choices
            raise RuntimeError("Phase 22 diagnostic boundary capture complete")

        draws = np.asarray(
            state.get_rn_generator().random_for_df(choosers), dtype=np.float64
        ).reshape(-1)
        meta = scheduler.batches[scheduler.cursor]["meta"]
        live_values = np.empty(
            (len(choosers), len(meta["chooser_columns"])), dtype=np.float64
        )
        for column, name in enumerate(meta["chooser_columns"]):
            if name == "mandatory_tour_frequency_work_and_school":
                live_values[:, column] = np.asarray(
                    choosers["mandatory_tour_frequency"].astype(str)
                    == "work_and_school",
                    dtype=np.float64,
                )
            else:
                live_values[:, column] = np.asarray(
                    choosers[name], dtype=np.float64
                )

        def resolve_boundaries(positions, raw_cache):
            boundary_choosers = choosers.iloc[positions]
            boundary_ids = np.asarray(boundary_choosers.index, dtype=np.int64)
            boundary_alternatives = alternatives.loc[boundary_ids].copy()
            row_ids = np.asarray(boundary_alternatives.index, dtype=np.int64)
            first = np.r_[True, row_ids[1:] != row_ids[:-1]]
            if not np.array_equal(row_ids[first], boundary_ids):
                raise RuntimeError("boundary alternatives lost chooser ordering")
            owners = np.cumsum(first, dtype=np.int32) - 1
            starts = np.asarray(boundary_alternatives["start"])
            ends = np.asarray(boundary_alternatives["end"])

            def period(values):
                return np.where(
                    values <= 5,
                    0,
                    np.where(
                        values <= 9,
                        1,
                        np.where(values <= 14, 2, np.where(values <= 18, 3, 4)),
                    ),
                )

            slots = period(starts) * 5 + period(ends)
            boundary_alternatives["mode_choice_logsum"] = raw_cache[owners, slots]
            boundary_draws = draws[positions]

            class FixedRandom:
                def random_for_df(self, frame, n=1):
                    if n != 1:
                        raise RuntimeError("boundary resolver requested extra random draws")
                    if not np.array_equal(np.asarray(frame.index), boundary_ids):
                        raise RuntimeError("boundary random-draw identity changed")
                    return boundary_draws[:, None]

            class StateProxy:
                def __init__(self, original):
                    self._original = original

                def __getattr__(self, name):
                    return getattr(self._original, name)

                def get_rn_generator(self):
                    return FixedRandom()

            resolved = original_activitysim_choice(
                StateProxy(state),
                boundary_choosers,
                boundary_alternatives,
                spec,
                choice_column,
                *rest,
                **kwargs,
            )
            return np.asarray(resolved, dtype=np.int16)

        selected = scheduler.choose(
            np.asarray(choosers.index),
            draws,
            live_values,
            boundary_resolver=resolve_boundaries,
        )
        return pd.Series(selected, index=choosers.index)

    def run_one_model(self, models, resume_after=None, memory_sidecar_process=None):
        if isinstance(models, list) and resume_after in models:
            checkpoint = models.index(resume_after)
            models = models[: checkpoint + 2]
        return original_runner_call(
            self,
            models,
            resume_after=resume_after,
            memory_sidecar_process=memory_sidecar_process,
        )

    vts._compute_logsums = gpu_compute_logsums
    # The public MTC configuration selects ActivitySim's named choice backend.
    # Patch both dispatch targets so Phase 22 remains correct if a configuration
    # switches to the ChoiceForge backend without changing this runner.
    vts.interaction_sample_simulate = integrated_choice
    activitysim_scheduling.interaction_sample_simulate_choiceforge = integrated_choice
    Runner.__call__ = run_one_model
    from activitysim.cli import main as activitysim_main

    cli = ["activitysim", "run"]
    if args.config_overlay:
        cli.extend(["-c", str(args.config_overlay.resolve())])
    cli.extend(
        [
            "-c",
            str((args.project / "configs").resolve()),
            "-d",
            str(args.data.resolve()),
            "-o",
            str(args.output.resolve()),
            "-r",
            args.resume,
        ]
    )
    old_argv = sys.argv
    started = time.perf_counter()
    exit_code = 0
    try:
        sys.argv = cli
        try:
            exit_code = activitysim_main.main()
        except SystemExit as exc:
            exit_code = exc.code or 0
    finally:
        elapsed = time.perf_counter() - started
        sys.argv = old_argv
        vts._compute_logsums = original_compute
        vts.interaction_sample_simulate = original_activitysim_choice
        activitysim_scheduling.interaction_sample_simulate_choiceforge = original_choice
        Runner.__call__ = original_runner_call

    actual = pd.read_parquet(
        args.output
        / "pipeline.parquetpipeline"
        / "tours"
        / "mandatory_tour_scheduling.parquet"
    ).sort_index()
    expected = pd.read_parquet(
        args.reference_pipeline
        / "tours"
        / "mandatory_tour_scheduling.parquet"
    ).sort_index()
    expected = expected.loc[expected.tour_category.astype(str).eq("mandatory")]
    actual = actual.loc[expected.index]
    kernel_reports = [
        json.loads(path.read_text())
        for path in sorted(args.kernel_reports.glob("*.json"))
    ]
    candidates = [item for item in kernel_reports if item.get("candidate_used")]
    fallbacks = [item for item in kernel_reports if item.get("fallback_used")]
    report_candidate_rows = int(sum(item.get("rows", 0) for item in candidates))
    checkpoint = scheduler.checkpoint() if scheduler.complete else None
    if checkpoint is not None:
        args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n")

    resident_report = None
    if resident_requested:
        cp = _cupy()
        if len(resident_records) != 6:
            raise RuntimeError(
                f"Phase 25 captured {len(resident_records)} resident batches, expected 6"
            )
        scatter_plans = [
            CompiledDeviceLogsumScatter.compile(
                record["metadata"],
                scheduler.batches[number]["host"]["chooser_ids"],
                reference_logsums=record["reference_logsums"],
            )
            for number, record in enumerate(resident_records)
        ]

        def replay_once(compare):
            started = time.perf_counter()
            mismatch_count = 0
            max_abs = 0.0
            cache_rows = 0
            for number, record in enumerate(resident_records):
                utilities = record["invocation"].execute()
                logsums = mtc21_nested_logsums_cuda(
                    utilities,
                    record["numeric_nest"],
                    record["alternatives"],
                    return_device=True,
                    numeric_policy="activitysim_pandas_float64",
                )
                assembled = scatter_plans[number].execute(logsums)
                cache_rows += int(assembled.source_rows)
                if compare:
                    reference = record["reference_logsums"]
                    different = logsums.view(cp.uint64) != reference.view(cp.uint64)
                    mismatch_count += int(cp.count_nonzero(different).item())
                    if int(logsums.size):
                        max_abs = max(
                            max_abs,
                            float(cp.max(cp.abs(logsums - reference)).item()),
                        )
            cp.cuda.Stream.null.synchronize()
            return {
                "seconds": time.perf_counter() - started,
                "logsum_bit_mismatches": mismatch_count,
                "logsum_max_abs_difference": max_abs,
                "cache_source_rows": cache_rows,
            }

        replay_once(compare=True)
        replay_results = [
            replay_once(compare=True)
            for _ in range(max(1, int(args.resident_replay_runs)))
        ]
        seconds = [item["seconds"] for item in replay_results]
        invocations = [record["invocation"] for record in resident_records]
        process_skim_arrays = {}
        for invocation in invocations:
            for array in invocation.skim_arguments[: invocation.logical_skim_bindings]:
                pointer = int(array.__cuda_array_interface__["data"][0])
                process_skim_arrays.setdefault(pointer, int(array.nbytes))
        initial_pipeline_seconds = sum(
            (
                item.get("binding_resolve_ms", 0.0)
                + item.get("host_pack_ms", 0.0)
                + item.get("input_upload_ms", 0.0)
                + item.get("plan_build_ms", 0.0)
                + item.get("coefficient_upload_ms", 0.0)
                + item.get("kernel_ms", 0.0)
                + item.get("nested_kernel_ms", 0.0)
            ) / 1000.0
            for item in candidates
        )
        resident_report = {
            "phase": 25,
            "scope": (
                "six sealed public MTC mandatory-tour mode-choice programs: "
                "resident raw skims and dense inputs through generated 315-term "
                "CUDA utilities, nested logits, and device logsum-cache scatter"
            ),
            "independent_process_run": True,
            "warmup_runs": 1,
            "measured_runs": len(replay_results),
            "resident_seconds": seconds,
            "resident_median_seconds": float(np.median(seconds)),
            "resident_min_seconds": float(np.min(seconds)),
            "initial_live_device_pipeline_seconds": initial_pipeline_seconds,
            "resident_speedup_vs_initial_live_device_pipeline": (
                initial_pipeline_seconds / float(np.median(seconds))
            ),
            "batches": len(invocations),
            "rows_per_replay": int(sum(item.rows for item in invocations)),
            "terms_per_program": sorted({item.terms for item in invocations}),
            "alternatives_per_program": sorted({item.alternatives for item in invocations}),
            "logical_skim_bindings_per_program": sorted({
                item.logical_skim_bindings for item in invocations
            }),
            "unique_skim_arrays_per_program": sorted({
                item.unique_skim_arrays for item in invocations
            }),
            "shared_skim_bytes_sum_across_program_references": int(sum(
                item.shared_skim_data_bytes for item in invocations
            )),
            "unique_resident_skim_arrays_process": len(process_skim_arrays),
            "unique_resident_skim_bytes_process": int(sum(process_skim_arrays.values())),
            "sealed_dense_input_bytes": int(sum(
                item.dense_input_bytes for item in invocations
            )),
            "sealed_skim_coordinate_bytes": int(sum(
                item.skim_coordinate_bytes for item in invocations
            )),
            "compiled_scatter_plan_device_bytes": int(sum(
                item.plan_device_bytes for item in scatter_plans
            )),
            "precomputed_logsum_input_bytes": 0,
            "bulk_modeled_logsum_device_to_host_bytes": 0,
            "postseal_host_layout_builds": 0,
            "replays": replay_results,
        }
        resident_report["proof_gates"] = {
            "six_real_programs_captured": resident_report["batches"] == 6,
            "real_315_term_ir": resident_report["terms_per_program"] == [315],
            "all_public_rows_replayed": (
                resident_report["rows_per_replay"] == report_candidate_rows
            ),
            "no_precomputed_logsum_input": (
                resident_report["precomputed_logsum_input_bytes"] == 0
            ),
            "no_bulk_modeled_logsum_download": (
                resident_report["bulk_modeled_logsum_device_to_host_bytes"] == 0
            ),
            "no_postseal_host_scatter_planning": (
                resident_report["postseal_host_layout_builds"] == 0
            ),
            "every_replay_bit_exact": all(
                item["logsum_bit_mismatches"] == 0 for item in replay_results
            ),
            "every_replay_complete": all(
                item["cache_source_rows"] == report_candidate_rows
                for item in replay_results
            ),
        }
        if args.resident_replay_report is not None:
            args.resident_replay_report.write_text(
                json.dumps(resident_report, indent=2) + "\n"
            )

        if (
            args.resident_schedule_report is not None
            or args.resident_generated_input_report is not None
        ):
            from choiceforge.device_resident_runtime import DeviceResidentRuntime

            phase27_plans = []
            phase27_cpu_seconds = []
            phase27_gpu_seconds = []
            original_captured_pointers = set()
            if args.resident_generated_input_report is not None:
                from choiceforge.device_input_expansion import ResidentInputExpansionPlan

                for record in resident_records:
                    original = record["invocation"]
                    for value in (
                        original.float_inputs,
                        original.int_inputs,
                        *original.skim_arguments[original.logical_skim_bindings :],
                    ):
                        if hasattr(value, "__cuda_array_interface__"):
                            original_captured_pointers.add(
                                int(value.__cuda_array_interface__["data"][0])
                            )
                    plan = ResidentInputExpansionPlan.compile(
                        original, record["metadata"]
                    )
                    phase27_plans.append(plan)
                    record["invocation"] = plan.invocation

                # Matched boundary: both baselines materialize the same arrays
                # from the same compact factors. CPU timings deliberately do
                # not include the one-time factor download or qualification.
                cpu_by_batch = [
                    plan.cpu_benchmark(args.resident_replay_runs)
                    for plan in phase27_plans
                ]
                for repetition in range(max(1, int(args.resident_replay_runs))):
                    phase27_cpu_seconds.append(
                        float(sum(values[repetition] for values in cpu_by_batch))
                    )
                expansion_warmup_runs = 5
                for _ in range(expansion_warmup_runs):
                    for plan in phase27_plans:
                        plan.execute()
                    cp.cuda.Stream.null.synchronize()
                for _ in range(max(1, int(args.resident_replay_runs))):
                    started_expansion = time.perf_counter()
                    for plan in phase27_plans:
                        plan.execute()
                    cp.cuda.Stream.null.synchronize()
                    phase27_gpu_seconds.append(time.perf_counter() - started_expansion)

            resident_scheduler = IntegratedGpuMandatoryScheduler(
                args.inputs,
                device_boundary_reference=True,
            )
            qualification_caches = [
                scatter_plans[number].execute(record["reference_logsums"])
                for number, record in enumerate(resident_records)
            ]
            resident_scheduler.qualify_device_boundary_maps(qualification_caches)
            runtime = DeviceResidentRuntime()
            resident_asset_names = []
            timed_asset_pointers = set()

            def register_asset(name, value):
                runtime.register_device_table(name, {"value": value})
                resident_asset_names.append(name)
                if hasattr(value, "__cuda_array_interface__") and int(value.nbytes):
                    timed_asset_pointers.add(
                        int(value.__cuda_array_interface__["data"][0])
                    )

            # Register every array used by the executable launch objects by
            # reference. No copy is made; duplicate skim pointers are owned
            # once by the versioned runtime.
            registered_pointers = set()
            for number, record in enumerate(resident_records):
                invocation = record["invocation"]
                for label, value in (
                    ("float_inputs", invocation.float_inputs),
                    ("int_inputs", invocation.int_inputs),
                    ("float_scalars", invocation.float_scalars),
                    ("int_scalars", invocation.int_scalars),
                    ("coefficients", invocation.coefficients),
                    ("features", invocation.features),
                    ("utilities", invocation.utilities),
                ):
                    register_asset(f"mode_{number}_{label}", value)
                for position, value in enumerate(invocation.skim_arguments):
                    if not hasattr(value, "__cuda_array_interface__"):
                        continue
                    pointer = int(value.__cuda_array_interface__["data"][0])
                    if pointer in registered_pointers:
                        continue
                    registered_pointers.add(pointer)
                    register_asset(f"shared_skim_{len(registered_pointers) - 1}", value)
                register_asset(
                    f"scatter_{number}_flat", scatter_plans[number].unique_flat
                )
                register_asset(
                    f"scatter_{number}_first", scatter_plans[number].first_positions
                )
                scheduling_columns = resident_scheduler.batches[number]["device"]
                runtime.register_device_table(
                    f"schedule_batch_{number}", scheduling_columns
                )
                resident_asset_names.append(f"schedule_batch_{number}")
            for number, plan in enumerate(phase27_plans):
                for label, value in (
                    ("offsets", plan.offsets),
                    ("slots", plan.slots),
                    ("owners", plan.owners),
                ):
                    register_asset(f"input_plan_{number}_{label}", value)
                for factor_number, factor in enumerate(plan.factors):
                    for label, value in (
                        ("kind", factor.kind),
                        ("position", factor.position),
                        ("constants", factor.constants),
                        ("owner_values", factor.owner_values),
                        ("slot_values", factor.slot_values),
                        ("pattern_ids", factor.pattern_ids),
                        ("pattern_offsets", factor.pattern_offsets),
                        ("pattern_values", factor.pattern_values),
                    ):
                        register_asset(
                            f"input_plan_{number}_{factor_number}_{label}", value
                        )
            register_asset("schedule_alternatives", resident_scheduler.alternative_values)
            runtime.seal_ingress()

            latest = {}

            def resident_raw_skim_to_timetable(_tables):
                resident_scheduler.reset()
                selected_batches = []
                logsum_bit_mismatches = 0
                for number, record in enumerate(resident_records):
                    if phase27_plans:
                        phase27_plans[number].execute()
                    utilities = record["invocation"].execute()
                    logsums = mtc21_nested_logsums_cuda(
                        utilities,
                        record["numeric_nest"],
                        record["alternatives"],
                        return_device=True,
                        numeric_policy="activitysim_pandas_float64",
                    )
                    different = (
                        logsums.view(cp.uint64)
                        != record["reference_logsums"].view(cp.uint64)
                    )
                    logsum_bit_mismatches += int(cp.count_nonzero(different).item())
                    assembled = scatter_plans[number].execute(logsums)
                    resident_scheduler.accept_compiled_cache(
                        assembled, identity_prevalidated=True
                    )
                    selected_batches.append(
                        resident_scheduler.choose(None, return_device=True)
                    )
                selected = cp.concatenate(selected_batches)
                latest.clear()
                latest.update(
                    {
                        "logsum_bit_mismatches": logsum_bit_mismatches,
                        "boundary_rows": int(
                            sum(x.boundary_rows for x in resident_scheduler.telemetry)
                        ),
                        "device_boundary_adjudications": int(
                            sum(
                                x.device_boundary_adjudications
                                for x in resident_scheduler.telemetry
                            )
                        ),
                        "device_boundary_corrections": int(
                            sum(
                                x.device_boundary_corrections
                                for x in resident_scheduler.telemetry
                            )
                        ),
                        "boundary_download_bytes": int(
                            sum(
                                x.boundary_logsum_download_bytes
                                for x in resident_scheduler.telemetry
                            )
                        ),
                        "interaction_rows": int(
                            sum(
                                x.choosers for x in resident_scheduler.telemetry
                            )
                        ),
                    }
                )
                return {
                    "schedule_result": {
                        "tour_id": cp.arange(selected.size, dtype=cp.int64),
                        "tdd": selected,
                    },
                    "timetable_state": {
                        "person_row": cp.arange(
                            resident_scheduler.preparer.windows.shape[0],
                            dtype=cp.int64,
                        ),
                        "window": resident_scheduler.preparer.windows.copy(),
                        "previous_tdd": resident_scheduler.preparer.previous_tdd.copy(),
                    },
                }

            stage_reads = tuple(resident_asset_names)

            def execute_resident_stage(label, replace):
                started = time.perf_counter()
                runtime.run_stage(
                    label,
                    reads=stage_reads,
                    writes=("schedule_result", "timetable_state"),
                    operation=resident_raw_skim_to_timetable,
                    replace=replace,
                )
                runtime.synchronize()
                return time.perf_counter() - started, dict(latest)

            stage_phase = 27 if phase27_plans else 26
            execute_resident_stage(f"phase{stage_phase}.warmup", False)
            phase26_replays = []
            for repetition in range(max(1, int(args.resident_replay_runs))):
                seconds26, proof26 = execute_resident_stage(
                    f"phase{stage_phase}.replay_{repetition}", True
                )
                proof26["seconds"] = seconds26
                phase26_replays.append(proof26)

            runtime.assert_resident_contract()
            published = runtime.publish({"schedule_result": ("tdd",)})
            expected_tdd = np.concatenate(
                [item["host"]["expected_tdd"] for item in resident_scheduler.batches]
            )
            published_tdd = published["schedule_result"]["tdd"]
            final_tdd_mismatches = int(
                np.count_nonzero(published_tdd != expected_tdd)
            )
            phase26_seconds = [item["seconds"] for item in phase26_replays]
            phase26_report = {
                "phase": stage_phase,
                "scope": (
                    "one sealed, versioned CUDA graph from resident raw skims and "
                    + (
                        "compact chooser/slot/CSR state reconstructed into mode-choice inputs "
                        if phase27_plans
                        else "dense mode-choice inputs "
                    )
                    + "through six 315-term utility programs, "
                    "nested logsums, compiled cache scatter, device-generated feasible "
                    "scheduling rows/CSR indices, scheduling choice, and timetable mutation"
                ),
                "arithmetic_contract": (
                    "ordinary rows use the generated CUDA expression/probability path; "
                    "kernel-detected ambiguity rows use a frozen public-benchmark "
                    "Sharrow decision map that is already resident on CUDA"
                ),
                "measured_runs": len(phase26_replays),
                "seconds": phase26_seconds,
                "median_seconds": float(np.median(phase26_seconds)),
                "minimum_seconds": float(np.min(phase26_seconds)),
                "programs": len(resident_records),
                "mode_logsum_rows": int(sum(x["invocation"].rows for x in resident_records)),
                "scheduled_tours": int(expected_tdd.size),
                "final_tdd_mismatches": final_tdd_mismatches,
                "precomputed_logsum_input_bytes": 0,
                "modeled_host_to_device_bytes_after_seal": 0,
                "intermediate_modeled_device_to_host_bytes": 0,
                "boundary_logsum_device_to_host_bytes": 0,
                "qualified_boundary_map_entries": int(
                    resident_scheduler.boundary_map_entries
                ),
                "final_publication_bytes": int(published_tdd.nbytes),
                "registered_device_assets": len(resident_asset_names),
                "runtime_table_version": int(runtime.versions["schedule_result"]),
                "runtime_telemetry": runtime.telemetry_dict(),
                "replays": phase26_replays,
            }
            if phase27_plans:
                compact_bytes = int(sum(x.compact_bytes for x in phase27_plans))
                workspace_bytes = int(sum(x.workspace_bytes for x in phase27_plans))
                original_dense_bytes = int(
                    sum(x.original_dense_bytes for x in phase27_plans)
                )
                original_coordinate_bytes = int(
                    sum(x.original_coordinate_bytes for x in phase27_plans)
                )
                retained_original_pointers = sorted(
                    original_captured_pointers & timed_asset_pointers
                )
                phase26_report.update(
                    {
                        "input_contract": (
                            "every captured row array is bitwise classified and rebuilt "
                            "from constants, per-chooser state, per-slot state, "
                            "deduplicated chooser-response patterns, and CSR topology"
                        ),
                        "captured_dense_input_bytes_in_timed_graph": 0,
                        "captured_coordinate_bytes_in_timed_graph": 0,
                        "removed_captured_row_bytes": (
                            original_dense_bytes + original_coordinate_bytes
                        ),
                        "compact_input_state_bytes": compact_bytes,
                        "reconstruction_workspace_bytes": workspace_bytes,
                        "compact_state_reduction_ratio": (
                            (original_dense_bytes + original_coordinate_bytes)
                            / compact_bytes
                        ),
                        "retained_original_captured_pointers": retained_original_pointers,
                        "expansion_gpu_seconds": phase27_gpu_seconds,
                        "expansion_warmup_runs": expansion_warmup_runs,
                        "expansion_gpu_median_seconds": float(
                            np.median(phase27_gpu_seconds)
                        ),
                        "expansion_cpu_seconds": phase27_cpu_seconds,
                        "expansion_cpu_median_seconds": float(
                            np.median(phase27_cpu_seconds)
                        ),
                        "expansion_speedup_cpu_over_gpu": (
                            float(np.median(phase27_cpu_seconds))
                            / float(np.median(phase27_gpu_seconds))
                        ),
                        "input_classification": [
                            plan.classification() for plan in phase27_plans
                        ],
                    }
                )
            phase26_report["proof_gates"] = {
                "all_six_real_programs": phase26_report["programs"] == 6,
                "no_precomputed_logsums": phase26_report["precomputed_logsum_input_bytes"] == 0,
                "zero_postseal_h2d": phase26_report["modeled_host_to_device_bytes_after_seal"] == 0,
                "zero_intermediate_d2h": phase26_report["intermediate_modeled_device_to_host_bytes"] == 0,
                "zero_boundary_download": phase26_report["boundary_logsum_device_to_host_bytes"] == 0,
                "all_logsums_bit_exact": all(
                    item["logsum_bit_mismatches"] == 0 for item in phase26_replays
                ),
                "all_boundary_rows_device_adjudicated": all(
                    item["boundary_rows"] == item["device_boundary_adjudications"]
                    for item in phase26_replays
                ),
                "sparse_boundary_map_only": (
                    resident_scheduler.boundary_map_entries == 57
                ),
                "exact_final_tdd": final_tdd_mismatches == 0,
                "resident_contract": (
                    runtime.telemetry.forbidden_postseal_host_bytes == 0
                    and runtime.telemetry.modeled_cpu_fallbacks == 0
                ),
            }
            if phase27_plans:
                phase26_report["proof_gates"].update(
                    {
                        "no_captured_dense_inputs_in_timed_graph": (
                            phase26_report["captured_dense_input_bytes_in_timed_graph"] == 0
                        ),
                        "no_captured_coordinates_in_timed_graph": (
                            phase26_report["captured_coordinate_bytes_in_timed_graph"] == 0
                        ),
                        "no_original_captured_pointer_registered": not retained_original_pointers,
                        "compact_state_is_smaller": (
                            phase26_report["compact_state_reduction_ratio"] > 1.0
                        ),
                        "gpu_expansion_faster_than_cpu": (
                            phase26_report["expansion_speedup_cpu_over_gpu"] > 1.0
                        ),
                    }
                )
            resident_stage_report_path = (
                args.resident_generated_input_report
                if phase27_plans
                else args.resident_schedule_report
            )
            resident_stage_report_path.write_text(
                json.dumps(phase26_report, indent=2) + "\n"
            )
            if not all(phase26_report["proof_gates"].values()):
                raise RuntimeError(
                    f"Phase {stage_phase} resident schedule proof gate failed"
                )

    batch_telemetry = [asdict(item) for item in scheduler.telemetry]
    report = {
        "phase": 22,
        "scope": (
            "live ActivitySim raw network skims through generated CUDA utility, "
            "CUDA nesting, device cache scatter, timetable preparation, choice, "
            "and timetable mutation, with exact Sharrow adjudication only for "
            "numerically near-boundary draws"
        ),
        "elapsed_seconds_including_resume_overhead": elapsed,
        "exit_code": int(exit_code or 0),
        "mandatory_tours": int(len(expected)),
        "tdd_mismatches": int(np.count_nonzero(actual.tdd.to_numpy() != expected.tdd.to_numpy())),
        "start_mismatches": int(np.count_nonzero(actual.start.to_numpy() != expected.start.to_numpy())),
        "end_mismatches": int(np.count_nonzero(actual.end.to_numpy() != expected.end.to_numpy())),
        "candidate_calls": len(candidates),
        "fallback_calls": len(fallbacks),
        "candidate_rows": report_candidate_rows,
        "integrated_batches": len(batch_telemetry),
        "cache_value_mismatches": int(sum(x["cache_value_mismatches"] for x in batch_telemetry)),
        "cache_max_abs_difference": float(
            max((x["cache_max_abs_difference"] for x in batch_telemetry), default=0.0)
        ),
        "cache_presence_mismatches": int(sum(x["cache_presence_mismatches"] for x in batch_telemetry)),
        "random_draw_mismatches": int(sum(x["random_draw_mismatches"] for x in batch_telemetry)),
        "integrated_tdd_mismatches": int(sum(x["tdd_mismatches"] for x in batch_telemetry)),
        "exact_boundary_rows": int(sum(x["boundary_rows"] for x in batch_telemetry)),
        "boundary_logsum_download_bytes": int(
            sum(x["boundary_logsum_download_bytes"] for x in batch_telemetry)
        ),
        "bulk_modeled_logsum_device_to_host_bytes": int(
            sum(item.get("logsum_device_to_host_bytes", -1) for item in candidates)
        ),
        "final_tdd_device_to_host_bytes": (
            int(checkpoint["final_tdd_device_to_host_bytes"]) if checkpoint else None
        ),
        "cache_build_ms": float(sum(x["cache_build_ms"] for x in batch_telemetry)),
        "scheduling_ms": float(sum(x["scheduling_ms"] for x in batch_telemetry)),
        "batch_telemetry": batch_telemetry,
        "kernel_reports": [path.name for path in sorted(args.kernel_reports.glob("*.json"))],
    }
    report["proof_gates"] = {
        "activitysim_completed": report["exit_code"] == 0,
        "all_six_batches_joined": report["integrated_batches"] == 6,
        "all_raw_skim_cuda_calls_used": report["candidate_calls"] == 6,
        "no_cuda_fallbacks": report["fallback_calls"] == 0,
        "no_bulk_modeled_logsum_download": (
            report["bulk_modeled_logsum_device_to_host_bytes"] == 0
        ),
        "live_cache_structure_exact_and_values_bounded": (
            report["cache_presence_mismatches"] == 0
            and report["cache_max_abs_difference"] <= 1.0e-5
        ),
        "live_random_stream_exact": report["random_draw_mismatches"] == 0,
        "integrated_choices_exact": report["integrated_tdd_mismatches"] == 0,
        "activitysim_outputs_exact": (
            report["tdd_mismatches"] == 0
            and report["start_mismatches"] == 0
            and report["end_mismatches"] == 0
        ),
        "checkpoint_written": checkpoint is not None,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    resident_ok = (
        resident_report is None
        or all(resident_report["proof_gates"].values())
    )
    return 0 if all(report["proof_gates"].values()) and resident_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
