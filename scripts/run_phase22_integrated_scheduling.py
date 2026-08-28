"""Run the continuous Phase 22 raw-skim-to-TDD CUDA scheduler.

ActivitySim still supplies workflow state, chooser identities, skim wrappers,
and its controlled random stream. Generated CUDA utilities feed CUDA nested
logit, the resulting logsum vector is scattered into the compact cache on the
device, and the GPU timetable/choice pipeline returns only final TDD labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
    parser.add_argument("--config-overlay", type=Path, action="append")
    parser.add_argument("--resume", default="mandatory_tour_frequency")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help=(
            "Phase 32: run the complete model and release shared CUDA skims "
            "after the final qualified GPU consumer"
        ),
    )
    parser.add_argument(
        "--phase33-model-wide",
        action="store_true",
        help=(
            "extend the qualified full-model CUDA boundary to non-mandatory "
            "destination logsums, non-mandatory scheduling, and primary tour mode"
        ),
    )
    parser.add_argument(
        "--phase34-location-choice",
        action="store_true",
        help=(
            "extend the Phase 33 runtime to school, workplace, joint-tour, "
            "and at-work location-choice logsum programs"
        ),
    )
    parser.add_argument("--households-sample-size", type=int, default=50_000)
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
    parser.add_argument(
        "--resident-semantic-input-report",
        type=Path,
        help="also qualify Phase 28 named semantic CUDA input generation graph",
    )
    parser.add_argument(
        "--resident-raw-table-input-report",
        type=Path,
        help="also qualify Phase 29 declared raw-table-to-calendar CUDA graph",
    )
    parser.add_argument(
        "--native-abi-bootstrap-report",
        type=Path,
        help=(
            "Phase 30: bypass dense ActivitySim logsum preprocessing, compile the "
            "strict ABI from reviewed IR/raw metadata, and qualify the full resident graph"
        ),
    )
    parser.add_argument(
        "--native-abi-live",
        action="store_true",
        help=(
            "use the Phase 30 native ABI in a live/full-model performance run "
            "without retaining the six invocations for resident replay"
        ),
    )
    parser.add_argument(
        "--native-skim-store",
        type=Path,
        help=(
            "Phase 31: load all native ABI skim cubes from a versioned, "
            "byte-verified artifact and bypass Sharrow dataset materialization"
        ),
    )
    parser.add_argument(
        "--qualification-logsum-hash-report",
        type=Path,
        help=(
            "optional out-of-band qualification artifact hashing the six generated "
            "logsum vectors; the deliberate D2H copy is excluded from performance runs"
        ),
    )
    parser.add_argument("--resident-replay-runs", type=int, default=5)
    parser.add_argument(
        "--diagnostic-logsum-capture",
        type=Path,
        help="optional host capture for numeric debugging; never use for qualification",
    )
    args = parser.parse_args()
    if args.phase34_location_choice:
        args.phase33_model_wide = True
    if args.phase33_model_wide and not args.full_model:
        parser.error("--phase33-model-wide requires --full-model")
    native_abi_enabled = bool(
        args.native_abi_live or args.native_abi_bootstrap_report
    )
    if args.native_skim_store and not args.native_abi_bootstrap_report:
        parser.error("--native-skim-store requires --native-abi-bootstrap-report")
    if args.full_model and args.native_skim_store:
        parser.error(
            "a full model still needs ActivitySim skims outside mandatory scheduling; "
            "use --native-abi-live to reuse that already-resident host dataset"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    args.kernel_reports.mkdir(parents=True, exist_ok=True)
    phase17_mode_reports = args.kernel_reports / "mode"
    phase17_mode_reports.mkdir(parents=True, exist_ok=True)
    phase33_scheduling_reports = args.kernel_reports / "scheduling"
    phase33_scheduling_reports.mkdir(parents=True, exist_ok=True)
    if args.resident_replay_report:
        args.resident_replay_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_schedule_report:
        args.resident_schedule_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_generated_input_report:
        args.resident_generated_input_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_semantic_input_report:
        args.resident_semantic_input_report.parent.mkdir(parents=True, exist_ok=True)
    if args.resident_raw_table_input_report:
        args.resident_raw_table_input_report.parent.mkdir(parents=True, exist_ok=True)
    if args.native_abi_bootstrap_report:
        args.native_abi_bootstrap_report.parent.mkdir(parents=True, exist_ok=True)
    if args.qualification_logsum_hash_report:
        args.qualification_logsum_hash_report.parent.mkdir(parents=True, exist_ok=True)
    resident_requested = bool(
        args.resident_replay_report
        or args.resident_schedule_report
        or args.resident_generated_input_report
        or args.resident_semantic_input_report
        or args.resident_raw_table_input_report
        or args.native_abi_bootstrap_report
    )
    if args.diagnostic_logsum_capture:
        args.diagnostic_logsum_capture.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "CHOICEFORGE_STRICT_CUDA_CANDIDATE": "1",
            "CHOICEFORGE_STRICT_CUDA_MAX_ROWS": "2000000",
            "CHOICEFORGE_STRICT_CUDA_TILE_ROWS": "1",
            "CHOICEFORGE_STRICT_CUDA_LOCALITY": "1",
            "CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS": "0",
            "CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32": "1",
            "CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS": "1",
            "CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES": "1",
            "CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN": "1",
            # Phase 17 qualified reusable plans, but its reusable workspace
            # remained an opt-in experiment and can retain allocations across
            # model steps. Keep the stable primary full-model policy here.
            "CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS": "0" if args.full_model else "1",
            "CHOICEFORGE_STRICT_CUDA_MODE_CHOICE": "1" if args.full_model else "0",
            "CHOICEFORGE_STRICT_CUDA_TOUR_MODE_CHOICE": (
                "1" if args.phase33_model_wide else "0"
            ),
            # The strict CPU/CUDA shadow is a qualification tool, not part of
            # a production timing boundary. Full-model exactness is checked
            # out of band against every final table after the run.
            "CHOICEFORGE_STRICT_CUDA_BATCHES": "0" if args.full_model else "1000",
            "CHOICEFORGE_STRICT_CUDA_SHARROW_FMA": "1",
            "CHOICEFORGE_PHASE17_REPORT_DIR": str(args.kernel_reports.resolve()),
            "CHOICEFORGE_PHASE17_MODE_REPORT_DIR": str(
                phase17_mode_reports.resolve()
            ),
            "CHOICEFORGE_PHASE17_RUN_ID": "phase22-integrated-scheduling",
            "CHOICEFORGE_SCHEDULING_REPORT_DIR": (
                str(phase33_scheduling_reports.resolve())
                if args.phase33_model_wide else ""
            ),
            "CHOICEFORGE_SCHEDULING_RUN_ID": "phase33-model-wide",
        }
    )

    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from activitysim.core import los as activitysim_los
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

    scheduler_started = time.perf_counter()
    scheduler = IntegratedGpuMandatoryScheduler(
        args.inputs,
        # Phase 31 must not recreate Sharrow merely to adjudicate the 57
        # already-qualified scheduling ambiguities.  The sparse reference map
        # is derived from the frozen public proof artifact and stays on CUDA.
        device_boundary_reference=native_abi_enabled,
    )
    scheduler_initialization_seconds = time.perf_counter() - scheduler_started
    original_compute = vts._compute_logsums
    original_simple_simulate_logsums = simulate.simple_simulate_logsums
    original_skims_for_logsums = vts.skims_for_logsums
    original_network_los_load_skim_info = activitysim_los.Network_LOS.load_skim_info
    original_network_los_load_data = activitysim_los.Network_LOS.load_data
    original_runner_call = Runner.__call__
    original_runner_by_name = Runner.by_name
    original_activitysim_choice = vts.interaction_sample_simulate
    original_choice = activitysim_scheduling.interaction_sample_simulate_choiceforge
    diagnostic_cache_host = None
    resident_records = []
    current_raw_source = None
    current_native_manifest = None
    native_manifests = []
    native_skim_store = None
    full_model_scheduler_checkpoint = None
    native_skim_stub_calls = 0
    native_network_info_bypass_calls = 0
    native_network_load_bypass_calls = 0
    full_model_native_release_calls = 0
    full_model_native_release_seconds = 0.0
    full_model_native_release_freed_bytes = 0
    full_model_native_release_after_model = None
    raw_mode_constants = None
    raw_cbd_threshold = None
    if args.resident_raw_table_input_report or native_abi_enabled:
        import yaml

        raw_mode_settings = yaml.safe_load(
            (args.project / "configs" / "tour_mode_choice.yaml").read_text()
        )
        raw_global_settings = yaml.safe_load(
            (args.project / "configs" / "settings.yaml").read_text()
        )
        raw_mode_constants = raw_mode_settings["CONSTANTS"]
        raw_cbd_threshold = raw_global_settings["cbd_threshold"]

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
                "raw_source": current_raw_source,
                "native_abi_manifest": current_native_manifest,
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
        nonlocal current_raw_source, current_native_manifest, native_skim_store
        original_simple = simulate.simple_simulate_logsums

        if native_abi_enabled:
            native_started = time.perf_counter()
            if len(compute_args) < 8:
                raise RuntimeError("Phase 30 requires the positional ActivitySim logsum ABI")
            (
                raw_state, raw_alt_tdd, raw_tours, raw_purpose, model_settings,
                _network_los, raw_skims, trace_label,
            ) = compute_args[:8]
            from activitysim.abm.models.tour_mode_choice import TourModeComponentSettings
            from activitysim.core import config
            from choiceforge.cuda_skims import cuda_cube_from_activitysim
            from choiceforge.native_abi_bootstrap import (
                NativeSkimCube,
                compile_native_strict_abi,
            )
            from choiceforge.native_skim_store import NativeSkimStore
            from choiceforge.raw_table_input_generation import ResidentRawTableInputPlan
            from choiceforge.sharrow_ir import specification_ir

            logsum_settings = TourModeComponentSettings.read_settings_file(
                raw_state.filesystem,
                str(model_settings.LOGSUM_SETTINGS),
                mandatory=False,
            )
            constants = config.get_model_constants(logsum_settings)
            coefficients = raw_state.filesystem.get_segment_coefficients(
                logsum_settings, raw_purpose
            )
            logsum_spec = raw_state.filesystem.read_model_spec(
                file_name=logsum_settings.SPEC
            )
            logsum_spec = simulate.eval_coefficients(
                raw_state, logsum_spec, coefficients, estimator=None
            )
            numeric_nest = config.get_logit_model_settings(logsum_settings)
            numeric_nest = simulate.eval_nest_coefficients(
                numeric_nest, coefficients, trace_label
            )
            document = specification_ir(logsum_spec.reset_index())
            scalar_environment = raw_state.get_global_constants().copy()
            scalar_environment.update(constants)
            scalar_environment.update(coefficients)

            def cube_loader(source):
                nonlocal native_skim_store
                if args.native_skim_store:
                    if native_skim_store is None:
                        live_zone_ids = np.asarray(
                            raw_state.get_dataframe("land_use").index,
                            dtype=np.int64,
                        )
                        if np.array_equal(
                            live_zone_ids,
                            np.arange(live_zone_ids.size, dtype=np.int64),
                        ):
                            # ActivitySim recodes the public MTC TAZ identity
                            # 1..1454 to zero-based row positions in its saved
                            # pipeline.  Reconstitute that proven positional
                            # mapping before checking the store's source-ID hash.
                            native_zone_ids = live_zone_ids + 1
                        else:
                            native_zone_ids = live_zone_ids
                        native_skim_store = NativeSkimStore.load(
                            args.native_skim_store,
                            document,
                            native_zone_ids,
                            budget_bytes=8 * 1024**3,
                        )
                    data, dest_count, time_count, rank = native_skim_store.cube(
                        source
                    )
                    return NativeSkimCube(data, dest_count, time_count, rank)
                _, direction, key = source
                wrapper_name = "od_skims" if direction == "od_skims_reverse" else direction
                if wrapper_name not in raw_skims:
                    raise ValueError(f"Phase 30 skim direction {direction!r} is absent")
                data, dest_count, time_count, rank = cuda_cube_from_activitysim(
                    raw_skims[wrapper_name], key
                )
                return NativeSkimCube(data, dest_count, time_count, rank)

            metadata = {
                "trace_label": str(trace_label) + ".native_abi",
                "chooser_ids": np.asarray(raw_alt_tdd.index, dtype=np.int64),
                "start": np.asarray(raw_alt_tdd["start"], dtype=np.int16),
                "end": np.asarray(raw_alt_tdd["end"], dtype=np.int16),
                "out_period": np.asarray(raw_alt_tdd["out_period"].astype(str)),
                "in_period": np.asarray(raw_alt_tdd["in_period"].astype(str)),
            }
            rng = raw_state.get_rn_generator()
            draws = rng.normal_for_df(raw_alt_tdd, broadcast=True, size=6)
            draws = draws.to_numpy(copy=False) if hasattr(draws, "to_numpy") else np.asarray(draws)
            first = ~raw_alt_tdd.index.duplicated(keep="first")
            current_raw_source = {
                "tours": raw_tours.copy(),
                "land_use": raw_state.get_dataframe("land_use")[[
                    "TOTPOP", "TOTEMP", "TOTACRE", "PRKCST", "area_type",
                    "TOPOLOGY", "TERMINAL", "density_index",
                ]].copy(),
                "tour_purpose": str(raw_purpose),
                "constants": constants,
                "cbd_threshold": raw_cbd_threshold,
                "standard_normal_draws": np.asarray(draws[first], dtype=np.float64).copy(),
            }
            native = compile_native_strict_abi(
                document, scalar_environment, cube_loader, rows=len(raw_alt_tdd)
            )
            current_native_manifest = {
                **native.manifest,
                "purpose": str(raw_purpose),
                "rows": int(len(raw_alt_tdd)),
                "dense_preprocessor_rows_avoided": int(len(raw_alt_tdd)),
            }
            native_manifests.append(current_native_manifest)
            immediate_plan = ResidentRawTableInputPlan.compile(
                native.invocation, metadata, current_raw_source,
                validate_oracle=False,
            )
            immediate_plan.execute()
            utilities = immediate_plan.invocation.execute()
            logsums = mtc21_nested_logsums_cuda(
                utilities,
                numeric_nest,
                document["alternatives"],
                return_device=True,
                numeric_policy="activitysim_pandas_float64",
            )
            current_native_manifest["bootstrap_seconds"] = (
                time.perf_counter() - native_started
            )
            device_logsum_sink(logsums, metadata)
            resident_invocation_sink(
                immediate_plan.invocation,
                numeric_nest,
                tuple(document["alternatives"]),
                metadata,
                logsums,
            )
            current_raw_source = None
            current_native_manifest = None
            # The integrated GPU cache is authoritative. ActivitySim only needs
            # an index-aligned placeholder for its legacy alternatives table.
            return pd.Series(
                np.zeros(len(raw_alt_tdd), dtype=np.float64), index=raw_alt_tdd.index
            )

        rng = None
        original_normal_for_df = None
        if args.resident_raw_table_input_report:
            if len(compute_args) < 4:
                raise RuntimeError("Phase 29 requires the positional ActivitySim logsum ABI")
            raw_state, raw_alt_tdd, raw_tours, raw_purpose = compute_args[:4]
            land_columns = [
                "TOTPOP", "TOTEMP", "TOTACRE", "PRKCST", "area_type",
                "TOPOLOGY", "TERMINAL", "density_index",
            ]
            current_raw_source = {
                "tours": raw_tours.copy(),
                "land_use": raw_state.get_dataframe("land_use")[land_columns].copy(),
                "tour_purpose": str(raw_purpose),
                "constants": raw_mode_constants,
                "cbd_threshold": raw_cbd_threshold,
            }
            rng = raw_state.get_rn_generator()
            original_normal_for_df = rng.normal_for_df

            def capture_normal_for_df(df, *normal_args, **normal_kwargs):
                result = original_normal_for_df(df, *normal_args, **normal_kwargs)
                if (
                    bool(normal_kwargs.get("broadcast"))
                    and int(normal_kwargs.get("size") or 0) == 6
                    and len(df) == len(raw_alt_tdd)
                    and np.array_equal(np.asarray(df.index), np.asarray(raw_alt_tdd.index))
                ):
                    array = (
                        result.to_numpy(copy=False)
                        if hasattr(result, "to_numpy") else np.asarray(result)
                    )
                    if array.shape != (len(df), 6):
                        raise RuntimeError(
                            f"Phase 29 controlled draw matrix has shape {array.shape}"
                        )
                    first = ~df.index.duplicated(keep="first")
                    current_raw_source["standard_normal_draws"] = np.asarray(
                        array[first], dtype=np.float64
                    ).copy()
                return result

            rng.normal_for_df = capture_normal_for_df

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
            if rng is not None:
                rng.normal_for_df = original_normal_for_df
            current_raw_source = None

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
        if not args.full_model and isinstance(models, list) and resume_after in models:
            checkpoint = models.index(resume_after)
            models = models[: checkpoint + 2]
        return original_runner_call(
            self,
            models,
            resume_after=resume_after,
            memory_sidecar_process=memory_sidecar_process,
        )

    def run_full_model_step(self, model_name):
        nonlocal full_model_native_release_calls, full_model_native_release_seconds
        nonlocal full_model_native_release_freed_bytes
        nonlocal full_model_native_release_after_model
        nonlocal full_model_scheduler_checkpoint
        model_name_text = str(model_name)
        location_logsum_steps = {
            "school_location",
            "workplace_location",
            "joint_tour_destination",
            "non_mandatory_tour_destination",
            "atwork_subtour_destination",
        }
        if (
            args.phase33_model_wide
            and model_name_text == "tour_mode_choice_simulate"
            and os.environ.get("CHOICEFORGE_STRICT_CUDA_TOUR_MODE_CHOICE", "0") == "1"
        ):
            from choiceforge.activitysim_mode_choice import (
                install_activitysim_tour_mode_candidate,
            )

            install_activitysim_tour_mode_candidate()
        if args.phase34_location_choice and model_name_text == "atwork_subtour_mode_choice":
            from choiceforge.activitysim_mode_choice import (
                install_activitysim_atwork_mode_candidate,
            )

            install_activitysim_atwork_mode_candidate()
        location_candidate_enabled = (
            model_name_text == "non_mandatory_tour_destination"
            or (
                args.phase34_location_choice
                and model_name_text in location_logsum_steps
            )
        )
        if args.phase33_model_wide and location_candidate_enabled:
            def generated_location_logsums(
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
                simulate.simple_simulate_logsums = original_simple_simulate_logsums
                try:
                    return _simple_simulate_mtc21_logsums_cuda(
                        state,
                        choosers,
                        spec,
                        nest_spec,
                        skims or {},
                        locals_d or {},
                        trace_label
                        or f"{model_name_text}.compute_logsums",
                        explicit_chunk_size,
                    )
                finally:
                    simulate.simple_simulate_logsums = generated_location_logsums

            simulate.simple_simulate_logsums = generated_location_logsums
        try:
            result = original_runner_by_name(self, model_name)
        finally:
            if args.phase33_model_wide and location_candidate_enabled:
                simulate.simple_simulate_logsums = original_simple_simulate_logsums
        if args.full_model and model_name_text == "mandatory_tour_scheduling":
            full_model_scheduler_checkpoint = scheduler.checkpoint()
            # The scheduling-only hooks must end here, but its immutable CUDA
            # skim cubes are also inputs to trip destination and trip mode.
            # Retain those cubes through the final GPU skim consumer so the
            # full model pays one upload instead of a 6+ GB rebuild.
            vts._compute_logsums = original_compute
            vts.interaction_sample_simulate = original_activitysim_choice
            activitysim_scheduling.interaction_sample_simulate_choiceforge = original_choice
        if args.full_model and model_name_text == "trip_mode_choice":
            from choiceforge.cuda_backend import _cupy
            from choiceforge.cuda_skims import clear_cuda_dataset_cache

            cp = _cupy()
            cp.cuda.Stream.null.synchronize()
            before = int(cp.get_default_memory_pool().used_bytes())
            release_started = time.perf_counter()
            clear_cuda_dataset_cache()
            cp.cuda.Stream.null.synchronize()
            after = int(cp.get_default_memory_pool().used_bytes())
            full_model_native_release_calls += 1
            full_model_native_release_seconds += time.perf_counter() - release_started
            full_model_native_release_freed_bytes += max(0, before - after)
            full_model_native_release_after_model = str(model_name)
        return result

    def native_skims_for_logsums(
        state, tour_purpose, model_settings, trace_label
    ):
        nonlocal native_skim_stub_calls
        native_skim_stub_calls += 1
        destination = model_settings.DESTINATION_FOR_TOUR_PURPOSE
        if isinstance(destination, dict):
            destination = destination.get(tour_purpose)
        if not isinstance(destination, str):
            raise TypeError(
                f"Phase 31 has no destination field for purpose {tour_purpose!r}"
            )
        return {
            "choiceforge_native_skim_store": True,
            "orig_col_name": "home_zone_id",
            "dest_col_name": destination,
        }

    def native_network_los_load_data(self):
        nonlocal native_network_load_bypass_calls
        if self.zone_system != activitysim_los.ONE_ZONE:
            raise RuntimeError("Phase 31 native skim bypass requires a one-zone model")
        native_network_load_bypass_calls += 1
        # Period labeling below needs only the already validated network
        # settings.  The native CUDA store owns all skim values for this model
        # step, so constructing an ActivitySim SkimDataset would be redundant.
        self.skim_dicts.clear()

    def native_network_los_load_skim_info(self):
        nonlocal native_network_info_bypass_calls
        if self.zone_system != activitysim_los.ONE_ZONE:
            raise RuntimeError("Phase 31 native skim bypass requires a one-zone model")
        native_network_info_bypass_calls += 1
        # Network_LOS.__init__ normally opens every OMX file here merely to
        # inventory its matrices.  The signed native-store manifest is the
        # authoritative inventory for Phase 31, so keep the legacy map empty.
        self.skims_info.clear()

    vts._compute_logsums = gpu_compute_logsums
    if args.native_skim_store:
        vts.skims_for_logsums = native_skims_for_logsums
        activitysim_los.Network_LOS.load_skim_info = native_network_los_load_skim_info
        activitysim_los.Network_LOS.load_data = native_network_los_load_data
    # The public MTC configuration selects ActivitySim's named choice backend.
    # Patch both dispatch targets so Phase 22 remains correct if a configuration
    # switches to the ChoiceForge backend without changing this runner.
    vts.interaction_sample_simulate = integrated_choice
    activitysim_scheduling.interaction_sample_simulate_choiceforge = integrated_choice
    Runner.__call__ = run_one_model
    if args.full_model:
        Runner.by_name = run_full_model_step
    from activitysim.cli import main as activitysim_main

    cli = ["activitysim", "run"]
    for overlay in args.config_overlay or []:
        cli.extend(["-c", str(overlay.resolve())])
    cli.extend([
        "-c", str((args.project / "configs").resolve()),
        "-d", str(args.data.resolve()),
        "-o", str(args.output.resolve()),
    ])
    if args.full_model:
        cli.extend(["--households_sample_size", str(args.households_sample_size)])
    else:
        cli.extend(["-r", args.resume])
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
        vts.skims_for_logsums = original_skims_for_logsums
        activitysim_los.Network_LOS.load_skim_info = original_network_los_load_skim_info
        activitysim_los.Network_LOS.load_data = original_network_los_load_data
        vts.interaction_sample_simulate = original_activitysim_choice
        simulate.simple_simulate_logsums = original_simple_simulate_logsums
        activitysim_scheduling.interaction_sample_simulate_choiceforge = original_choice
        Runner.__call__ = original_runner_call
        Runner.by_name = original_runner_by_name

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
    kernel_report_paths = sorted(args.kernel_reports.rglob("*.json"))
    kernel_reports = [json.loads(path.read_text()) for path in kernel_report_paths]
    candidates = [item for item in kernel_reports if item.get("candidate_used")]
    fallbacks = [item for item in kernel_reports if item.get("fallback_used")]
    phase33_destination = [
        item for item in candidates
        if str(item.get("trace_label", "")).startswith(
            "non_mandatory_tour_destination."
        )
    ]
    phase33_scheduling = [
        item for item in candidates if item.get("component") == "tour_scheduling"
    ]
    phase33_tour_mode = [
        item for item in candidates if item.get("component") == "tour_mode_choice"
    ]
    phase34_location = [
        item for item in candidates
        if any(
            str(item.get("trace_label", "")).startswith(f"{prefix}.")
            for prefix in (
                "school_location",
                "workplace_location",
                "joint_tour_destination",
                "atwork_subtour_destination",
            )
        )
    ]
    phase34_location_groups = {
        prefix: [
            item for item in phase34_location
            if str(item.get("trace_label", "")).startswith(f"{prefix}.")
        ]
        for prefix in (
            "school_location",
            "workplace_location",
            "joint_tour_destination",
            "atwork_subtour_destination",
        )
    }
    phase34_atwork_mode = [
        item
        for item in candidates
        if item.get("component") == "atwork_subtour_mode_choice"
    ]
    timing_path = args.output / "timing_log.csv"
    model_timing_seconds = {}
    if timing_path.exists():
        timing_frame = pd.read_csv(timing_path)
        model_timing_seconds = {
            str(row.model_name): float(row.seconds)
            for row in timing_frame.itertuples(index=False)
        }
    report_candidate_rows = int(sum(item.get("rows", 0) for item in candidates))
    if native_abi_enabled:
        report_candidate_rows = int(sum(item["invocation"].rows for item in resident_records))
        if not resident_records:
            report_candidate_rows = int(sum(item["rows"] for item in native_manifests))
    checkpoint = full_model_scheduler_checkpoint
    if checkpoint is None and scheduler.complete:
        checkpoint = scheduler.checkpoint()
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
            or args.resident_semantic_input_report is not None
            or args.resident_raw_table_input_report is not None
            or args.native_abi_bootstrap_report is not None
        ):
            from choiceforge.device_resident_runtime import DeviceResidentRuntime

            phase27_plans = []
            phase27_cpu_seconds = []
            phase27_gpu_seconds = []
            original_captured_pointers = set()
            if (
                args.resident_generated_input_report is not None
                or args.resident_semantic_input_report is not None
                or args.resident_raw_table_input_report is not None
                or args.native_abi_bootstrap_report is not None
            ):
                from choiceforge.device_input_expansion import (
                    ResidentInputExpansionPlan,
                    ResidentSemanticInputPlan,
                )
                if (
                    args.resident_raw_table_input_report is not None
                    or args.native_abi_bootstrap_report is not None
                ):
                    from choiceforge.raw_table_input_generation import (
                        ResidentRawTableInputPlan,
                    )

                plan_type = (
                    ResidentRawTableInputPlan
                    if (
                        args.resident_raw_table_input_report is not None
                        or args.native_abi_bootstrap_report is not None
                    )
                    else (
                        ResidentSemanticInputPlan
                        if args.resident_semantic_input_report is not None
                        else ResidentInputExpansionPlan
                    )
                )

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
                    if (
                        args.resident_raw_table_input_report is not None
                        or args.native_abi_bootstrap_report is not None
                    ):
                        if record["raw_source"] is None:
                            raise RuntimeError("Phase 29 did not capture a raw source bundle")
                        plan = plan_type.compile(
                            original,
                            record["metadata"],
                            record["raw_source"],
                            validate_oracle=(
                                args.native_abi_bootstrap_report is None
                            ),
                        )
                    else:
                        plan = plan_type.compile(original, record["metadata"])
                    phase27_plans.append(plan)
                    record["invocation"] = plan.invocation

                # Matched boundary: both baselines materialize the same arrays
                # from the same compact factors. CPU timings deliberately do
                # not include the one-time factor download or qualification.
                if (
                    args.resident_semantic_input_report is None
                    and args.resident_raw_table_input_report is None
                    and args.native_abi_bootstrap_report is None
                ):
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
                if plan.semantic_program is not None:
                    for label, value in (
                        ("semantic_start", plan.semantic_program.slot_start),
                        ("semantic_end", plan.semantic_program.slot_end),
                        ("semantic_parking_rates", plan.semantic_program.parking_rates),
                    ):
                        register_asset(f"input_plan_{number}_{label}", value)
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

            stage_phase = (
                31 if args.native_skim_store is not None
                else 30 if args.native_abi_bootstrap_report is not None
                else 29 if args.resident_raw_table_input_report is not None
                else 28 if phase27_plans and phase27_plans[0].semantic_program is not None
                else 27 if phase27_plans else 26
            )
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
                        "expansion_cpu_median_seconds": (
                            float(np.median(phase27_cpu_seconds))
                            if phase27_cpu_seconds else None
                        ),
                        "expansion_speedup_cpu_over_gpu": (
                            float(np.median(phase27_cpu_seconds))
                            / float(np.median(phase27_gpu_seconds))
                            if phase27_cpu_seconds else None
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
                            if phase26_report["expansion_speedup_cpu_over_gpu"] is not None
                            else True
                        ),
                    }
                )
                if stage_phase in {28, 29, 30, 31}:
                    manifests = [
                        plan.semantic_program.manifest() for plan in phase27_plans
                    ]
                    phase26_report["semantic_input_programs"] = manifests
                    phase26_report["input_contract"] = (
                        "every chooser-response input is regenerated by a named CUDA "
                        "formula from compact chooser/alternative state and resident raw skims; "
                        "anonymous response dictionaries are absent"
                    )
                    phase26_report["proof_gates"].update(
                        {
                            "all_response_columns_semantically_generated": all(
                                item["generated_float_columns"]
                                + item["generated_int_columns"] > 0
                                for item in manifests
                            ),
                            "zero_anonymous_response_patterns": all(
                                item["anonymous_response_pattern_columns"] == 0
                                for item in manifests
                            ),
                        }
                    )
                if stage_phase in {29, 30, 31}:
                    raw_manifests = [plan.raw_manifest() for plan in phase27_plans]
                    phase26_report["raw_table_input_programs"] = raw_manifests
                    phase26_report["input_contract"] = (
                        "every strict dense input and skim coordinate is declared from "
                        "one-row-per-tour ActivitySim tables, land use, controlled random "
                        "draws, alternative slots, and resident raw skims; dense preprocessor "
                        "rows are an oracle only and are not read by the compiler"
                    )
                    phase26_report["proof_gates"].update(
                        {
                            "zero_dense_oracle_bytes_read_for_compile": all(
                                item["dense_oracle_bytes_read_for_compile"] == 0
                                for item in raw_manifests
                            ),
                            "all_raw_sources_declared": all(
                                item["source_count"] > 0 for item in raw_manifests
                            ),
                            "all_eighteen_availability_formulas_generated": all(
                                item["availability_formulas"] == 18
                                for item in raw_manifests
                            ),
                            "direct_land_use_parking_rate": all(
                                item["parking_rate_source"]
                                == "land_use.PRKCST_or_free_parking_at_work"
                                for item in raw_manifests
                            ),
                        }
                    )
                if stage_phase in {30, 31}:
                    phase26_report["native_abi_programs"] = native_manifests
                    phase26_report["native_bootstrap_seconds"] = float(
                        sum(item["bootstrap_seconds"] for item in native_manifests)
                    )
                    phase26_report["dense_preprocessor_rows_avoided"] = int(
                        sum(
                            item["dense_preprocessor_rows_avoided"]
                            for item in native_manifests
                        )
                    )
                    phase26_report["scheduling_arithmetic_contract"] = {
                        "utility_dot": "sharrow65_four_lane_float32",
                        "probability_sum": "numpy_pairwise_float32",
                        "probability_search": "source_order_float32_inverse_cdf",
                        "exponential": "cuda_libdevice_expf",
                        "remaining_cross_library_ambiguity_entries": int(
                            resident_scheduler.boundary_map_entries
                        ),
                    }
                    phase26_report["input_contract"] = (
                        "reviewed hashed utility IR, named raw-table sources, scalar "
                        "settings, and immutable raw skim metadata compile the strict "
                        "CUDA ABI without joining dense chooser-alternative rows or "
                        "executing ActivitySim's logsum preprocessor"
                    )
                    phase26_report["proof_gates"].update(
                        {
                            "all_native_abi_programs_declared": (
                                len(native_manifests) == 6
                                and all(item["bindings"] for item in native_manifests)
                            ),
                            "zero_dense_preprocessor_rows_read": all(
                                item["dense_preprocessor_rows_read"] == 0
                                for item in native_manifests
                            ),
                            "zero_dense_preprocessor_values_read": all(
                                item["dense_preprocessor_values_read"] == 0
                                for item in native_manifests
                            ),
                            "all_programs_share_reviewed_codegen": all(
                                item["codegen"] == native_manifests[0]["codegen"]
                                for item in native_manifests
                            ),
                            "three_purpose_schemas_repeat_exactly": (
                                len({
                                    (item["purpose"], item["schema_sha256"])
                                    for item in native_manifests
                                }) == 3
                            ),
                            "all_public_dense_preprocessor_rows_avoided": (
                                phase26_report["dense_preprocessor_rows_avoided"]
                                == phase26_report["mode_logsum_rows"]
                            ),
                        }
                    )
                if stage_phase == 31:
                    if native_skim_store is None:
                        raise RuntimeError("Phase 31 native skim store was never loaded")
                    store_telemetry = native_skim_store.telemetry_dict()
                    phase26_report["native_skim_store"] = store_telemetry
                    phase26_report["input_contract"] = (
                        "a versioned, byte-verified 149-cube native artifact supplies "
                        "all 209 logical skim bindings directly to CUDA; ActivitySim's "
                        "6.452 GB Sharrow dataset is never materialized"
                    )
                    phase26_report["proof_gates"].update(
                        {
                            "native_store_all_payload_bytes_verified": (
                                store_telemetry["verified_payload_bytes"]
                                == store_telemetry["payload_bytes"]
                            ),
                            "native_store_has_209_logical_bindings": (
                                store_telemetry["logical_bindings"] == 209
                            ),
                            "native_store_has_149_physical_cubes": (
                                store_telemetry["physical_cubes"] == 149
                            ),
                            "all_six_sharrow_skim_requests_bypassed": (
                                native_skim_stub_calls == 6
                            ),
                            "activitysim_network_skim_inventory_bypassed_once": (
                                native_network_info_bypass_calls == 1
                            ),
                            "activitysim_network_data_load_bypassed_once": (
                                native_network_load_bypass_calls == 1
                            ),
                        }
                    )
            resident_stage_report_path = (
                (
                    args.native_abi_bootstrap_report
                    or args.resident_raw_table_input_report
                    or args.resident_semantic_input_report
                    or args.resident_generated_input_report
                )
                if phase27_plans
                else args.resident_schedule_report
            )
            resident_stage_report_path.write_text(
                json.dumps(phase26_report, indent=2) + "\n"
            )
            if args.qualification_logsum_hash_report:
                logsum_hashes = []
                for number, record in enumerate(resident_records):
                    host_logsums = np.ascontiguousarray(
                        cp.asnumpy(record["reference_logsums"])
                    )
                    logsum_hashes.append(
                        {
                            "batch": number,
                            "trace_label": str(record["metadata"]["trace_label"]),
                            "rows": int(host_logsums.size),
                            "dtype": str(host_logsums.dtype),
                            "sha256": hashlib.sha256(
                                host_logsums.view(np.uint8)
                            ).hexdigest(),
                        }
                    )
                hash_document = {
                    "phase": 31 if args.native_skim_store else 30,
                    "scope": (
                        "out-of-band exact qualification only; these deliberate "
                        "device-to-host copies are not part of any timed resident run"
                    ),
                    "bootstrap": (
                        "native_abi"
                        if args.native_abi_bootstrap_report
                        else "activitysim_dense_oracle"
                    ),
                    "programs": logsum_hashes,
                    "aggregate_sha256": hashlib.sha256(
                        "\n".join(item["sha256"] for item in logsum_hashes).encode(
                            "ascii"
                        )
                    ).hexdigest(),
                }
                args.qualification_logsum_hash_report.write_text(
                    json.dumps(hash_document, indent=2) + "\n"
                )
            if not all(phase26_report["proof_gates"].values()):
                raise RuntimeError(
                    f"Phase {stage_phase} resident schedule proof gate failed"
                )

    batch_telemetry = [asdict(item) for item in scheduler.telemetry]
    report = {
        "phase": (
            34 if args.phase34_location_choice else
            33 if args.phase33_model_wide else
            (32 if args.full_model else 22)
        ),
        "scope": (
            "full public ActivitySim model with the Phase 33 runtime plus "
            "school, workplace, joint-tour, and at-work location logsum CUDA programs"
            if args.phase34_location_choice else
            "full public ActivitySim model with six qualified GPU consumers and "
            "one shared CUDA skim residency interval through trip mode choice"
            if args.phase33_model_wide else
            "full public ActivitySim model with native mandatory scheduling and "
            "one shared CUDA skim residency interval through trip mode choice"
            if args.full_model else
            "live ActivitySim raw network skims through generated CUDA utility, "
            "CUDA nesting, device cache scatter, timetable preparation, choice, "
            "and timetable mutation"
        ),
        "elapsed_seconds_including_resume_overhead": elapsed,
        "exit_code": int(exit_code or 0),
        "mandatory_tours": int(len(expected)),
        "tdd_mismatches": int(np.count_nonzero(actual.tdd.to_numpy() != expected.tdd.to_numpy())),
        "start_mismatches": int(np.count_nonzero(actual.start.to_numpy() != expected.start.to_numpy())),
        "end_mismatches": int(np.count_nonzero(actual.end.to_numpy() != expected.end.to_numpy())),
        "candidate_calls": (
            len(native_manifests) if native_abi_enabled else len(candidates)
        ),
        "fallback_calls": len(fallbacks),
        "phase33_non_mandatory_destination_cuda_calls": len(phase33_destination),
        "phase33_non_mandatory_destination_rows": int(
            sum(item.get("rows", 0) for item in phase33_destination)
        ),
        "phase33_non_mandatory_scheduling_cuda_calls": len(phase33_scheduling),
        "phase33_non_mandatory_scheduling_choosers": int(
            sum(item.get("choosers", 0) for item in phase33_scheduling)
        ),
        "phase33_non_mandatory_scheduling_alternative_rows": int(
            sum(item.get("alternative_rows", 0) for item in phase33_scheduling)
        ),
        "phase33_tour_mode_cuda_calls": len(phase33_tour_mode),
        "phase33_tour_mode_rows": int(
            sum(item.get("rows", 0) for item in phase33_tour_mode)
        ),
        "phase34_location_cuda_calls": len(phase34_location),
        "phase34_location_rows": int(
            sum(item.get("rows", 0) for item in phase34_location)
        ),
        "phase34_location_trace_labels": [
            str(item.get("trace_label", "")) for item in phase34_location
        ],
        "phase34_location_groups": {
            prefix: {
                "cuda_calls": len(items),
                "rows": int(sum(item.get("rows", 0) for item in items)),
            }
            for prefix, items in phase34_location_groups.items()
        },
        "phase34_atwork_mode_cuda_calls": len(phase34_atwork_mode),
        "phase34_atwork_mode_rows": int(
            sum(item.get("rows", 0) for item in phase34_atwork_mode)
        ),
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
        "bulk_modeled_logsum_device_to_host_bytes": (
            0 if native_abi_enabled else int(
                sum(item.get("logsum_device_to_host_bytes", -1) for item in candidates)
            )
        ),
        "final_tdd_device_to_host_bytes": (
            int(checkpoint["final_tdd_device_to_host_bytes"]) if checkpoint else None
        ),
        "cache_build_ms": float(sum(x["cache_build_ms"] for x in batch_telemetry)),
        "scheduling_ms": float(sum(x["scheduling_ms"] for x in batch_telemetry)),
        "batch_telemetry": batch_telemetry,
        "kernel_reports": [
            path.relative_to(args.kernel_reports).as_posix()
            for path in kernel_report_paths
        ],
        "native_abi_bootstrap_used": native_abi_enabled,
        "native_abi_live_only": bool(args.native_abi_live),
        "native_abi_programs": len(native_manifests),
        "native_skim_store_used": bool(args.native_skim_store),
        "scheduler_initialization_seconds": scheduler_initialization_seconds,
        "cold_component_seconds_including_scheduler_initialization": (
            elapsed + scheduler_initialization_seconds
        ),
        "native_skim_stub_calls": int(native_skim_stub_calls),
        "native_network_info_bypass_calls": int(native_network_info_bypass_calls),
        "native_network_load_bypass_calls": int(native_network_load_bypass_calls),
        "native_skim_store_load_seconds": (
            native_skim_store.telemetry.total_load_seconds
            if native_skim_store is not None else None
        ),
        "device_boundary_map_entries": int(scheduler.boundary_map_entries),
        "device_boundary_adjudications": int(
            sum(item.device_boundary_adjudications for item in scheduler.telemetry)
        ),
        "full_model": bool(args.full_model),
        "households_sample_size": (
            int(args.households_sample_size) if args.full_model else None
        ),
        "full_model_native_release_calls": int(full_model_native_release_calls),
        "full_model_native_release_seconds": float(full_model_native_release_seconds),
        "full_model_native_release_freed_bytes": int(
            full_model_native_release_freed_bytes
        ),
        "full_model_native_release_after_model": (
            full_model_native_release_after_model
        ),
        "model_timing_seconds": model_timing_seconds,
        "activitysim_all_model_steps_seconds": float(
            sum(model_timing_seconds.values())
        ),
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
    if args.native_skim_store:
        report["proof_gates"].update(
            {
                "sharrow_skim_materialization_bypassed": native_skim_stub_calls == 6,
                "activitysim_network_skim_inventory_bypassed": (
                    native_network_info_bypass_calls == 1
                ),
                "activitysim_network_data_load_bypassed": (
                    native_network_load_bypass_calls == 1
                ),
                "native_skim_store_loaded": native_skim_store is not None,
                "all_57_boundary_choices_adjudicated_on_device": (
                    scheduler.boundary_map_entries >= 57
                    and sum(
                        item.device_boundary_adjudications
                        for item in scheduler.telemetry
                    ) == 57
                ),
                "zero_boundary_logsum_download": (
                    sum(
                        item.boundary_logsum_download_bytes
                        for item in scheduler.telemetry
                    ) == 0
                ),
            }
        )
    if args.full_model:
        report["proof_gates"].update(
            {
                "all_34_model_steps_timed": len(model_timing_seconds) == 34,
                "native_cuda_skims_released_after_last_gpu_consumer_once": (
                    full_model_native_release_calls == 1
                    and full_model_native_release_freed_bytes > 5_000_000_000
                    and full_model_native_release_after_model == "trip_mode_choice"
                ),
                "all_57_boundary_choices_adjudicated_on_device": (
                    scheduler.boundary_map_entries >= 57
                    and report["device_boundary_adjudications"] == 57
                ),
                "zero_boundary_logsum_download": (
                    report["boundary_logsum_download_bytes"] == 0
                ),
            }
        )
    if args.phase33_model_wide:
        report["proof_gates"].update(
            {
                "all_6_non_mandatory_destination_cuda_calls_used": (
                    report["phase33_non_mandatory_destination_cuda_calls"] == 6
                    and report["phase33_non_mandatory_destination_rows"] > 0
                ),
                "all_7_non_mandatory_scheduling_cuda_calls_used": (
                    report["phase33_non_mandatory_scheduling_cuda_calls"] == 7
                    and report["phase33_non_mandatory_scheduling_choosers"] > 0
                    and report["phase33_non_mandatory_scheduling_alternative_rows"] > 0
                ),
                "all_9_primary_tour_mode_cuda_calls_used": (
                    report["phase33_tour_mode_cuda_calls"] == 9
                    and report["phase33_tour_mode_rows"] > 0
                ),
            }
        )
    if args.phase34_location_choice:
        report["proof_gates"].update(
            {
                "all_13_phase34_location_cuda_programs_used": (
                    report["phase34_location_cuda_calls"] == 13
                    and report["phase34_location_rows"] == 2_932_524
                ),
                "phase34_location_workload_shape_exact": (
                    report["phase34_location_groups"]
                    == {
                        "school_location": {"cuda_calls": 3, "rows": 685_915},
                        "workplace_location": {"cuda_calls": 4, "rows": 1_859_082},
                        "joint_tour_destination": {"cuda_calls": 5, "rows": 76_559},
                        "atwork_subtour_destination": {"cuda_calls": 1, "rows": 310_968},
                    }
                ),
                "atwork_mode_cuda_program_used": (
                    report["phase34_atwork_mode_cuda_calls"] == 1
                    and report["phase34_atwork_mode_rows"] > 0
                ),
            }
        )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    resident_ok = (
        resident_report is None
        or all(resident_report["proof_gates"].values())
    )
    return 0 if all(report["proof_gates"].values()) and resident_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
