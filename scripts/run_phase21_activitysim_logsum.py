"""Run mandatory scheduling with its raw-skim mode-choice logsums on CUDA.

This is the live integration companion to the standalone Phase 21 preparation
benchmark.  ActivitySim remains responsible for orchestration and dataframe
assembly, while ChoiceForge replaces the 21-alternative utility compiler and
nested-logit reduction used to construct the compact scheduling logsum cache.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kernel-reports", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=("cuda", "cpu"), default="cuda",
        help="run the CUDA candidate or an unmodified ActivitySim/Sharrow control",
    )
    parser.add_argument(
        "--logsum-capture",
        type=Path,
        help="optional directory for exact per-call logsum diagnostics",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.kernel_reports.mkdir(parents=True, exist_ok=True)
    if args.logsum_capture:
        args.logsum_capture.mkdir(parents=True, exist_ok=True)

    if args.engine == "cuda":
        os.environ.update({
            "CHOICEFORGE_STRICT_CUDA_CANDIDATE": "1",
            "CHOICEFORGE_STRICT_CUDA_MAX_ROWS": "2000000",
            "CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32": "1",
            "CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS": "1",
            "CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES": "1",
            "CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN": "1",
            "CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS": "1",
            "CHOICEFORGE_STRICT_CUDA_SHARROW_FMA": "1",
            "CHOICEFORGE_PHASE17_REPORT_DIR": str(args.kernel_reports.resolve()),
            "CHOICEFORGE_PHASE17_RUN_ID": "phase21-scheduling-logsum",
        })

    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from activitysim.core import simulate
    from activitysim.core.workflow.runner import Runner
    from choiceforge.activitysim_destination import _simple_simulate_mtc21_logsums_cuda

    original_compute = vts._compute_logsums
    original_runner_call = Runner.__call__
    logsum_captures = []

    def capture_logsums(compute_args, result):
        if not args.logsum_capture:
            return
        alt_tdd = compute_args[1]
        logsum_captures.append(
            {
                "purpose": str(compute_args[3]),
                "trace_label": str(compute_args[7]),
                "chooser_ids": np.asarray(alt_tdd.index, dtype=np.int64),
                "start": np.asarray(alt_tdd["start"], dtype=np.int16),
                "end": np.asarray(alt_tdd["end"], dtype=np.int16),
                "logsums": np.asarray(result, dtype=np.float64),
            }
        )

    def gpu_compute_logsums(*compute_args, **compute_kwargs):
        if args.engine == "cpu":
            result = original_compute(*compute_args, **compute_kwargs)
            capture_logsums(compute_args, result)
            return result
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
            # The candidate helper invokes the authoritative function after
            # installing its generated-utility and nested-reduction hooks.
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
                )
            finally:
                simulate.simple_simulate_logsums = gpu_simple

        simulate.simple_simulate_logsums = gpu_simple
        try:
            result = original_compute(*compute_args, **compute_kwargs)
            capture_logsums(compute_args, result)
            return result
        finally:
            simulate.simple_simulate_logsums = original_simple

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
        Runner.__call__ = original_runner_call

    actual = pd.read_parquet(
        args.output / "pipeline.parquetpipeline" / "tours" / "mandatory_tour_scheduling.parquet"
    ).sort_index()
    expected = pd.read_parquet(
        args.reference_pipeline / "tours" / "mandatory_tour_scheduling.parquet"
    ).sort_index()
    mandatory = expected.tour_category.astype(str).eq("mandatory")
    expected = expected.loc[mandatory]
    actual = actual.loc[expected.index]
    reports = []
    for path in sorted(args.kernel_reports.glob("*.json")):
        reports.append(json.loads(path.read_text()))
    fallback_reports = [item for item in reports if item.get("fallback_used")]
    candidate_reports = [item for item in reports if item.get("candidate_used")]
    if args.logsum_capture:
        capture_manifest = []
        for number, item in enumerate(logsum_captures):
            filename = f"batch_{number:03d}_{item['purpose']}.npz"
            np.savez_compressed(
                args.logsum_capture / filename,
                chooser_ids=item["chooser_ids"],
                start=item["start"],
                end=item["end"],
                logsums=item["logsums"],
            )
            capture_manifest.append(
                {
                    "file": filename,
                    "purpose": item["purpose"],
                    "trace_label": item["trace_label"],
                    "rows": int(item["logsums"].size),
                }
            )
        (args.logsum_capture / "manifest.json").write_text(
            json.dumps({"phase": 21, "batches": capture_manifest}, indent=2) + "\n"
        )
    report = {
        "phase": 21,
        "engine": args.engine,
        "scope": "live ActivitySim mandatory-scheduling mode-choice logsums from raw network skims",
        "elapsed_seconds_including_resume_overhead": elapsed,
        "exit_code": int(exit_code or 0),
        "mandatory_tours": int(len(expected)),
        "tdd_mismatches": int(np.count_nonzero(actual.tdd.to_numpy() != expected.tdd.to_numpy())),
        "start_mismatches": int(np.count_nonzero(actual.start.to_numpy() != expected.start.to_numpy())),
        "end_mismatches": int(np.count_nonzero(actual.end.to_numpy() != expected.end.to_numpy())),
        "candidate_calls": len(candidate_reports),
        "fallback_calls": len(fallback_reports),
        "candidate_rows": int(sum(item.get("rows", 0) for item in candidate_reports)),
        "utility_device_to_host_bytes": int(
            sum(item.get("utility_device_to_host_bytes", 0) for item in candidate_reports)
        ),
        "nested_host_to_device_bytes": int(
            sum(item.get("nested_host_to_device_bytes", 0) for item in candidate_reports)
        ),
        "kernel_reports": [path.name for path in sorted(args.kernel_reports.glob("*.json"))],
    }
    report["proof_gates"] = {
        "activitysim_completed": report["exit_code"] == 0,
        "all_tdds_exact": report["tdd_mismatches"] == 0,
        "all_start_end_exact": report["start_mismatches"] == 0 and report["end_mismatches"] == 0,
        "cuda_candidate_used": report["candidate_calls"] > 0,
        "no_cuda_fallbacks": report["fallback_calls"] == 0,
        "device_resident_utility_handoff": (
            report["utility_device_to_host_bytes"] == 0
            and report["nested_host_to_device_bytes"] == 0
        ),
    }
    if args.engine == "cpu":
        report["proof_gates"] = {
            "activitysim_completed": report["exit_code"] == 0,
            "frozen_reference_exact": (
                report["tdd_mismatches"] == 0
                and report["start_mismatches"] == 0
                and report["end_mismatches"] == 0
            ),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    required_gates = (
        [report["proof_gates"]["activitysim_completed"]]
        if args.engine == "cpu"
        else list(report["proof_gates"].values())
    )
    if not all(required_gates):
        return 2
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
