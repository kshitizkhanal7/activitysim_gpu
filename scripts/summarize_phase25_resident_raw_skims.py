"""Build the cross-process qualification for Phase 25 resident computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--live", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.input]
    live_reports = [json.loads(path.read_text()) for path in args.live]
    if len(reports) < 3:
        raise ValueError("Phase 25 qualification requires at least three processes")
    if len(live_reports) != len(reports):
        raise ValueError("provide one live ActivitySim report for each resident report")

    invariant_fields = (
        "batches",
        "rows_per_replay",
        "terms_per_program",
        "alternatives_per_program",
        "logical_skim_bindings_per_program",
        "unique_skim_arrays_per_program",
        "sealed_dense_input_bytes",
        "sealed_skim_coordinate_bytes",
        "compiled_scatter_plan_device_bytes",
        "precomputed_logsum_input_bytes",
        "bulk_modeled_logsum_device_to_host_bytes",
        "postseal_host_layout_builds",
    )
    invariants = {
        field: reports[0][field]
        for field in invariant_fields
    }
    invariant_match = all(
        report.get(field) == expected
        for field, expected in invariants.items()
        for report in reports
    )
    process_medians = [report["resident_median_seconds"] for report in reports]
    process_minima = [report["resident_min_seconds"] for report in reports]
    speedups = [
        report["resident_speedup_vs_initial_live_device_pipeline"]
        for report in reports
    ]
    process_memory_reports = [
        report for report in reports
        if report.get("unique_resident_skim_bytes_process") is not None
    ]
    memory_consistent = bool(process_memory_reports) and len({
        (
            report["unique_resident_skim_arrays_process"],
            report["unique_resident_skim_bytes_process"],
        )
        for report in process_memory_reports
    }) == 1
    summary = {
        "phase": 25,
        "benchmark": (
            "three-process public MTC resident raw-skims-to-mode-logsum-cache replay"
        ),
        "processes": len(reports),
        **invariants,
        "unique_resident_skim_arrays_process": (
            process_memory_reports[0]["unique_resident_skim_arrays_process"]
            if process_memory_reports else None
        ),
        "unique_resident_skim_bytes_process": (
            process_memory_reports[0]["unique_resident_skim_bytes_process"]
            if process_memory_reports else None
        ),
        "process_median_seconds": process_medians,
        "process_min_seconds": process_minima,
        "median_of_process_medians_seconds": float(np.median(process_medians)),
        "slowest_process_median_seconds": float(np.max(process_medians)),
        "resident_speedups_vs_initial_live_device_pipeline": speedups,
        "median_resident_speedup_vs_initial_live_device_pipeline": float(
            np.median(speedups)
        ),
        "minimum_resident_speedup_vs_initial_live_device_pipeline": float(
            np.min(speedups)
        ),
        "total_measured_replays": int(sum(
            report["measured_runs"] for report in reports
        )),
        "total_logsum_bit_mismatches": int(sum(
            replay["logsum_bit_mismatches"]
            for report in reports for replay in report["replays"]
        )),
        "maximum_logsum_absolute_difference": float(max(
            replay["logsum_max_abs_difference"]
            for report in reports for replay in report["replays"]
        )),
        "live_activitysim_seconds": [
            report["elapsed_seconds_including_resume_overhead"]
            for report in live_reports
        ],
        "live_activitysim_output_mismatches": int(sum(
            report["tdd_mismatches"]
            + report["start_mismatches"]
            + report["end_mismatches"]
            for report in live_reports
        )),
        "proof_gates": {
            "three_or_more_independent_processes": len(reports) >= 3,
            "all_source_gates_pass": all(
                all(report["proof_gates"].values()) for report in reports
            ),
            "all_live_activitysim_gates_pass": all(
                all(report["proof_gates"].values()) for report in live_reports
            ),
            "all_live_activitysim_outputs_exact": all(
                report["tdd_mismatches"] == 0
                and report["start_mismatches"] == 0
                and report["end_mismatches"] == 0
                for report in live_reports
            ),
            "schema_and_workload_invariant": invariant_match,
            "resident_skim_memory_consistent_where_recorded": memory_consistent,
            "all_fifteen_replays_bit_exact": all(
                replay["logsum_bit_mismatches"] == 0
                for report in reports for replay in report["replays"]
            ),
            "no_precomputed_logsum_input": all(
                report["precomputed_logsum_input_bytes"] == 0
                for report in reports
            ),
            "resident_replay_faster_than_initial_pipeline_every_process": all(
                speedup > 1.0 for speedup in speedups
            ),
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in args.input
        ],
        "live_inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in args.live
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["proof_gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
