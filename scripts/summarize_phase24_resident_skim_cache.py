"""Build a fail-closed replicated summary of Phase 24 process reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 3:
        raise ValueError("Phase 24 replication requires at least three processes")
    reports = [json.loads(path.read_text()) for path in args.input]
    if any(report.get("phase") != 24 for report in reports):
        raise ValueError("all source reports must be Phase 24")
    if any(not all(report["proof_gates"].values()) for report in reports):
        raise AssertionError("a Phase 24 source proof gate failed")
    invariants = (
        "valid_rows", "logical_skim_bindings", "physical_device_cubes",
        "logical_skim_reads_per_run",
    )
    for name in invariants:
        if len({report["workload"][name] for report in reports}) != 1:
            raise AssertionError(f"replicated workload field {name!r} changed")
    if len({report["correctness"]["published_hash_sha256"] for report in reports}) != 1:
        raise AssertionError("replicated public skim result hash changed")
    speedups = [report["speedup"]["resident_gpu_vs_cpu_hot_cache"] for report in reports]
    setup_speedups = [report["speedup"]["upload_inclusive_single_run"] for report in reports]
    ten_speedups = [report["speedup"]["ten_runs_upload_amortized"] for report in reports]
    summary = {
        "phase": 24,
        "independent_processes": len(reports),
        "workload": {name: reports[0]["workload"][name] for name in invariants},
        "resident_float32_bytes": reports[0]["cache"]["resident_float32_bytes"],
        "cpu_medians_seconds": [x["timings_seconds"]["cpu_hot_cache_median"] for x in reports],
        "gpu_medians_seconds": [x["timings_seconds"]["gpu_resident_median"] for x in reports],
        "process_resident_speedups": speedups,
        "replicated_median_resident_speedup": statistics.median(speedups),
        "replicated_minimum_resident_speedup": min(speedups),
        "process_upload_inclusive_single_run_speedups": setup_speedups,
        "replicated_median_upload_inclusive_single_run_speedup": statistics.median(setup_speedups),
        "process_ten_run_amortized_speedups": ten_speedups,
        "replicated_median_ten_run_amortized_speedup": statistics.median(ten_speedups),
        "cpu_gpu_hash_word_mismatches": sum(
            x["correctness"]["cpu_gpu_hash_word_mismatches"] for x in reports
        ),
        "repeat_hash_word_mismatches": sum(
            sum(x["correctness"]["repeat_hash_word_mismatches"]) for x in reports
        ),
        "postseal_modeled_transfer_bytes": sum(
            x["runtime_telemetry"]["forbidden_postseal_host_bytes"] for x in reports
        ),
        "modeled_cpu_fallbacks": sum(
            x["runtime_telemetry"]["modeled_cpu_fallbacks"] for x in reports
        ),
        "published_hash_sha256": reports[0]["correctness"]["published_hash_sha256"],
        "source_reports": [
            {"path": str(path), "sha256": sha256(path)} for path in args.input
        ],
    }
    summary["proof_gates"] = {
        "three_or_more_independent_processes": len(reports) >= 3,
        "all_process_gates_passed": True,
        "replicated_result_hash_exact": True,
        "zero_cpu_gpu_mismatches": summary["cpu_gpu_hash_word_mismatches"] == 0,
        "zero_repeat_mismatches": summary["repeat_hash_word_mismatches"] == 0,
        "zero_postseal_modeled_transfers": summary["postseal_modeled_transfer_bytes"] == 0,
        "zero_modeled_cpu_fallbacks": summary["modeled_cpu_fallbacks"] == 0,
        "gpu_won_every_process": min(speedups) > 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(summary["proof_gates"].values()):
        raise SystemExit("Phase 24 replicated proof gate failed")


if __name__ == "__main__":
    main()
