"""Create a replicated Phase 23 qualification from independent processes."""

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
        raise ValueError("at least three independent Phase 23 reports are required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if any(item.get("phase") != 23 for item in reports):
        raise ValueError("all inputs must be Phase 23 reports")

    cpu = [item["timings_seconds"]["cpu_modeled_median"] for item in reports]
    gpu = [item["timings_seconds"]["gpu_resident_median"] for item in reports]
    setup = [item["timings_seconds"]["one_time_setup_total"] for item in reports]
    publication = [item["timings_seconds"]["final_publication"] for item in reports]
    resident_speedups = [c / g for c, g in zip(cpu, gpu)]
    inclusive_speedups = [
        c / (s + g + p) for c, s, g, p in zip(cpu, setup, gpu, publication)
    ]
    gates = {
        "three_independent_processes": len(reports) >= 3,
        "all_source_proof_gates_pass": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_behavioral_outputs_exact": all(
            item["correctness"]["auto_checkpoint_mismatches"] == 0
            and item["correctness"]["mandatory_frequency_checkpoint_mismatches"] == 0
            and all(
                value == 0
                for value in item["correctness"]["tour_column_mismatches"].values()
            )
            and item["correctness"]["tdd_mismatches"] == 0
            and item["correctness"]["timetable_mismatches_vs_cpu"] == 0
            for item in reports
        ),
        "resident_gpu_faster_every_process": all(value > 1.0 for value in resident_speedups),
        "setup_inclusive_gpu_faster_every_process": all(
            value > 1.0 for value in inclusive_speedups
        ),
    }
    result = {
        "phase": 23,
        "qualification": "three independent process-level device-resident runs",
        "processes": len(reports),
        "measured_repetitions_per_process": [
            len(item["timings_seconds"]["gpu_resident_samples"]) for item in reports
        ],
        "cpu_modeled_medians_seconds": cpu,
        "gpu_resident_medians_seconds": gpu,
        "one_time_setup_seconds": setup,
        "final_publication_seconds": publication,
        "resident_speedups": resident_speedups,
        "setup_inclusive_speedups": inclusive_speedups,
        "median_cpu_modeled_seconds": statistics.median(cpu),
        "median_gpu_resident_seconds": statistics.median(gpu),
        "median_resident_speedup": statistics.median(resident_speedups),
        "median_setup_inclusive_speedup": statistics.median(inclusive_speedups),
        "minimum_resident_speedup": min(resident_speedups),
        "minimum_setup_inclusive_speedup": min(inclusive_speedups),
        "median_ten_repeat_amortized_speedup": statistics.median(
            item["speedup"]["ten_repeated_runs_amortized"] for item in reports
        ),
        "median_hundred_repeat_amortized_speedup": statistics.median(
            item["speedup"]["hundred_repeated_runs_amortized"] for item in reports
        ),
        "proof_gates": gates,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in args.input
        ],
    }
    if not all(gates.values()):
        raise AssertionError(f"Phase 23 replicated gate failed: {gates}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
