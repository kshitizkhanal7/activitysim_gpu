"""Summarize repeated Phase 22 CPU/GPU live runs into a proof artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(paths):
    return [(path, json.loads(path.read_text())) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.cpu) != len(args.gpu) or len(args.cpu) < 3:
        raise ValueError("at least three paired CPU/GPU reports are required")

    cpu = load(args.cpu)
    gpu = load(args.gpu)
    cpu_seconds = [x[1]["elapsed_seconds_including_resume_overhead"] for x in cpu]
    gpu_seconds = [x[1]["elapsed_seconds_including_resume_overhead"] for x in gpu]
    paired_speedups = [c / g for c, g in zip(cpu_seconds, gpu_seconds)]
    exact_cpu = all(
        item["exit_code"] == 0
        and item["tdd_mismatches"] == 0
        and item["start_mismatches"] == 0
        and item["end_mismatches"] == 0
        for _, item in cpu
    )
    exact_gpu = all(
        item["exit_code"] == 0
        and all(item["proof_gates"].values())
        and item["tdd_mismatches"] == 0
        and item["start_mismatches"] == 0
        and item["end_mismatches"] == 0
        for _, item in gpu
    )
    boundary_rows = [item["exact_boundary_rows"] for _, item in gpu]
    report = {
        "phase": 22,
        "benchmark": "paired live mandatory-tour scheduling from raw public MTC skims",
        "pairs": len(cpu),
        "mandatory_tours_per_run": gpu[0][1]["mandatory_tours"],
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "paired_speedups": paired_speedups,
        "cpu_median_seconds": float(np.median(cpu_seconds)),
        "gpu_median_seconds": float(np.median(gpu_seconds)),
        "median_of_paired_speedups": float(np.median(paired_speedups)),
        "ratio_of_medians": float(np.median(cpu_seconds) / np.median(gpu_seconds)),
        "gpu_faster_in_every_pair": all(x > 1.0 for x in paired_speedups),
        "exact_boundary_rows_per_run": boundary_rows,
        "exact_boundary_fraction": float(
            boundary_rows[0] / gpu[0][1]["mandatory_tours"]
        ),
        "boundary_logsum_download_bytes_per_run": [
            item["boundary_logsum_download_bytes"] for _, item in gpu
        ],
        "proof_gates": {
            "three_or_more_pairs": len(cpu) >= 3,
            "all_cpu_controls_exact": exact_cpu,
            "all_gpu_runs_exact": exact_gpu,
            "gpu_faster_in_every_pair": all(x > 1.0 for x in paired_speedups),
            "boundary_population_deterministic": len(set(boundary_rows)) == 1,
        },
        "inputs": {
            "cpu": [
                {"path": str(path), "sha256": file_sha256(path)} for path, _ in cpu
            ],
            "gpu": [
                {"path": str(path), "sha256": file_sha256(path)} for path, _ in gpu
            ],
        },
    }
    if not all(report["proof_gates"].values()):
        raise AssertionError(f"Phase 22 paired qualification failed: {report['proof_gates']}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
