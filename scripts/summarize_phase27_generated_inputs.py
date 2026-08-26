"""Hash-chain independent Phase 27 compact-input reconstruction proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--phase26", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 3:
        raise ValueError("Phase 27 qualification requires three independent processes")

    reports = [json.loads(path.read_text()) for path in args.input]
    replays = [item for report in reports for item in report["replays"]]
    process_medians = [float(item["median_seconds"]) for item in reports]
    gpu_expansion_medians = [
        float(item["expansion_gpu_median_seconds"]) for item in reports
    ]
    cpu_expansion_medians = [
        float(item["expansion_cpu_median_seconds"]) for item in reports
    ]
    phase26 = json.loads(args.phase26.read_text()) if args.phase26 else None
    phase26_seconds = (
        float(phase26["median_of_process_medians_seconds"]) if phase26 else None
    )
    phase27_seconds = statistics.median(process_medians)
    gates = {
        "three_independent_processes": len(reports) >= 3,
        "every_source_gate_passed": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_programs_rows_and_tours_replayed": all(
            item["programs"] == 6
            and item["mode_logsum_rows"] == 1_210_124
            and item["scheduled_tours"] == 81_983
            for item in reports
        ),
        "all_reconstructed_logsums_bit_exact": all(
            item["logsum_bit_mismatches"] == 0 for item in replays
        ),
        "all_final_tdds_exact": all(
            item["final_tdd_mismatches"] == 0 for item in reports
        ),
        "captured_row_arrays_absent": all(
            item["captured_dense_input_bytes_in_timed_graph"] == 0
            and item["captured_coordinate_bytes_in_timed_graph"] == 0
            and not item["retained_original_captured_pointers"]
            for item in reports
        ),
        "compact_state_at_least_20x_smaller": all(
            item["compact_state_reduction_ratio"] >= 20.0 for item in reports
        ),
        "gpu_expansion_wins_every_process": all(
            gpu < cpu
            for gpu, cpu in zip(gpu_expansion_medians, cpu_expansion_medians)
        ),
        "zero_postseal_transfer_or_cpu_fallback": all(
            item["modeled_host_to_device_bytes_after_seal"] == 0
            and item["intermediate_modeled_device_to_host_bytes"] == 0
            and item["runtime_telemetry"]["modeled_cpu_fallbacks"] == 0
            for item in reports
        ),
    }
    summary = {
        "phase": 27,
        "claim": (
            "three-process public MTC proof that a sealed CUDA graph rebuilds "
            "all strict mode-choice row leaves and skim coordinates from compact "
            "chooser, exact-slot, response-pattern, and CSR state"
        ),
        "processes": len(reports),
        "measured_full_graph_replays": len(replays),
        "process_median_seconds": process_medians,
        "median_of_process_medians_seconds": phase27_seconds,
        "slowest_process_median_seconds": max(process_medians),
        "minimum_full_graph_replay_seconds": min(
            float(item["seconds"]) for item in replays
        ),
        "expansion_gpu_process_median_seconds": gpu_expansion_medians,
        "expansion_cpu_process_median_seconds": cpu_expansion_medians,
        "expansion_gpu_median_of_medians_seconds": statistics.median(
            gpu_expansion_medians
        ),
        "expansion_cpu_median_of_medians_seconds": statistics.median(
            cpu_expansion_medians
        ),
        "expansion_speedup_cpu_over_gpu_median_boundary": (
            statistics.median(cpu_expansion_medians)
            / statistics.median(gpu_expansion_medians)
        ),
        "removed_captured_row_bytes": reports[0]["removed_captured_row_bytes"],
        "compact_input_state_bytes": reports[0]["compact_input_state_bytes"],
        "compact_state_reduction_ratio": reports[0]["compact_state_reduction_ratio"],
        "phase26_median_seconds": phase26_seconds,
        "phase27_overhead_vs_phase26_percent": (
            (phase27_seconds / phase26_seconds - 1.0) * 100.0
            if phase26_seconds else None
        ),
        "workload": {
            "households": 50_000,
            "scheduled_tours": reports[0]["scheduled_tours"],
            "mode_logsum_rows": reports[0]["mode_logsum_rows"],
            "programs": reports[0]["programs"],
            "terms_per_program": 315,
        },
        "proof_gates": gates,
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in args.input
        ],
    }
    if args.phase26:
        summary["phase26_source"] = {
            "path": str(args.phase26), "sha256": sha256(args.phase26)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(gates.values()):
        raise SystemExit("Phase 27 summary proof gate failed")


if __name__ == "__main__":
    main()
