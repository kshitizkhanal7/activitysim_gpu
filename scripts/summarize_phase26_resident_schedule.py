"""Hash-chain independent Phase 26 resident raw-skim-to-timetable proofs."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 3:
        raise ValueError("Phase 26 qualification requires three independent processes")

    reports = [json.loads(path.read_text()) for path in args.input]
    process_medians = [float(item["median_seconds"]) for item in reports]
    all_replays = [replay for item in reports for replay in item["replays"]]
    gates = {
        "three_independent_processes": len(reports) >= 3,
        "every_source_gate_passed": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_programs_and_rows_replayed": all(
            item["programs"] == 6 and item["mode_logsum_rows"] == 1_210_124
            for item in reports
        ),
        "all_final_tdds_exact": all(
            item["final_tdd_mismatches"] == 0 for item in reports
        ),
        "all_logsums_bit_exact": all(
            item["logsum_bit_mismatches"] == 0 for item in all_replays
        ),
        "all_ambiguity_rows_resolved_on_device": all(
            item["boundary_rows"] == item["device_boundary_adjudications"]
            and item["boundary_download_bytes"] == 0
            for item in all_replays
        ),
        "sparse_map_contains_only_qualified_ambiguities": all(
            item["qualified_boundary_map_entries"] == 57 for item in reports
        ),
        "zero_modeled_postseal_transfers": all(
            item["modeled_host_to_device_bytes_after_seal"] == 0
            and item["intermediate_modeled_device_to_host_bytes"] == 0
            for item in reports
        ),
        "zero_modeled_cpu_fallbacks": all(
            item["runtime_telemetry"]["modeled_cpu_fallbacks"] == 0
            for item in reports
        ),
    }
    summary = {
        "phase": 26,
        "claim": (
            "three-process public MTC proof of a sealed raw-skim-to-timetable "
            "CUDA graph with device-generated scheduling rows and a versioned "
            "device-resident Sharrow ambiguity map"
        ),
        "processes": len(reports),
        "measured_replays": len(all_replays),
        "process_median_seconds": process_medians,
        "median_of_process_medians_seconds": statistics.median(process_medians),
        "slowest_process_median_seconds": max(process_medians),
        "minimum_replay_seconds": min(float(item["seconds"]) for item in all_replays),
        "workload": {
            "households": 50_000,
            "scheduled_tours": reports[0]["scheduled_tours"],
            "mode_logsum_rows": reports[0]["mode_logsum_rows"],
            "programs": reports[0]["programs"],
            "terms_per_program": 315,
        },
        "arithmetic_contract": reports[0]["arithmetic_contract"],
        "boundary_rows_per_replay": sorted(
            {int(item["boundary_rows"]) for item in all_replays}
        ),
        "qualified_boundary_map_entries": sorted(
            {int(item["qualified_boundary_map_entries"]) for item in reports}
        ),
        "device_boundary_corrections_per_replay": sorted(
            {int(item["device_boundary_corrections"]) for item in all_replays}
        ),
        "final_publication_bytes_per_process": reports[0]["final_publication_bytes"],
        "proof_gates": gates,
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in args.input
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(gates.values()):
        raise SystemExit("Phase 26 summary proof gate failed")


if __name__ == "__main__":
    main()
