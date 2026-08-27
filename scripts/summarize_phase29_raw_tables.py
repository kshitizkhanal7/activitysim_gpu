"""Hash-chain the three Phase 29 public processes and changed worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--live", type=Path, action="append", required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--phase28", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 3 or len(args.live) != 3:
        raise SystemExit("Phase 29 requires exactly three resident and three live reports")
    resident = [read(path) for path in args.input]
    live = [read(path) for path in args.live]
    scenarios, scenarios_hash = read(args.scenarios)
    phase28, phase28_hash = read(args.phase28)
    reports = [item[0] for item in resident]
    live_reports = [item[0] for item in live]
    medians = [item["median_seconds"] for item in reports]
    generation = [item["expansion_gpu_median_seconds"] for item in reports]
    cold = [item["elapsed_seconds_including_resume_overhead"] for item in live_reports]
    compact = {item["compact_input_state_bytes"] for item in reports}
    ratios = {item["compact_state_reduction_ratio"] for item in reports}
    phase29_seconds = statistics.median(medians)
    phase28_seconds = phase28["median_of_process_medians_seconds"]
    raw_manifests = [
        manifest for item in reports for manifest in item["raw_table_input_programs"]
    ]
    semantic_manifests = [
        manifest for item in reports for manifest in item["semantic_input_programs"]
    ]
    proof = {
        "three_independent_processes": len(reports) == 3,
        "all_process_gates_passed": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_live_gates_passed": all(
            all(item["proof_gates"].values()) for item in live_reports
        ),
        "all_changed_world_gates_passed": all(scenarios["proof_gates"].values()),
        "consistent_compact_state": len(compact) == 1 and len(ratios) == 1,
        "all_57_sources_declared_per_program": all(
            item["source_count"] == 57 for item in raw_manifests
        ),
        "zero_dense_oracle_bytes_read_for_compile": all(
            item["dense_oracle_bytes_read_for_compile"] == 0
            for item in raw_manifests
        ),
        "all_18_availability_formulas_per_program": all(
            item["generated_int_columns"] == 18 for item in semantic_manifests
        ),
        "direct_land_use_parking_rates": all(
            item["parking_rate_source"]
            == "land_use.PRKCST_or_free_parking_at_work"
            for item in raw_manifests
        ),
        "zero_anonymous_response_patterns": all(
            item["anonymous_response_pattern_columns"] == 0
            for item in semantic_manifests
        ),
        "all_logsums_and_schedules_exact": all(
            item["final_tdd_mismatches"] == 0
            and all(replay["logsum_bit_mismatches"] == 0 for replay in item["replays"])
            for item in reports
        ),
        "no_postseal_host_traffic_or_cpu_fallback": all(
            item["modeled_host_to_device_bytes_after_seal"] == 0
            and item["intermediate_modeled_device_to_host_bytes"] == 0
            and item["runtime_telemetry"]["modeled_cpu_fallbacks"] == 0
            for item in reports
        ),
    }
    result = {
        "phase": 29,
        "scope": (
            "declared one-row-per-tour and land-use source compilation through "
            "the complete sealed mandatory-tour raw-skim-to-timetable graph"
        ),
        "workload": {
            "households": 50000,
            "mode_logsum_rows": reports[0]["mode_logsum_rows"],
            "scheduled_tours": reports[0]["scheduled_tours"],
            "programs": reports[0]["programs"],
            "declared_sources_per_program": 57,
            "availability_formulas_per_program": 18,
        },
        "complete_graph_process_medians_seconds": medians,
        "median_of_process_medians_seconds": phase29_seconds,
        "minimum_complete_graph_seconds": min(min(item["seconds"]) for item in reports),
        "raw_input_generation_process_medians_seconds": generation,
        "raw_input_generation_median_seconds": statistics.median(generation),
        "cold_activitysim_process_seconds": cold,
        "cold_activitysim_median_seconds": statistics.median(cold),
        "compact_input_state_bytes": compact.pop(),
        "removed_captured_row_bytes": reports[0]["removed_captured_row_bytes"],
        "compact_state_reduction_ratio": ratios.pop(),
        "dense_oracle_bytes_read_for_compile": 0,
        "phase29_overhead_vs_phase28_percent": (
            phase29_seconds / phase28_seconds - 1
        ) * 100,
        "phase29_compact_state_change_vs_phase28_percent": (
            reports[0]["compact_input_state_bytes"]
            / phase28["compact_input_state_bytes"] - 1
        ) * 100,
        "qualification": {
            "full_public_process_replays": sum(item["measured_runs"] for item in reports),
            "changed_raw_table_scenarios": len(scenarios["raw_table_scenarios"]),
            "changed_raw_tours": sum(
                item["raw_tours"] for item in scenarios["raw_table_scenarios"]
            ),
            "changed_cuda_scenarios": len(scenarios["cuda_scenarios"]),
            "changed_cuda_rows": sum(
                item["rows"] for item in scenarios["cuda_scenarios"]
            ),
        },
        "proof_gates": proof,
        "source_hashes": {
            **{str(path): digest for path, (_, digest) in zip(args.input, resident)},
            **{str(path): digest for path, (_, digest) in zip(args.live, live)},
            str(args.scenarios): scenarios_hash,
            str(args.phase28): phase28_hash,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not all(proof.values()):
        raise SystemExit("Phase 29 summary proof gate failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
