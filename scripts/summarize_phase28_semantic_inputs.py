"""Hash-chain three Phase 28 processes and changed-scenario qualification."""

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
    parser.add_argument("--phase27", type=Path, required=True)
    parser.add_argument("--phase26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 3 or len(args.live) != 3:
        raise SystemExit("Phase 28 requires exactly three resident and three live reports")
    resident = [read(path) for path in args.input]
    live = [read(path) for path in args.live]
    scenarios, scenarios_hash = read(args.scenarios)
    phase27, phase27_hash = read(args.phase27)
    phase26, phase26_hash = read(args.phase26)
    reports = [item[0] for item in resident]
    live_reports = [item[0] for item in live]
    process_medians = [item["median_seconds"] for item in reports]
    expansion_medians = [item["expansion_gpu_median_seconds"] for item in reports]
    cold_seconds = [item["elapsed_seconds_including_resume_overhead"] for item in live_reports]
    compact = {item["compact_input_state_bytes"] for item in reports}
    ratios = {item["compact_state_reduction_ratio"] for item in reports}
    removed_dictionaries = [
        sum(program["removed_response_dictionary_bytes"] for program in item["semantic_input_programs"])
        for item in reports
    ]
    phase28_seconds = statistics.median(process_medians)
    phase27_seconds = phase27["median_of_process_medians_seconds"]
    phase26_seconds = phase26["median_of_process_medians_seconds"]
    proof = {
        "three_independent_processes": len(reports) == 3,
        "all_process_gates_passed": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_live_gates_passed": all(
            all(item["proof_gates"].values()) for item in live_reports
        ),
        "all_changed_scenario_gates_passed": all(scenarios["proof_gates"].values()),
        "consistent_compact_state": len(compact) == 1 and len(ratios) == 1,
        "all_fifteen_formulas_present": all(
            sum(
                program["generated_float_columns"] + program["generated_int_columns"]
                for program in item["semantic_input_programs"]
            ) >= 89
            for item in reports
        ),
        "zero_anonymous_response_patterns": all(
            program["anonymous_response_pattern_columns"] == 0
            for item in reports for program in item["semantic_input_programs"]
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
        "phase": 28,
        "scope": (
            "named semantic CUDA input generation through the complete sealed "
            "mandatory-tour raw-skim-to-timetable graph"
        ),
        "workload": {
            "households": 50000,
            "mode_logsum_rows": reports[0]["mode_logsum_rows"],
            "scheduled_tours": reports[0]["scheduled_tours"],
            "programs": reports[0]["programs"],
        },
        "complete_graph_process_medians_seconds": process_medians,
        "median_of_process_medians_seconds": phase28_seconds,
        "minimum_complete_graph_seconds": min(min(item["seconds"]) for item in reports),
        "semantic_generation_process_medians_seconds": expansion_medians,
        "semantic_generation_median_seconds": statistics.median(expansion_medians),
        "cold_activitysim_process_seconds": cold_seconds,
        "cold_activitysim_median_seconds": statistics.median(cold_seconds),
        "compact_input_state_bytes": compact.pop(),
        "removed_captured_row_bytes": reports[0]["removed_captured_row_bytes"],
        "compact_state_reduction_ratio": ratios.pop(),
        "removed_response_dictionary_bytes": statistics.median(removed_dictionaries),
        "phase28_overhead_vs_phase27_percent": (phase28_seconds / phase27_seconds - 1) * 100,
        "phase28_overhead_vs_phase26_percent": (phase28_seconds / phase26_seconds - 1) * 100,
        "phase28_compact_bytes_reduction_vs_phase27_percent": (
            1 - reports[0]["compact_input_state_bytes"] / phase27["compact_input_state_bytes"]
        ) * 100,
        "qualification": {
            "full_public_process_replays": sum(item["measured_runs"] for item in reports),
            "changed_scenarios": len(scenarios["scenarios"]),
            "changed_scenario_rows": sum(item["rows"] for item in scenarios["scenarios"]),
        },
        "proof_gates": proof,
        "source_hashes": {
            **{str(path): digest for path, (_, digest) in zip(args.input, resident)},
            **{str(path): digest for path, (_, digest) in zip(args.live, live)},
            str(args.scenarios): scenarios_hash,
            str(args.phase27): phase27_hash,
            str(args.phase26): phase26_hash,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not all(proof.values()):
        raise SystemExit("Phase 28 summary proof gate failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
