"""Assemble and validate Phase 43's compact controlled-random qualification."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stage_total(report, name):
    return sum(
        event[name]
        for event in report["phase35_trip_destination_stages"]["events"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase43-p43final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    summary = read(results / "phase43-p43final-summary.json")
    baselines = [
        read(results / f"phase43-p43final-base-{trial}.json")
        for trial in range(1, 4)
    ]
    candidates = [
        read(results / f"phase43-p43final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(results / f"phase43-p43final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    compact = [item["phase43_compact_trip_state"] for item in candidates]
    baseline_direct = [stage_total(item, "total_seconds") for item in baselines]
    candidate_direct = [stage_total(item, "total_seconds") for item in candidates]
    baseline_simulation = [
        stage_total(item, "simulation_seconds") for item in baselines
    ]
    candidate_simulation = [
        stage_total(item, "simulation_seconds") for item in candidates
    ]
    candidate_choice = [
        sum(
            event["simulation_profile"]["choice_seconds"]
            for event in item["phase35_trip_destination_stages"]["events"]
        )
        for item in candidates
    ]
    compact_choice_calls = [
        sum(
            event["simulation_profile"]["compact_choice_calls"]
            for event in item["phase35_trip_destination_stages"]["events"]
        )
        for item in candidates
    ]
    trip = next(
        item
        for item in summary["component_comparison"]
        if item["model_name"] == "trip_destination"
    )
    gates = {
        "three_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "all_compact_packets_cover_exact_public_workload": all(
            item["compact_draw_rows"] == 183_048
            and item["expanded_draw_rows_avoided"] == 4_005_264
            and item["choice_draw_rows"] == 91_524
            and item["choice_draws_consumed"] == 91_524
            and item["rng_calls"] == 9
            for item in compact
        ),
        "all_thirty_final_choice_calls_consume_compact_draws": all(
            value == 30 for value in compact_choice_calls
        ),
        "three_independent_output_verifiers_are_byte_exact": all(
            item["success"]
            and item["decision_cells_different"] == 0
            and item["diagnostic_max_abs"] == 0.0
            and item["mode_choice_logsum_max_abs"] == 0.0
            and len(item["byte_identical_outputs"]) == 7
            for item in exact
        ),
        "direct_trip_boundary_improves_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(baseline_direct, candidate_direct)
        ),
        "final_simulation_boundary_improves_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(
                baseline_simulation, candidate_simulation
            )
        ),
        "median_public_trip_destination_component_improves": (
            trip["median_candidate_seconds"] < trip["median_baseline_seconds"]
        ),
        "median_complete_model_does_not_regress": (
            summary["median_candidate_all_model_seconds"]
            < summary["median_baseline_all_model_seconds"]
        ),
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
        "summary_declares_every_target_component_pair_won": summary[
            "candidate_won_target_component_every_pair"
        ],
    }
    document = {
        "phase": 43,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "matched_pairs": 3,
        "trip_rows": 91_524,
        "sampled_destination_rows": 2_094_156,
        "expanded_directional_draw_rows_avoided": 4_005_264,
        "activitysim_rng_calls": 9,
        "decision_cell_differences": [
            item["decision_cells_different"] for item in exact
        ],
        "destination_logsum_max_abs": [item["diagnostic_max_abs"] for item in exact],
        "mode_logsum_max_abs": [
            item["mode_choice_logsum_max_abs"] for item in exact
        ],
        "direct_trip_boundary_seconds": {
            "phase42": baseline_direct,
            "phase43": candidate_direct,
            "median_phase42": statistics.median(baseline_direct),
            "median_phase43": statistics.median(candidate_direct),
            "median_speedup": statistics.median(baseline_direct)
            / statistics.median(candidate_direct),
        },
        "final_simulation_seconds": {
            "phase42": baseline_simulation,
            "phase43": candidate_simulation,
            "median_phase42": statistics.median(baseline_simulation),
            "median_phase43": statistics.median(candidate_simulation),
            "median_speedup": statistics.median(baseline_simulation)
            / statistics.median(candidate_simulation),
            "phase43_choice_seconds": candidate_choice,
        },
        "median_trip_destination_component": trip,
        "complete_model": {
            "median_phase42_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase43_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
            "interpretation": (
                "The phase-specific trip boundary wins all three pairs; whole-model "
                "pair noise is larger than this sub-second optimization."
            ),
        },
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 43 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
