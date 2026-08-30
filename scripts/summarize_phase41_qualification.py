"""Assemble and validate the Phase 41 promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase41-p41final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    utility_report = read(results / "phase39-p41abishadow-gpu-1.json")
    utility_events = utility_report["phase39_trip_destination_sampling"]
    summary = read(results / "phase41-p41final-summary.json")
    candidates = [read(results / f"phase41-p41final-gpu-{trial}.json") for trial in range(1, 4)]
    exact = [read(results / f"phase41-p41final-exact-{trial}.json") for trial in range(1, 4)]
    arithmetic_probe = read(results / "phase41-openblas-arithmetic-probe.json")
    probability_probe = read(results / "phase41-numpy-probability-probe.json")

    candidate_events = [item["phase41_trip_destination_sampling"] for item in candidates]
    gates = {
        "openblas_probe_is_bit_exact": arithmetic_probe["comparisons"]["group4_left"]["bit_mismatches"] == 0,
        "full_utility_shadow_covers_every_cell": sum(
            item["shadow_utility_cells_compared"] for item in utility_events
        ) == 133_075_896,
        "full_utility_shadow_is_bit_exact": sum(
            item["shadow_utility_mismatches"] for item in utility_events
        ) == 0,
        "numpy_probability_pairwise_probe_is_bit_exact": probability_probe["pairwise_bit_mismatches"] == 0,
        "three_candidate_reports_pass_every_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "all_candidates_have_zero_cpu_guard_rows": all(
            sum(event["arithmetic_guard_rows"] for event in events) == 0
            for events in candidate_events
        ),
        "all_candidates_use_one_versioned_utility_and_probability_abi": all(
            len({event["arithmetic_abi_version"] for event in events}) == 1
            and len({event["arithmetic_abi_sha256"] for event in events}) == 1
            and len({event["probability_abi_version"] for event in events}) == 1
            and len({event["probability_abi_sha256"] for event in events}) == 1
            for events in candidate_events
        ),
        "all_candidates_cover_complete_public_workload": all(
            len(events) == 30
            and sum(event["chooser_rows"] for event in events) == 91_524
            and sum(event["utility_cells"] for event in events) == 133_075_896
            and sum(event["random_draws"] for event in events) == 2_745_720
            for events in candidate_events
        ),
        "three_independent_output_verifiers_pass": all(
            item["success"] and item["decision_cells_different"] == 0
            for item in exact
        ),
        "candidate_wins_every_matched_pair": summary["candidate_won_every_pair"],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    document = {
        "phase": 41,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "arithmetic_probe_rows": arithmetic_probe["rows"],
        "utility_shadow_cells": sum(
            item["shadow_utility_cells_compared"] for item in utility_events
        ),
        "utility_shadow_bit_mismatches": sum(
            item["shadow_utility_mismatches"] for item in utility_events
        ),
        "probability_probe_rows": probability_probe["pairwise_rows"],
        "probability_probe_cells": probability_probe["pairwise_cells"],
        "probability_pairwise_bit_mismatches": probability_probe["pairwise_bit_mismatches"],
        "candidate_guard_rows": [
            sum(event["arithmetic_guard_rows"] for event in events)
            for events in candidate_events
        ],
        "decision_cell_differences": [item["decision_cells_different"] for item in exact],
        "destination_logsum_max_abs": [item["diagnostic_max_abs"] for item in exact],
        "mode_logsum_max_abs": [item["mode_choice_logsum_max_abs"] for item in exact],
        "median_phase40_all_model_seconds": summary["median_baseline_all_model_seconds"],
        "median_phase41_all_model_seconds": summary["median_candidate_all_model_seconds"],
        "median_all_model_speedup": summary["median_speedup"],
        "median_trip_destination": next(
            item for item in summary["component_comparison"]
            if item["model_name"] == "trip_destination"
        ),
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 41 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
