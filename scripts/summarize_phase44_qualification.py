"""Assemble and validate Phase 44's compact final-choice qualification."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def baseline_final_seconds(report) -> float:
    return sum(
        event["simulation_profile"]["interaction_seconds"]
        for event in report["phase35_trip_destination_stages"]["events"]
    )


def candidate_final_seconds(report) -> float:
    return sum(
        event["total_seconds"]
        for event in report["phase44_compact_final_simulation"]["events"]
    )


def candidate_stage_seconds(report, name: str) -> float:
    return sum(
        event[name]
        for event in report["phase44_compact_final_simulation"]["events"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase44-p44final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    summary = read(results / "phase44-p44final-summary.json")
    baselines = [
        read(results / f"phase44-p44final-base-{trial}.json")
        for trial in range(1, 4)
    ]
    candidates = [
        read(results / f"phase44-p44final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(results / f"phase44-p44final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    compact = [item["phase44_compact_final_simulation"] for item in candidates]
    baseline_final = [baseline_final_seconds(item) for item in baselines]
    candidate_final = [candidate_final_seconds(item) for item in candidates]
    trip = next(
        item
        for item in summary["component_comparison"]
        if item["model_name"] == "trip_destination"
    )
    gates = {
        "three_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "all_thirty_compact_programs_cover_public_workload": all(
            item["calls"] == 30
            and item["chooser_rows"] == 91_524
            and item["alternative_rows"] == 2_094_156
            for item in compact
        ),
        "reviewed_sixteen_slot_fourteen_term_abi_used_everywhere": all(
            len(item["events"]) == 30
            and all(
                event["expression_slots"] == 16
                and event["effective_utility_terms"] == 14
                and event["max_alternatives"] <= 30
                for event in item["events"]
            )
            for item in compact
        ),
        "three_independent_output_verifiers_are_byte_exact": all(
            item["success"]
            and item["decision_cells_different"] == 0
            and item["diagnostic_max_abs"] == 0.0
            and item["mode_choice_logsum_max_abs"] == 0.0
            and len(item["byte_identical_outputs"]) == 7
            for item in exact
        ),
        "compact_final_boundary_improves_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(baseline_final, candidate_final)
        ),
        "trip_destination_component_improves_in_every_pair": summary[
            "candidate_won_target_component_every_pair"
        ],
        "complete_model_improves_in_every_pair": summary[
            "candidate_won_every_pair"
        ],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    document = {
        "phase": 44,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "matched_pairs": 3,
        "final_programs": 30,
        "trip_rows": 91_524,
        "sampled_destination_rows": 2_094_156,
        "expression_slots": 16,
        "effective_utility_terms": 14,
        "decision_cell_differences": [
            item["decision_cells_different"] for item in exact
        ],
        "destination_logsum_max_abs": [item["diagnostic_max_abs"] for item in exact],
        "mode_logsum_max_abs": [
            item["mode_choice_logsum_max_abs"] for item in exact
        ],
        "final_choice_boundary_seconds": {
            "phase43": baseline_final,
            "phase44": candidate_final,
            "median_phase43": statistics.median(baseline_final),
            "median_phase44": statistics.median(candidate_final),
            "median_seconds_saved": statistics.median(baseline_final)
            - statistics.median(candidate_final),
            "median_speedup": statistics.median(baseline_final)
            / statistics.median(candidate_final),
            "phase44_stage_medians": {
                name: statistics.median(
                    candidate_stage_seconds(report, name) for report in candidates
                )
                for name in (
                    "frame_seconds",
                    "utility_seconds",
                    "padding_seconds",
                    "probability_seconds",
                    "choice_seconds",
                )
            },
        },
        "median_trip_destination_component": trip,
        "complete_model": {
            "median_phase43_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase44_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
        },
        "claim_boundary": (
            "Phase 44 removes generic final-choice table mechanics around the "
            "authoritative Sharrow CPU utility compiler; it retains the earlier "
            "GPU sampling/logsum runtime and does not claim that the final "
            "16-slot expression evaluator itself executes on the GPU."
        ),
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 44 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
