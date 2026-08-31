"""Assemble and validate Phase 45's model-wide destination qualification."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "school_location",
    "workplace_location",
    "joint_tour_destination",
    "non_mandatory_tour_destination",
    "atwork_subtour_destination",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase45-p45final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    summary = read(results / "phase45-p45final-summary.json")
    candidates = [
        read(results / f"phase45-p45final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(results / f"phase45-p45final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    sampling = [item["phase45_modelwide_sampling"] for item in candidates]
    choice = [item["phase45_modelwide_choice"] for item in candidates]
    component = {
        item["model_name"]: item for item in summary["component_comparison"]
    }
    target_baselines = [
        sum(run["baseline_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    target_candidates = [
        sum(run["candidate_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    guard_rows = [
        sum(event["exact_guard_rows"] for event in run) for run in sampling
    ]
    gates = {
        "three_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "all_nineteen_sampling_programs_cover_public_workload": all(
            len(run) == 19
            and sum(event["chooser_rows"] for event in run) == 201_390
            and sum(event["utility_cells"] for event in run) == 274_223_637
            and sum(event["random_draws"] for event in run) == 6_041_700
            for run in sampling
        ),
        "all_five_final_choice_families_use_compact_runtime": all(
            run["calls"] == 19
            and run["chooser_rows"] == 201_390
            and run["alternative_rows"] == 4_696_676
            and set(run["groups"]) == set(TARGETS)
            for run in choice
        ),
        "all_dense_programs_use_cuda_without_fallback": all(
            all(not event["fallback"] for event in run) for run in sampling
        ),
        "sparse_exact_boundary_guard_is_below_five_percent": all(
            guarded < 0.05 * 201_390 for guarded in guard_rows
        ),
        "three_independent_verifiers_have_exact_decisions": all(
            item["success"] and item["decision_cells_different"] == 0
            for item in exact
        ),
        "all_declared_logsums_remain_inside_tolerance": all(
            item["diagnostic_max_abs"] <= item["diagnostic_gate"]
            and item["mode_choice_logsum_max_abs"] <= item["mode_choice_logsum_gate"]
            and item["diagnostic_columns"]["school_location_logsum"]["max_abs"] <= 1e-5
            and item["diagnostic_columns"]["workplace_location_logsum"]["max_abs"] <= 1e-5
            for item in exact
        ),
        "target_component_aggregate_improves_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(target_baselines, target_candidates)
        ),
        "complete_model_improves_in_every_pair": summary["candidate_won_every_pair"],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    document = {
        "phase": 45,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "matched_pairs": 3,
        "target_families": list(TARGETS),
        "sampling_programs": 19,
        "chooser_rows": 201_390,
        "dense_utility_cells": 274_223_637,
        "keyed_random_draws": 6_041_700,
        "sampled_alternative_rows": 4_696_676,
        "exact_guard_rows": guard_rows,
        "exact_guard_fraction": [value / 201_390 for value in guard_rows],
        "decision_cell_differences": [
            item["decision_cells_different"] for item in exact
        ],
        "diagnostic_max_abs": {
            name: [item["diagnostic_columns"][name]["max_abs"] for item in exact]
            for name in (
                "school_location_logsum",
                "workplace_location_logsum",
                "destination_logsum",
                "mode_choice_logsum",
            )
        },
        "target_component_aggregate_seconds": {
            "phase44": target_baselines,
            "phase45": target_candidates,
            "median_phase44": statistics.median(target_baselines),
            "median_phase45": statistics.median(target_candidates),
            "median_speedup": statistics.median(target_baselines)
            / statistics.median(target_candidates),
        },
        "component_medians": {name: component[name] for name in TARGETS},
        "complete_model": {
            "median_phase44_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase45_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_reduction_percent": summary["median_reduction_percent"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
        },
        "older_regular_activitysim_context": {
            "median_seconds": 206.6,
            "phase45_median_seconds": summary["median_candidate_all_model_seconds"],
            "descriptive_speedup": 206.6 / summary["median_candidate_all_model_seconds"],
            "warning": "Older established control, not a fresh Phase 45 matched pair.",
        },
        "claim_boundary": (
            "Phase 45 compiles the reviewed public destination-sampling utility "
            "surface and inverse-CDF selection on CUDA for five model families. "
            "ActivitySim retains keyed random draws; NumPy adjudicates only "
            "measured CDF-boundary rows. Final sampled-choice utility remains "
            "the authoritative Sharrow CPU evaluator inside a compact runtime."
        ),
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 45 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
