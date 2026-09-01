"""Assemble and fail-closed validate Phase 46's persistent destination service."""

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
        default=ROOT / "benchmark-results" / "phase46-p46final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    summary = read(results / "phase46-p46final-summary.json")
    candidates = [
        read(results / f"phase46-p46final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(results / f"phase46-p46final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    sampling = [item["phase45_modelwide_sampling"] for item in candidates]
    choice = [item["phase45_modelwide_choice"] for item in candidates]
    services = [item["phase46_persistent_destination"] for item in candidates]
    prewarms = [item["phase46_prewarm"] for item in candidates]
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
        "all_nineteen_sampling_programs_use_persistent_runtime": all(
            len(run) == 19
            and sum(event["chooser_rows"] for event in run) == 201_390
            and sum(event["utility_cells"] for event in run) == 274_223_637
            and all(event["runtime"] == "phase46_persistent" for event in run)
            for run in sampling
        ),
        "all_five_final_choice_families_use_persistent_runtime": all(
            run["calls"] == 19
            and run["chooser_rows"] == 201_390
            and run["alternative_rows"] == 4_696_676
            and set(run["groups"]) == set(TARGETS)
            and all(
                event["runtime"] == "phase46_persistent"
                for event in run["events"]
            )
            for run in choice
        ),
        "gpu_rng_covers_sampling_and_final_choice_exact_workload": all(
            service["random_calls"] == 38
            and service["random_rows"] == 402_780
            and service["random_draw_values"] == 6_243_090
            for service in services
        ),
        "persistent_workspace_covers_largest_program_under_one_gibibyte": all(
            service["cell_capacity"] >= 26_496 * 1_454
            and service["row_capacity"] >= 26_496
            and service["workspace_bytes"] < 1024**3
            for service in services
        ),
        "all_four_public_programs_are_cold_prewarmed": all(
            prewarm["programs"] == 4
            and prewarm["new_programs_compiled"] == 4
            and prewarm["seconds"] > 0
            for prewarm in prewarms
        ),
        "sparse_exact_boundary_guard_is_below_five_percent": all(
            guarded < 0.05 * 201_390 for guarded in guard_rows
        ),
        "three_independent_verifiers_have_exact_decisions": all(
            item["success"] and item["decision_cells_different"] == 0
            for item in exact
        ),
        "all_seven_published_outputs_are_byte_identical": all(
            len(item["byte_identical_outputs"]) == 7 for item in exact
        ),
        "all_declared_logsums_remain_inside_tolerance": all(
            item["diagnostic_max_abs"] <= item["diagnostic_gate"]
            and item["mode_choice_logsum_max_abs"] <= item["mode_choice_logsum_gate"]
            and item["diagnostic_columns"]
            .get("school_location_logsum", {"max_abs": 0.0})["max_abs"]
            <= 1e-5
            and item["diagnostic_columns"]
            .get("workplace_location_logsum", {"max_abs": 0.0})["max_abs"]
            <= 1e-5
            for item in exact
        ),
        "target_component_aggregate_improves_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(target_baselines, target_candidates)
        ),
        "complete_model_lifecycle_improves_in_every_pair": summary[
            "candidate_won_every_pair"
        ],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    median_target_baseline = statistics.median(target_baselines)
    median_target_candidate = statistics.median(target_candidates)
    document = {
        "phase": 46,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "matched_pairs": 3,
        "target_families": list(TARGETS),
        "sampling_programs": 19,
        "chooser_rows": 201_390,
        "dense_utility_cells": 274_223_637,
        "gpu_keyed_random_calls": [item["random_calls"] for item in services],
        "gpu_keyed_random_rows": [item["random_rows"] for item in services],
        "gpu_keyed_random_values": [
            item["random_draw_values"] for item in services
        ],
        "workspace_bytes": [item["workspace_bytes"] for item in services],
        "prewarm_seconds": [item["seconds"] for item in prewarms],
        "exact_guard_rows": guard_rows,
        "exact_guard_fraction": [value / 201_390 for value in guard_rows],
        "decision_cell_differences": [
            item["decision_cells_different"] for item in exact
        ],
        "target_component_aggregate_seconds": {
            "phase45": target_baselines,
            "phase46": target_candidates,
            "median_phase45": median_target_baseline,
            "median_phase46": median_target_candidate,
            "median_seconds_saved": median_target_baseline
            - median_target_candidate,
            "median_reduction_percent": 100
            * (median_target_baseline - median_target_candidate)
            / median_target_baseline,
            "median_speedup": median_target_baseline / median_target_candidate,
        },
        "component_medians": {name: component[name] for name in TARGETS},
        "complete_model_lifecycle": {
            "median_phase45_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase46_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_reduction_percent": summary["median_reduction_percent"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
            "includes_phase46_prewarm": True,
        },
        "claim_boundary": (
            "Phase 46 keeps the nineteen reviewed public destination samplers "
            "inside a prewarmed persistent CUDA service, generates ActivitySim's "
            "keyed MT19937 draws exactly on GPU, evaluates each exponential once, "
            "and uses ActivitySim's authoritative Numba implementation only for "
            "measured sparse probability-boundary rows. Final sampled-choice "
            "utility remains Sharrow on CPU. Lifecycle timing charges cold prewarm."
        ),
        "known_component_tradeoff": (
            "The small joint-tour destination family regressed in the median, "
            "while the five-family aggregate and complete lifecycle won every pair."
        ),
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 46 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
