"""Assemble and fail-closed validate Phase 47 qualification evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results"
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
    summary = read(RESULTS / "phase47-p47final-summary.json")
    candidates = [
        read(RESULTS / f"phase47-p47final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(RESULTS / f"phase47-p47final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    shadow = read(RESULTS / "phase47-p47shadow2-gpu.json")
    runtimes = [item["phase47_device_final_choice"] for item in candidates]
    target_baselines = [
        sum(run["baseline_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    target_candidates = [
        sum(run["candidate_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    phase46_final_seconds = []
    for trial in range(1, 4):
        prior = read(RESULTS / f"phase46-p46final-gpu-{trial}.json")
        phase46_final_seconds.append(sum(
            event["total_seconds"]
            for event in prior["phase45_modelwide_choice"]["events"]
        ))
    phase47_final_seconds = [item["seconds"] for item in runtimes]
    shadow_events = shadow["phase47_device_final_choice"]["events"]
    gates = {
        "three_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "all_nineteen_calls_cover_the_public_workload": all(
            item["calls"] == 19
            and item["chooser_rows"] == 201_390
            and item["alternative_rows"] == 4_696_676
            for item in runtimes
        ),
        "all_four_final_programs_and_widths_are_prewarmed": all(
            item["phase47_prewarm"]["programs"] == 4
            and item["phase47_prewarm"]["widths"] == [21, 25, 29, 30]
            for item in candidates
        ),
        "cold_numba_compile_is_removed": all(
            item["phase46_prewarm"]["exact_guard_runtime"] == "numpy"
            for item in candidates
        ),
        "exhaustive_live_utility_shadow_is_bit_identical": (
            len(shadow_events) == 19
            and sum(item["alternative_rows"] for item in shadow_events) == 4_696_676
            and all(item["utility_shadow_bit_mismatches"] == 0 for item in shadow_events)
            and all(item["utility_shadow_max_abs"] == 0 for item in shadow_events)
        ),
        "production_sparse_guard_has_zero_pre_guard_mismatches": all(
            item["guard_rows"] == 7
            and item["pre_guard_mismatches"] == 0
            for item in runtimes
        ),
        "three_independent_verifiers_find_exact_outputs": all(
            item["success"]
            and item["decision_cells_different"] == 0
            and len(item["byte_identical_outputs"]) == 7
            for item in exact
        ),
        "five_target_components_improve_in_every_pair": all(
            candidate < baseline
            for baseline, candidate in zip(target_baselines, target_candidates)
        ),
        "complete_lifecycle_improves_in_every_pair": summary["candidate_won_every_pair"],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    median_old = statistics.median(phase46_final_seconds)
    median_new = statistics.median(phase47_final_seconds)
    document = {
        "phase": 47,
        "benchmark": summary["benchmark"],
        "households": 50_000,
        "zones": 1_454,
        "matched_pairs": 3,
        "calls": 19,
        "chooser_rows": 201_390,
        "sampled_alternative_rows": 4_696_676,
        "final_choice_runtime": {
            "phase46_seconds": phase46_final_seconds,
            "phase47_seconds": phase47_final_seconds,
            "median_phase46_seconds": median_old,
            "median_phase47_seconds": median_new,
            "median_seconds_saved": median_old - median_new,
            "median_speedup": median_old / median_new,
        },
        "five_target_component_aggregate": {
            "phase46_seconds": target_baselines,
            "phase47_seconds": target_candidates,
            "median_phase46_seconds": statistics.median(target_baselines),
            "median_phase47_seconds": statistics.median(target_candidates),
        },
        "complete_model_lifecycle": {
            "median_phase46_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase47_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_reduction_percent": summary["median_reduction_percent"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
            "includes_all_cold_prewarms": True,
        },
        "utility_shadow_bit_mismatches": sum(
            item["utility_shadow_bit_mismatches"] for item in shadow_events
        ),
        "decision_cell_differences": [
            item["decision_cells_different"] for item in exact
        ],
        "proof_gates": gates,
        "success": all(gates.values()),
        "claim_boundary": (
            "Phase 47 compiles reviewed sampled final-choice utility and guarded "
            "selection to CUDA. Exact logsums and normalization remain authoritative "
            "NumPy on the compact transferred surface; ActivitySim still orchestrates."
        ),
    }
    if not document["success"]:
        raise RuntimeError(
            "Phase 47 qualification failed: "
            + repr([name for name, value in gates.items() if not value])
        )
    output = RESULTS / "phase47-p47final-qualification.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
