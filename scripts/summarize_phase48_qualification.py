"""Assemble and fail-closed validate Phase 48 qualification evidence."""

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
SHADOW_FIELDS = (
    "utility_shadow_bit_mismatches",
    "weight_shadow_bit_mismatches",
    "total_shadow_bit_mismatches",
    "probability_shadow_bit_mismatches",
    "choice_shadow_mismatches",
    "logsum_shadow_bit_mismatches",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    summary = read(RESULTS / "phase48-p48final-summary.json")
    baselines = [
        read(RESULTS / f"phase48-p48final-base-{trial}.json")
        for trial in range(1, 4)
    ]
    candidates = [
        read(RESULTS / f"phase48-p48final-gpu-{trial}.json")
        for trial in range(1, 4)
    ]
    exact = [
        read(RESULTS / f"phase48-p48final-exact-{trial}.json")
        for trial in range(1, 4)
    ]
    shadow = read(RESULTS / "phase48-p48shadow6-gpu.json")
    shadow_exact = read(RESULTS / "phase48-p48shadow6-output-verification.json")
    exp_scan = read(RESULTS / "phase48-exp-domain-scan.json")
    old_seconds = [item["phase47_device_final_choice"]["seconds"] for item in baselines]
    new_seconds = [item["phase48_resident_destination_graph"]["seconds"] for item in candidates]
    target_baselines = [
        sum(run["baseline_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    target_candidates = [
        sum(run["candidate_component_seconds"][name] for name in TARGETS)
        for run in summary["runs"]
    ]
    runtimes = [item["phase48_resident_destination_graph"] for item in candidates]
    rng = [item["phase46_persistent_destination"] for item in candidates]
    shadow_events = shadow["phase48_resident_destination_graph"]["events"]
    direct_median_old = statistics.median(old_seconds)
    direct_median_new = statistics.median(new_seconds)
    gates = {
        "three_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "resident_boundary_improves_in_every_matched_pair": all(
            new < old for old, new in zip(old_seconds, new_seconds)
        ),
        "all_nineteen_calls_cover_the_public_workload": all(
            item["calls"] == 19
            and item["chooser_rows"] == 201_390
            and item["alternative_rows"] == 4_696_676
            for item in runtimes
        ),
        "all_final_draws_resume_resident_keyed_mt19937": all(
            item["rng_resume_hits"] == 19 and item["rng_resume_misses"] == 0
            for item in rng
        ),
        "dense_final_utility_download_is_eliminated": all(
            item["dense_utility_download_bytes_avoided"] == 23_759_764
            and item["device_to_host_bytes"] < 3_000_000
            for item in runtimes
        ),
        "exhaustive_live_shadow_is_bit_identical_at_every_stage": (
            len(shadow_events) == 19
            and sum(item["alternative_rows"] for item in shadow_events) == 4_696_676
            and all(
                sum(item[field] for item in shadow_events) == 0
                for field in SHADOW_FIELDS
            )
        ),
        "all_float32_patterns_scanned_and_domain_table_is_complete": (
            exp_scan["success"]
            and exp_scan["input_bit_patterns_scanned"] == 2**32
            and exp_scan["finite_input_patterns"] == 4_278_190_080
            and exp_scan["domain_mismatches_before_correction"] == 73
            and exp_scan["domain_correction_table_exact"]
        ),
        "three_production_and_one_shadow_verifier_find_exact_outputs": (
            all(
                item["success"]
                and item["decision_cells_different"] == 0
                and len(item["byte_identical_outputs"]) == 7
                for item in exact
            )
            and shadow_exact["success"]
            and len(shadow_exact["byte_identical_outputs"]) == 7
        ),
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    document = {
        "phase": 48,
        "benchmark": summary["benchmark"],
        "households": 50_000,
        "zones": 1_454,
        "matched_pairs": 3,
        "calls": 19,
        "chooser_rows": 201_390,
        "sampled_alternative_rows": 4_696_676,
        "resident_final_boundary": {
            "phase47_seconds": old_seconds,
            "phase48_seconds": new_seconds,
            "median_phase47_seconds": direct_median_old,
            "median_phase48_seconds": direct_median_new,
            "median_seconds_saved": direct_median_old - direct_median_new,
            "median_reduction_percent": 100 * (direct_median_old - direct_median_new) / direct_median_old,
            "median_speedup": direct_median_old / direct_median_new,
            "won_every_pair": all(new < old for old, new in zip(old_seconds, new_seconds)),
        },
        "five_target_component_aggregate": {
            "phase47_seconds": target_baselines,
            "phase48_seconds": target_candidates,
            "median_phase47_seconds": statistics.median(target_baselines),
            "median_phase48_seconds": statistics.median(target_candidates),
        },
        "complete_model_lifecycle": {
            "median_phase47_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase48_seconds": summary["median_candidate_all_model_seconds"],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_reduction_percent": summary["median_reduction_percent"],
            "median_speedup": summary["median_speedup"],
            "candidate_won_every_pair": summary["candidate_won_every_pair"],
            "includes_all_cold_prewarms": True,
        },
        "transfer_contract": {
            "dense_utility_bytes_avoided": runtimes[0]["dense_utility_download_bytes_avoided"],
            "actual_device_to_host_bytes": runtimes[0]["device_to_host_bytes"],
        },
        "shadow_observed_utility_range": [
            min(item["shadow_utility_min"] for item in shadow_events),
            max(item["shadow_utility_max"] for item in shadow_events),
        ],
        "shadow_bit_mismatches": {
            field: sum(item[field] for item in shadow_events) for field in SHADOW_FIELDS
        },
        "exponential_domain_scan": exp_scan,
        "proof_gates": gates,
        "success": all(gates.values()),
        "claim_boundary": (
            "Phase 48 moves exact float32 exponential, pairwise normalization, "
            "one-draw selection, and compact logsum reduction onto the resident "
            "CUDA destination graph while continuing ActivitySim's keyed MT19937 "
            "state. ActivitySim still orchestrates; lifecycle timing is reported "
            "but is not used to overclaim a localized sub-second boundary gain."
        ),
    }
    if not document["success"]:
        raise RuntimeError(
            "Phase 48 qualification failed: "
            + repr([name for name, value in gates.items() if not value])
        )
    output = RESULTS / "phase48-p48final-qualification.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
