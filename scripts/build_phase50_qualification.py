"""Build and validate the consolidated Phase 50 qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


TARGET_COMPONENTS = (
    "school_location",
    "workplace_location",
    "joint_tour_destination",
    "non_mandatory_tour_destination",
    "atwork_subtour_destination",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _target_seconds(run, side):
    values = run[f"{side}_component_seconds"]
    return sum(float(values[name]) for name in TARGET_COMPONENTS)


def _median_metrics(baseline, candidate):
    base = float(median(baseline))
    gpu = float(median(candidate))
    return {
        "baseline_seconds": baseline,
        "phase50_seconds": candidate,
        "median_baseline_seconds": base,
        "median_phase50_seconds": gpu,
        "median_seconds_saved": base - gpu,
        "median_reduction_percent": 100.0 * (base - gpu) / base,
        "median_speedup": base / gpu,
        "won_every_pair": all(gpu_value < base_value for base_value, gpu_value in zip(baseline, candidate)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental-summary", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    incremental = _read(args.incremental_summary)
    cpu = _read(args.cpu_summary)
    if incremental.get("phase") != 50 or incremental.get("baseline") != "phase49":
        raise ValueError("incremental input is not the Phase 49/50 summary")
    if cpu.get("phase") != 50 or cpu.get("baseline") != "activitysim":
        raise ValueError("CPU input is not the regular ActivitySim/Phase 50 summary")
    if len(incremental["runs"]) != 3 or len(cpu["runs"]) != 3:
        raise ValueError("Phase 50 qualification requires three pairs in both experiments")

    root = args.incremental_summary.resolve().parents[1]
    reports = [_read(root / "benchmark-results" / f"phase50-p50final-gpu-{i}.json") for i in range(1, 4)]
    exact = [_read(root / "benchmark-results" / f"phase50-p50final-exact-{i}.json") for i in range(1, 4)]
    cpu_exact = [_read(root / "benchmark-results" / f"phase50-p50cpu-exact-{i}.json") for i in range(1, 4)]
    generated = [item["phase50_device_generated_destination_inputs"] for item in reports]

    target = _median_metrics(
        [_target_seconds(run, "baseline") for run in incremental["runs"]],
        [_target_seconds(run, "candidate") for run in incremental["runs"]],
    )
    lifecycle = _median_metrics(
        [float(run["baseline_all_model_seconds"]) for run in incremental["runs"]],
        [float(run["candidate_all_model_seconds"]) for run in incremental["runs"]],
    )
    lifecycle["won_pairs"] = sum(
        run["candidate_all_model_seconds"] < run["baseline_all_model_seconds"]
        for run in incremental["runs"]
    )
    cumulative = _median_metrics(
        [float(run["baseline_all_model_seconds"]) for run in cpu["runs"]],
        [float(run["candidate_all_model_seconds"]) for run in cpu["runs"]],
    )

    report = {
        "phase": 50,
        "benchmark": "public Prototype MTC extended, full 34-step ActivitySim model",
        "households": 50_000,
        "zones": 1_454,
        "incremental_matched_pairs": 3,
        "regular_activitysim_matched_pairs": 3,
        "calls_per_run": 19,
        "owners_per_run": 201_390,
        "sampled_alternative_rows_per_run": 4_696_676,
        "five_destination_components": target,
        "incremental_complete_model_lifecycle": lifecycle,
        "regular_activitysim_complete_model_lifecycle": cumulative,
        "compact_input_contract": {
            "float_row_sources": 10,
            "integer_row_sources": 31,
            "skim_coordinate_groups": 6,
            "dense_preprocessor_values_avoided": generated[0]["dense_preprocessor_values_avoided"],
            "dense_host_pack_bytes_avoided": generated[0]["dense_host_pack_bytes_avoided"],
            "compact_upload_bytes": generated[0]["compact_upload_bytes"],
            "net_upload_bytes_avoided": generated[0]["net_upload_bytes_avoided"],
            "upload_reduction_percent": 100.0 * generated[0]["net_upload_bytes_avoided"] / generated[0]["dense_host_pack_bytes_avoided"],
            "median_device_generate_seconds": median(item["device_generate_seconds"] for item in generated),
            "median_utility_kernel_seconds": median(item["utility_kernel_seconds"] for item in generated),
            "median_total_seconds": median(item["total_seconds"] for item in generated),
        },
        "decision_cell_differences": [item["decision_cells_different"] for item in exact],
        "regular_activitysim_decision_cell_differences": [item["decision_cells_different"] for item in cpu_exact],
        "diagnostic_bounds": {
            "destination_logsum_max_abs": max(item["diagnostic_columns"]["destination_logsum"]["max_abs"] for item in cpu_exact),
            "destination_logsum_gate": cpu_exact[0]["diagnostic_columns"]["destination_logsum"]["gate"],
            "mode_choice_logsum_max_abs": max(item["diagnostic_columns"]["mode_choice_logsum"]["max_abs"] for item in cpu_exact),
            "mode_choice_logsum_gate": cpu_exact[0]["diagnostic_columns"]["mode_choice_logsum"]["gate"],
        },
    }
    report["proof_gates"] = {
        "three_incremental_candidate_reports_pass_every_runtime_gate": all(
            all(item["proof_gates"].values()) for item in reports
        ),
        "all_nineteen_calls_cover_the_complete_public_workload": all(
            item["calls"] == 19 and item["rows"] == 4_696_676 and item["owners"] == 201_390
            for item in generated
        ),
        "complete_10_float_31_integer_6_group_abi_generated": all(
            item["all_source_abis_exact"] for item in generated
        ),
        "dense_host_preprocessor_pack_and_binding_eliminated": all(
            item["binding_resolution_calls"] == 0 and item["host_dense_pack_calls"] == 0
            for item in generated
        ),
        "net_upload_avoidance_exceeds_1_8_gb_per_run": all(
            item["net_upload_bytes_avoided"] > 1_800_000_000 for item in generated
        ),
        "zero_fallback_in_every_call": all(item["fallback_calls"] == 0 for item in generated),
        "five_destination_components_improve_in_every_incremental_pair": target["won_every_pair"],
        "three_incremental_verifiers_find_zero_changed_decisions": all(item["success"] and item["decision_cells_different"] == 0 for item in exact),
        "three_incremental_verifiers_find_all_seven_files_byte_identical": all(
            len(item["byte_identical_outputs"]) == 7 for item in exact
        ),
        "three_regular_activitysim_verifiers_find_zero_changed_decisions": all(item["success"] and item["decision_cells_different"] == 0 for item in cpu_exact),
        "regular_activitysim_lifecycle_improves_in_every_pair": cumulative["won_every_pair"],
        "both_summaries_declare_every_pair_exact": incremental["every_pair_exact"] and cpu["every_pair_exact"],
    }
    report["success"] = all(report["proof_gates"].values())
    report["claim_boundary"] = (
        "Phase 50 replaces the reviewed public destination-logsum pandas preprocessor and dense host transfer with compact owner/sample state and CUDA-generated inputs. Modeled decisions are exact and declared logsum diagnostics are bounded. The Phase 49 comparison isolates the incremental destination gain; the regular ActivitySim comparison measures the cumulative project runtime on this machine."
    )
    if not report["success"]:
        failed = [key for key, value in report["proof_gates"].items() if not value]
        raise RuntimeError("Phase 50 qualification failed: " + ", ".join(failed))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
