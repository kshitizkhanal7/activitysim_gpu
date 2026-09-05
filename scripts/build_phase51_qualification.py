"""Build and validate the consolidated Phase 51 qualification artifact."""

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


def _metrics(baseline, candidate, candidate_label="phase51_seconds"):
    base = float(median(baseline))
    gpu = float(median(candidate))
    return {
        "baseline_seconds": baseline,
        candidate_label: candidate,
        "median_baseline_seconds": base,
        f"median_{candidate_label}": gpu,
        "median_seconds_saved": base - gpu,
        "median_reduction_percent": 100.0 * (base - gpu) / base,
        "median_speedup": base / gpu,
        "won_pairs": sum(c < b for b, c in zip(baseline, candidate)),
        "won_every_pair": all(c < b for b, c in zip(baseline, candidate)),
    }


def _target(run, side):
    values = run[f"{side}_component_seconds"]
    return sum(float(values[name]) for name in TARGET_COMPONENTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental-summary", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    incremental = _read(args.incremental_summary)
    cpu = _read(args.cpu_summary)
    if incremental.get("phase") != 51 or incremental.get("baseline") != "phase50":
        raise ValueError("incremental input is not the Phase 50/51 summary")
    if cpu.get("phase") != 51 or cpu.get("baseline") != "activitysim":
        raise ValueError("CPU input is not the regular ActivitySim/Phase 51 summary")
    if len(incremental["runs"]) != 3 or len(cpu["runs"]) != 3:
        raise ValueError("Phase 51 qualification requires three pairs in both experiments")

    root = args.incremental_summary.resolve().parents[1]
    result_dir = root / "benchmark-results"
    candidate_reports = [
        _read(result_dir / f"phase51-p51final-gpu-{i}.json") for i in range(1, 4)
    ]
    baseline_reports = [
        _read(result_dir / f"phase51-p51final-base-{i}.json") for i in range(1, 4)
    ]
    incremental_exact = [
        _read(result_dir / f"phase51-p51final-exact-{i}.json") for i in range(1, 4)
    ]
    cpu_exact = [
        _read(result_dir / f"phase51-p51cpu-exact-{i}.json") for i in range(1, 4)
    ]
    fused = [item["phase51_fused_compact_destination_utility"] for item in candidate_reports]
    phase50 = [item["phase50_device_generated_destination_inputs"] for item in baseline_reports]

    target = _metrics(
        [_target(run, "baseline") for run in incremental["runs"]],
        [_target(run, "candidate") for run in incremental["runs"]],
    )
    lifecycle = _metrics(
        [float(run["baseline_all_model_seconds"]) for run in incremental["runs"]],
        [float(run["candidate_all_model_seconds"]) for run in incremental["runs"]],
    )
    cumulative = _metrics(
        [float(run["baseline_all_model_seconds"]) for run in cpu["runs"]],
        [float(run["candidate_all_model_seconds"]) for run in cpu["runs"]],
    )
    cumulative_target = _metrics(
        [_target(run, "baseline") for run in cpu["runs"]],
        [_target(run, "candidate") for run in cpu["runs"]],
    )
    old_kernel = [
        item["device_generate_seconds"] + item["utility_kernel_seconds"]
        for item in phase50
    ]
    fused_kernel = [
        item["row_owner_kernel_seconds"] + item["fused_kernel_seconds"]
        for item in fused
    ]
    kernel_comparison = _metrics(old_kernel, fused_kernel, "phase51_fused_seconds")
    service_comparison = _metrics(
        [item["total_seconds"] for item in phase50],
        [item["total_seconds"] for item in fused],
    )

    dense_bytes = int(fused[0]["dense_device_abi_bytes_eliminated"])
    row_owner_bytes = int(fused[0]["row_owner_device_bytes"])
    largest_call = max(fused[0]["events"], key=lambda item: item["rows"])
    old_upload = int(phase50[0]["compact_upload_bytes"])
    new_upload = int(fused[0]["compact_upload_bytes"])
    report = {
        "phase": 51,
        "benchmark": "public Prototype MTC extended, full 34-step ActivitySim model",
        "households": 50_000,
        "zones": 1_454,
        "incremental_matched_pairs": 3,
        "regular_activitysim_matched_pairs": 3,
        "calls_per_run": 19,
        "owners_per_run": 201_390,
        "sampled_alternative_rows_per_run": 4_696_676,
        "phase50_to_phase51_five_destination_components": target,
        "phase50_to_phase51_complete_model_lifecycle": lifecycle,
        "phase50_to_phase51_kernel_pipeline": kernel_comparison,
        "phase50_to_phase51_instrumented_service": service_comparison,
        "regular_activitysim_complete_model_lifecycle": cumulative,
        "regular_activitysim_five_destination_components": cumulative_target,
        "device_memory_contract": {
            "aggregate_dense_device_allocation_bytes_eliminated_across_19_calls": dense_bytes,
            "aggregate_row_owner_map_bytes_across_19_calls": row_owner_bytes,
            "aggregate_net_device_allocation_bytes_eliminated_across_19_calls": dense_bytes - row_owner_bytes,
            "largest_call_rows": int(largest_call["rows"]),
            "largest_call_dense_device_bytes_eliminated": int(largest_call["dense_device_abi_bytes_eliminated"]),
            "largest_call_row_owner_map_bytes": int(largest_call["row_owner_device_bytes"]),
            "largest_call_net_device_bytes_eliminated": int(largest_call["dense_device_abi_bytes_eliminated"] - largest_call["row_owner_device_bytes"]),
            "minimal_bootstrap_bytes_per_call": int(fused[0]["events"][0]["minimal_bootstrap_bytes"]),
            "aggregate_minimal_bootstrap_bytes_across_19_calls": int(fused[0]["minimal_bootstrap_bytes"]),
            "phase50_compact_upload_bytes": old_upload,
            "phase51_compact_upload_bytes": new_upload,
            "compact_upload_bytes_saved": old_upload - new_upload,
            "compact_upload_reduction_percent": 100.0 * (old_upload - new_upload) / old_upload,
            "dense_host_upload_bytes_avoided": int(fused[0]["net_upload_bytes_avoided"]),
        },
        "decision_cell_differences": [item["decision_cells_different"] for item in incremental_exact],
        "regular_activitysim_decision_cell_differences": [item["decision_cells_different"] for item in cpu_exact],
    }
    report["proof_gates"] = {
        "all_incremental_runtime_gates_pass": all(
            all(item["proof_gates"].values()) for item in candidate_reports
        ),
        "all_nineteen_calls_cover_the_public_workload": all(
            item["calls"] == 19
            and item["rows"] == 4_696_676
            and item["owners"] == 201_390
            for item in fused
        ),
        "complete_dense_device_row_abi_eliminated": all(
            item["all_dense_device_abis_eliminated"]
            and item["dense_device_abi_bytes_eliminated"] > 1_900_000_000
            for item in fused
        ),
        "minimal_bootstrap_stays_below_20_kb": all(
            item["minimal_bootstrap_bytes"] < 20_000 for item in fused
        ),
        "row_owner_is_exactly_one_int32_per_row": all(
            item["row_owner_device_bytes"] == item["rows"] * 4 for item in fused
        ),
        "zero_generator_and_fallback_calls": all(
            item["device_generate_seconds"] == 0
            and item["fallback_calls"] == 0
            and item["all_source_abis_exact"]
            for item in fused
        ),
        "compact_upload_reduced": new_upload < old_upload,
        "three_incremental_verifiers_find_zero_changed_decisions": all(
            item["success"] and item["decision_cells_different"] == 0
            for item in incremental_exact
        ),
        "three_incremental_verifiers_find_all_seven_files_byte_identical": all(
            len(item["byte_identical_outputs"]) == 7 for item in incremental_exact
        ),
        "three_regular_activitysim_verifiers_find_zero_changed_decisions": all(
            item["success"] and item["decision_cells_different"] == 0 for item in cpu_exact
        ),
        "regular_activitysim_lifecycle_improves_in_every_pair": cumulative["won_every_pair"],
        "regular_activitysim_destination_components_improve_in_every_pair": cumulative_target["won_every_pair"],
        "both_summaries_declare_every_pair_exact": incremental["every_pair_exact"] and cpu["every_pair_exact"],
    }
    report["success"] = all(report["proof_gates"].values())
    report["performance_interpretation"] = (
        "Phase 51 is qualified as a deterministic memory-capacity improvement. Its fused kernel pipeline is approximately performance-neutral against Phase 50 on this GPU, while first-use compilation makes the instrumented service modestly slower. The cumulative Phase 51 system is substantially faster than regular ActivitySim, but that cumulative gain belongs to Phases 1-51 rather than Phase 51 alone."
    )
    report["claim_boundary"] = (
        "Phase 51 fuses compact destination-state reconstruction into strict utility evaluation and removes the dense device row ABI for all nineteen reviewed public calls. Published decisions are exact. Incremental timing versus Phase 50 and cumulative timing versus regular pinned ActivitySim are reported separately; no incremental Phase 51 speedup is claimed."
    )
    if not report["success"]:
        failed = [key for key, value in report["proof_gates"].items() if not value]
        raise RuntimeError("Phase 51 qualification failed: " + ", ".join(failed))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
