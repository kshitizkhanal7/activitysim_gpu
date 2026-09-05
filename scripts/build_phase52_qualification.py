"""Build and validate the consolidated Phase 52 qualification artifact."""

from __future__ import annotations

import argparse
import hashlib
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


def _metrics(baseline, candidate, candidate_label="phase52_seconds"):
    baseline = [float(value) for value in baseline]
    candidate = [float(value) for value in candidate]
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
    if incremental.get("phase") != 52 or incremental.get("baseline") != "phase51":
        raise ValueError("incremental input is not the Phase 51/52 summary")
    if cpu.get("phase") != 52 or cpu.get("baseline") != "activitysim":
        raise ValueError("CPU input is not the regular ActivitySim/Phase 52 summary")
    if len(incremental["runs"]) != 3 or len(cpu["runs"]) != 3:
        raise ValueError("Phase 52 qualification requires three pairs in both experiments")

    root = args.incremental_summary.resolve().parents[1]
    result_dir = root / "benchmark-results"
    candidate_reports = [
        _read(result_dir / f"phase52-p52final-gpu-{i}.json") for i in range(1, 4)
    ]
    baseline_reports = [
        _read(result_dir / f"phase52-p52final-base-{i}.json") for i in range(1, 4)
    ]
    incremental_exact = [
        _read(result_dir / f"phase52-p52final-exact-{i}.json") for i in range(1, 4)
    ]
    cpu_exact = [
        _read(result_dir / f"phase52-p52cpu-exact-{i}.json") for i in range(1, 4)
    ]
    phase52 = [item["phase52_persistent_tiled_destination"] for item in candidate_reports]
    phase51 = [item["phase51_fused_compact_destination_utility"] for item in baseline_reports]

    target = _metrics(
        [_target(run, "baseline") for run in incremental["runs"]],
        [_target(run, "candidate") for run in incremental["runs"]],
    )
    lifecycle = _metrics(
        [run["baseline_all_model_seconds"] for run in incremental["runs"]],
        [run["candidate_all_model_seconds"] for run in incremental["runs"]],
    )
    cumulative = _metrics(
        [run["baseline_all_model_seconds"] for run in cpu["runs"]],
        [run["candidate_all_model_seconds"] for run in cpu["runs"]],
    )
    cumulative_target = _metrics(
        [_target(run, "baseline") for run in cpu["runs"]],
        [_target(run, "candidate") for run in cpu["runs"]],
    )
    kernel_comparison = _metrics(
        [item["row_owner_kernel_seconds"] + item["fused_kernel_seconds"] for item in phase51],
        [item["row_owner_kernel_seconds"] + item["fused_kernel_seconds"] for item in phase52],
        "phase52_tiled_kernel_seconds",
    )
    service_comparison = _metrics(
        [item["total_seconds"] for item in phase51],
        [item["total_seconds"] for item in phase52],
    )

    source_path = root / "src" / "choiceforge" / "kernels" / "phase52_public_destination_tile4.cu"
    # Match the runtime's platform-independent text fingerprint. Path.read_text
    # normalizes CRLF/LF so a Windows checkout verifies the same reviewed source.
    source_sha256 = hashlib.sha256(
        source_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    first = phase52[0]
    report = {
        "phase": 52,
        "benchmark": "public Prototype MTC extended, full 34-step ActivitySim model",
        "households": 50_000,
        "zones": 1_454,
        "incremental_matched_pairs": 3,
        "regular_activitysim_matched_pairs": 3,
        "calls_per_run": 19,
        "owners_per_run": 201_390,
        "sampled_alternative_rows_per_run": 4_696_676,
        "phase51_to_phase52_five_destination_components": target,
        "phase51_to_phase52_complete_model_lifecycle": lifecycle,
        "phase51_to_phase52_tiled_kernel_pipeline": kernel_comparison,
        "phase51_to_phase52_instrumented_service": service_comparison,
        "regular_activitysim_complete_model_lifecycle": cumulative,
        "regular_activitysim_five_destination_components": cumulative_target,
        "persistent_runtime_contract": {
            "tile_rows": 4,
            "checked_in_cuda_source": str(source_path.relative_to(root)).replace("\\", "/"),
            "source_sha256": source_sha256,
            "semantic_plan_cache_hits_per_run": [item["semantic_plan_cache_hits"] for item in phase52],
            "native_plan_cache_hits_per_run": [item["native_plan_cache_hits"] for item in phase52],
            "utility_workspace_hits_per_run": [item["utility_workspace_hits"] for item in phase52],
            "packet_workspace_hits_per_run": [item["packet_workspace_hits"] for item in phase52],
            "row_owner_workspace_hits_per_run": [item["row_owner_workspace_hits"] for item in phase52],
            "prewarm_seconds": [item["phase52_prewarm"]["seconds"] for item in candidate_reports],
            "early_release_bytes": [item["phase52_early_release_freed_bytes"] for item in candidate_reports],
        },
        "memory_and_transfer_contract": {
            "aggregate_dense_device_abi_bytes_eliminated": int(first["dense_device_abi_bytes_eliminated"]),
            "aggregate_dense_host_pack_bytes_avoided": int(first["dense_host_pack_bytes_avoided"]),
            "compact_upload_bytes": int(first["compact_upload_bytes"]),
            "net_upload_bytes_avoided": int(first["net_upload_bytes_avoided"]),
            "row_owner_device_bytes": int(first["row_owner_device_bytes"]),
            "minimal_bootstrap_bytes": int(first["minimal_bootstrap_bytes"]),
        },
        "decision_cell_differences": [item["decision_cells_different"] for item in incremental_exact],
        "regular_activitysim_decision_cell_differences": [item["decision_cells_different"] for item in cpu_exact],
    }
    report["proof_gates"] = {
        "all_incremental_runtime_gates_pass": all(
            all(item["proof_gates"].values()) for item in candidate_reports
        ),
        "all_nineteen_calls_cover_the_public_workload": all(
            item["calls"] == 19 and item["rows"] == 4_696_676 and item["owners"] == 201_390
            for item in phase52
        ),
        "all_calls_use_four_row_tiles": all(item["tile_rows"] == [4] for item in phase52),
        "checked_in_source_hash_matches_every_run": all(
            item["phase52_prewarm"]["available"]
            and item["phase52_prewarm"]["source_sha256"] == source_sha256
            and all(event["schema_sha256"] == source_sha256 for event in tiled["events"])
            for item, tiled in zip(candidate_reports, phase52)
        ),
        "no_timed_call_compiles_the_fused_kernel": all(
            not event["fused_kernel_compiled"] for tiled in phase52 for event in tiled["events"]
        ),
        "semantic_and_native_plans_reused": all(
            item["semantic_plan_cache_hits"] >= 9 and item["native_plan_cache_hits"] >= 9
            for item in phase52
        ),
        "all_three_device_workspaces_reused": all(
            item["utility_workspace_hits"] >= 16
            and item["packet_workspace_hits"] >= 152
            and item["row_owner_workspace_hits"] >= 16
            for item in phase52
        ),
        "workspaces_released_before_trip_destination": all(
            item["phase52_early_release_calls"] == 1
            and item["phase52_early_release_freed_bytes"] > 100_000_000
            for item in candidate_reports
        ),
        "dense_device_abi_remains_eliminated": all(
            item["all_dense_device_abis_eliminated"]
            and item["dense_device_abi_bytes_eliminated"] > 1_900_000_000
            and item["fallback_calls"] == 0
            for item in phase52
        ),
        "tiled_kernel_wins_every_incremental_pair": kernel_comparison["won_every_pair"],
        "instrumented_service_wins_every_incremental_pair": service_comparison["won_every_pair"],
        "five_destination_components_win_every_incremental_pair": target["won_every_pair"],
        "complete_lifecycle_wins_every_incremental_pair": lifecycle["won_every_pair"],
        "three_incremental_verifiers_find_zero_changed_decisions": all(
            item["success"] and item["decision_cells_different"] == 0 for item in incremental_exact
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
        "Phase 52 converts Phase 51's per-call fused path into a hash-verified, prewarmed, "
        "four-row tiled service with persistent plans and reusable device workspaces. It wins "
        "the tiled kernel, instrumented service, five destination components, and complete "
        "model lifecycle in all three matched Phase 51/52 pairs."
    )
    report["claim_boundary"] = (
        "Incremental gains are attributed only to Phase 52 versus already accelerated Phase 51. "
        "The larger regular-ActivitySim gains are cumulative across Phases 1-52. Published modeled "
        "decisions are exact; declared floating logsum diagnostics remain within verifier tolerances."
    )
    if not report["success"]:
        failed = [key for key, value in report["proof_gates"].items() if not value]
        raise RuntimeError("Phase 52 qualification failed: " + ", ".join(failed))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
