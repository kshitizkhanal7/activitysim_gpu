"""Build the Phase 53 sharded, matched-pair qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


DESTINATION_COMPONENTS = (
    "school_location",
    "workplace_location",
    "joint_tour_destination",
    "non_mandatory_tour_destination",
    "atwork_subtour_destination",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values):
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs=3, required=True)
    parser.add_argument("--pre", type=Path, nargs=3, required=True)
    parser.add_argument("--post", type=Path, nargs=3, required=True)
    args = parser.parse_args()

    baselines = [_read(path) for path in args.baseline]
    pres = [_read(path) for path in args.pre]
    posts = [_read(path) for path in args.post]
    pairs = []
    schema_hashes = set()
    for number, (baseline, pre, post) in enumerate(zip(baselines, pres, posts), 1):
        pre_runtime = pre["phase53_device_resident_destination_data_plane"]
        post_runtime = post["phase53_device_resident_destination_data_plane"]
        baseline_runtime = baseline["phase52_persistent_tiled_destination"]
        events = pre_runtime["events"] + post_runtime["events"]
        schema_hashes.update(event["schema_sha256"] for event in events)
        baseline_service = float(baseline_runtime["total_seconds"])
        candidate_service = float(pre_runtime["total_seconds"] + post_runtime["total_seconds"])
        baseline_prepare = sum(
            float(event["compact_prepare_seconds"])
            for event in baseline_runtime["events"]
        )
        candidate_prepare = sum(
            float(event["compact_prepare_seconds"]) for event in events
        )
        baseline_components = sum(
            float(baseline["model_timing_seconds"][name])
            for name in DESTINATION_COMPONENTS
        )
        candidate_components = (
            sum(float(pre["model_timing_seconds"][name]) for name in DESTINATION_COMPONENTS[:2])
            + sum(float(post["model_timing_seconds"][name]) for name in DESTINATION_COMPONENTS[2:])
        )
        projected_all_model = (
            float(baseline["elapsed_seconds_including_resume_overhead"])
            - baseline_components
            + candidate_components
        )
        pairs.append(
            {
                "pair": number,
                "phase52_destination_service_seconds": baseline_service,
                "phase53_destination_service_seconds": candidate_service,
                "destination_service_speedup": baseline_service / candidate_service,
                "destination_service_reduction_percent": 100.0 * (baseline_service - candidate_service) / baseline_service,
                "phase52_packet_prepare_seconds": baseline_prepare,
                "phase53_packet_prepare_seconds": candidate_prepare,
                "packet_prepare_speedup": baseline_prepare / candidate_prepare,
                "packet_prepare_reduction_percent": 100.0 * (baseline_prepare - candidate_prepare) / baseline_prepare,
                "phase52_five_destination_components_seconds": baseline_components,
                "phase53_five_destination_components_seconds": candidate_components,
                "five_destination_components_speedup": baseline_components / candidate_components,
                "five_destination_components_reduction_percent": 100.0 * (baseline_components - candidate_components) / baseline_components,
                "phase52_measured_all_model_seconds": float(baseline["elapsed_seconds_including_resume_overhead"]),
                "phase53_projected_all_model_seconds": projected_all_model,
                "all_model_projected_reduction_percent": 100.0 * (float(baseline["elapsed_seconds_including_resume_overhead"]) - projected_all_model) / float(baseline["elapsed_seconds_including_resume_overhead"]),
                "calls": int(pre_runtime["calls"] + post_runtime["calls"]),
                "rows": int(pre_runtime["rows"] + post_runtime["rows"]),
                "owners": int(pre_runtime["owners"] + post_runtime["owners"]),
                "all_shard_gates_pass": all(pre["proof_gates"].values()) and all(post["proof_gates"].values()),
            }
        )

    median_baseline_service = _median([item["phase52_destination_service_seconds"] for item in pairs])
    median_candidate_service = _median([item["phase53_destination_service_seconds"] for item in pairs])
    median_baseline_prepare = _median([item["phase52_packet_prepare_seconds"] for item in pairs])
    median_candidate_prepare = _median([item["phase53_packet_prepare_seconds"] for item in pairs])
    median_baseline_components = _median([item["phase52_five_destination_components_seconds"] for item in pairs])
    median_candidate_components = _median([item["phase53_five_destination_components_seconds"] for item in pairs])
    median_projected_all = _median([item["phase53_projected_all_model_seconds"] for item in pairs])
    regular_cpu_seconds = 205.4
    report = {
        "phase": 53,
        "benchmark": "public prototype_mtc_extended, 50,000 households, 1,454 zones",
        "method": "three matched Phase 52/53 pairs; Phase 53 split at the previously verified mandatory-scheduling checkpoint because this Windows host repeatedly refused an unrelated 79 MiB pandas allocation",
        "claim_boundary": "destination timings are measured; complete-model Phase 53 time is a component-substitution projection, not a successful monolithic timing",
        "pairs": pairs,
        "median_phase52_destination_service_seconds": median_baseline_service,
        "median_phase53_destination_service_seconds": median_candidate_service,
        "median_destination_service_speedup": median_baseline_service / median_candidate_service,
        "median_destination_service_reduction_percent": 100.0 * (median_baseline_service - median_candidate_service) / median_baseline_service,
        "median_phase52_packet_prepare_seconds": median_baseline_prepare,
        "median_phase53_packet_prepare_seconds": median_candidate_prepare,
        "median_packet_prepare_speedup": median_baseline_prepare / median_candidate_prepare,
        "median_packet_prepare_reduction_percent": 100.0 * (median_baseline_prepare - median_candidate_prepare) / median_baseline_prepare,
        "median_phase52_five_destination_components_seconds": median_baseline_components,
        "median_phase53_five_destination_components_seconds": median_candidate_components,
        "median_five_destination_components_speedup": median_baseline_components / median_candidate_components,
        "median_five_destination_components_reduction_percent": 100.0 * (median_baseline_components - median_candidate_components) / median_baseline_components,
        "median_phase52_measured_all_model_seconds": _median([item["phase52_measured_all_model_seconds"] for item in pairs]),
        "median_phase53_projected_all_model_seconds": median_projected_all,
        "projected_speedup_vs_regular_activitysim_cpu": regular_cpu_seconds / median_projected_all,
        "projected_reduction_vs_regular_activitysim_cpu_percent": 100.0 * (regular_cpu_seconds - median_projected_all) / regular_cpu_seconds,
        "schema_sha256": sorted(schema_hashes),
    }
    report["proof_gates"] = {
        "three_matched_pairs": len(pairs) == 3,
        "all_six_shards_pass_replication_and_runtime_gates": all(item["all_shard_gates_pass"] for item in pairs),
        "complete_public_destination_workload_covered_every_pair": all(item["calls"] == 19 and item["rows"] == 4_696_676 and item["owners"] == 201_390 for item in pairs),
        "phase53_destination_service_wins_all_three_pairs": all(item["phase53_destination_service_seconds"] < item["phase52_destination_service_seconds"] for item in pairs),
        "phase53_packet_preparation_wins_all_three_pairs": all(item["phase53_packet_prepare_seconds"] < item["phase52_packet_prepare_seconds"] for item in pairs),
        "phase53_five_components_win_all_three_pairs": all(item["phase53_five_destination_components_seconds"] < item["phase52_five_destination_components_seconds"] for item in pairs),
        "single_hash_verified_destination_program": len(schema_hashes) == 1,
        "median_destination_service_below_six_seconds": median_candidate_service < 6.0,
        "median_packet_preparation_reduced_at_least_sixty_percent": 100.0 * (median_baseline_prepare - median_candidate_prepare) / median_baseline_prepare >= 60.0,
    }
    report["success"] = all(report["proof_gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
