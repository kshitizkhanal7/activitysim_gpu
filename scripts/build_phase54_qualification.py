"""Build the Phase 54 sharded, matched-pair qualification artifact."""

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


def _runtime(report):
    return report["phase54_device_owned_destination_packet"]


def _components(pre, post):
    return sum(
        float(pre["model_timing_seconds"][name])
        for name in DESTINATION_COMPONENTS[:2]
    ) + sum(
        float(post["model_timing_seconds"][name])
        for name in DESTINATION_COMPONENTS[2:]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase53-qualification", type=Path, required=True)
    parser.add_argument("--baseline-pre", type=Path, nargs=3, required=True)
    parser.add_argument("--baseline-post", type=Path, nargs=3, required=True)
    parser.add_argument("--candidate-pre", type=Path, nargs=3, required=True)
    parser.add_argument("--candidate-post", type=Path, nargs=3, required=True)
    parser.add_argument("--full-runs", type=Path, nargs=3, required=True)
    args = parser.parse_args()

    phase53 = _read(args.phase53_qualification)
    baseline_pres = [_read(path) for path in args.baseline_pre]
    baseline_posts = [_read(path) for path in args.baseline_post]
    candidate_pres = [_read(path) for path in args.candidate_pre]
    candidate_posts = [_read(path) for path in args.candidate_post]
    full_runs = [_read(path) for path in args.full_runs]
    pairs = []
    schema_hashes = set()
    for number, (baseline_pre, baseline_post, candidate_pre, candidate_post) in enumerate(
        zip(baseline_pres, baseline_posts, candidate_pres, candidate_posts), 1
    ):
        baseline_events = (
            baseline_pre["phase53_device_resident_destination_data_plane"]["events"]
            + baseline_post["phase53_device_resident_destination_data_plane"]["events"]
        )
        candidate_pre_runtime = _runtime(candidate_pre)
        candidate_post_runtime = _runtime(candidate_post)
        candidate_events = candidate_pre_runtime["events"] + candidate_post_runtime["events"]
        schema_hashes.update(event["schema_sha256"] for event in candidate_events)
        baseline_service = sum(float(event["total_seconds"]) for event in baseline_events)
        candidate_service = sum(float(event["total_seconds"]) for event in candidate_events)
        baseline_prepare = sum(float(event["compact_prepare_seconds"]) for event in baseline_events)
        candidate_prepare = sum(float(event["compact_prepare_seconds"]) for event in candidate_events)
        baseline_components = _components(baseline_pre, baseline_post)
        candidate_components = _components(candidate_pre, candidate_post)
        phase53_projected_all = float(phase53["pairs"][number - 1]["phase53_projected_all_model_seconds"])
        projected_all = phase53_projected_all - baseline_components + candidate_components
        services = [candidate_pre["phase54_persistent_service"], candidate_post["phase54_persistent_service"]]
        pairs.append(
            {
                "pair": number,
                "phase53_destination_service_seconds": baseline_service,
                "phase54_destination_service_seconds": candidate_service,
                "destination_service_speedup": baseline_service / candidate_service,
                "destination_service_reduction_percent": 100.0 * (baseline_service - candidate_service) / baseline_service,
                "phase53_packet_prepare_seconds": baseline_prepare,
                "phase54_packet_prepare_seconds": candidate_prepare,
                "packet_prepare_speedup": baseline_prepare / candidate_prepare,
                "packet_prepare_reduction_percent": 100.0 * (baseline_prepare - candidate_prepare) / baseline_prepare,
                "phase53_five_destination_components_seconds": baseline_components,
                "phase54_five_destination_components_seconds": candidate_components,
                "five_destination_components_speedup": baseline_components / candidate_components,
                "five_destination_components_reduction_percent": 100.0 * (baseline_components - candidate_components) / baseline_components,
                "phase53_projected_all_model_seconds": phase53_projected_all,
                "phase54_projected_all_model_seconds": projected_all,
                "calls": int(candidate_pre_runtime["calls"] + candidate_post_runtime["calls"]),
                "rows": int(candidate_pre_runtime["rows"] + candidate_post_runtime["rows"]),
                "owners": int(candidate_pre_runtime["owners"] + candidate_post_runtime["owners"]),
                "normal_calls": sum(item["phase54_normal_calls"] for item in services),
                "normal_values": sum(item["phase54_normal_values"] for item in services),
                "lease_publishes": sum(item["phase54_sample_lease_publishes"] for item in services),
                "lease_consumes": sum(item["phase54_sample_lease_consumes"] for item in services),
                "all_shard_gates_pass": all(candidate_pre["proof_gates"].values()) and all(candidate_post["proof_gates"].values()),
            }
        )

    def med(name):
        return _median([item[name] for item in pairs])

    b_service = med("phase53_destination_service_seconds")
    c_service = med("phase54_destination_service_seconds")
    b_prepare = med("phase53_packet_prepare_seconds")
    c_prepare = med("phase54_packet_prepare_seconds")
    b_components = med("phase53_five_destination_components_seconds")
    c_components = med("phase54_five_destination_components_seconds")
    projected_all = med("phase54_projected_all_model_seconds")
    regular_cpu_seconds = 205.4
    phase52_full_seconds = 136.9912310000509
    full_elapsed = [float(item["elapsed_seconds_including_resume_overhead"]) for item in full_runs]
    full_model_steps = [float(item["activitysim_all_model_steps_seconds"]) for item in full_runs]
    full_service = [float(item["phase54_device_owned_destination_packet"]["total_seconds"]) for item in full_runs]
    full_components = [
        sum(float(item["model_timing_seconds"][name]) for name in DESTINATION_COMPONENTS)
        for item in full_runs
    ]
    measured_all = _median(full_elapsed)
    full_run_evidence = []
    for number, item in enumerate(full_runs, 1):
        runtime = item["phase54_device_owned_destination_packet"]
        service = item["phase46_persistent_destination"]
        full_run_evidence.append(
            {
                "run": number,
                "elapsed_seconds_including_resume_overhead": float(item["elapsed_seconds_including_resume_overhead"]),
                "activitysim_all_model_steps_seconds": float(item["activitysim_all_model_steps_seconds"]),
                "five_destination_components_seconds": sum(float(item["model_timing_seconds"][name]) for name in DESTINATION_COMPONENTS),
                "destination_service_seconds": float(runtime["total_seconds"]),
                "calls": int(runtime["calls"]),
                "rows": int(runtime["rows"]),
                "owners": int(runtime["owners"]),
                "normal_calls": int(service["phase54_normal_calls"]),
                "normal_values": int(service["phase54_normal_values"]),
                "lease_publishes": int(service["phase54_sample_lease_publishes"]),
                "lease_consumes": int(service["phase54_sample_lease_consumes"]),
                "workspace_bytes": int(service["workspace_bytes"]),
                "cache_max_abs_difference": float(item["cache_max_abs_difference"]),
                "all_proof_gates_pass": all(item["proof_gates"].values()),
            }
        )
    report = {
        "phase": 54,
        "benchmark": "public prototype_mtc_extended, 50,000 households, 1,454 zones",
        "method": "three matched Phase 53/54 shard pairs plus three fresh monolithic Phase 54 public-benchmark runs",
        "claim_boundary": "destination improvements are matched-pair measurements; complete-model Phase 54 is a fresh three-run measured median compared with the separately measured regular ActivitySim and Phase 52 medians",
        "pairs": pairs,
        "median_phase53_destination_service_seconds": b_service,
        "median_phase54_destination_service_seconds": c_service,
        "median_destination_service_speedup": b_service / c_service,
        "median_destination_service_reduction_percent": 100.0 * (b_service - c_service) / b_service,
        "median_phase53_packet_prepare_seconds": b_prepare,
        "median_phase54_packet_prepare_seconds": c_prepare,
        "median_packet_prepare_speedup": b_prepare / c_prepare,
        "median_packet_prepare_reduction_percent": 100.0 * (b_prepare - c_prepare) / b_prepare,
        "median_phase53_five_destination_components_seconds": b_components,
        "median_phase54_five_destination_components_seconds": c_components,
        "median_five_destination_components_speedup": b_components / c_components,
        "median_five_destination_components_reduction_percent": 100.0 * (b_components - c_components) / b_components,
        "median_phase54_projected_all_model_seconds": projected_all,
        "projected_speedup_vs_regular_activitysim_cpu": regular_cpu_seconds / projected_all,
        "projected_reduction_vs_regular_activitysim_cpu_percent": 100.0 * (regular_cpu_seconds - projected_all) / regular_cpu_seconds,
        "full_run_evidence": full_run_evidence,
        "median_phase54_measured_elapsed_seconds": measured_all,
        "median_phase54_measured_model_steps_seconds": _median(full_model_steps),
        "median_phase54_measured_destination_service_seconds": _median(full_service),
        "median_phase54_measured_five_destination_components_seconds": _median(full_components),
        "measured_speedup_vs_regular_activitysim_cpu": regular_cpu_seconds / measured_all,
        "measured_reduction_vs_regular_activitysim_cpu_percent": 100.0 * (regular_cpu_seconds - measured_all) / regular_cpu_seconds,
        "measured_speedup_vs_phase52": phase52_full_seconds / measured_all,
        "measured_reduction_vs_phase52_percent": 100.0 * (phase52_full_seconds - measured_all) / phase52_full_seconds,
        "max_phase54_workspace_bytes": max(item["workspace_bytes"] for item in full_run_evidence),
        "max_full_run_cache_abs_difference": max(item["cache_max_abs_difference"] for item in full_run_evidence),
        "schema_sha256": sorted(schema_hashes),
    }
    report["proof_gates"] = {
        "three_matched_pairs": len(pairs) == 3,
        "all_six_shards_pass_replication_and_runtime_gates": all(item["all_shard_gates_pass"] for item in pairs),
        "complete_public_destination_workload_covered_every_pair": all(item["calls"] == 19 and item["rows"] == 4_696_676 and item["owners"] == 201_390 for item in pairs),
        "all_exact_controlled_normal_streams_generated_on_gpu": all(item["normal_calls"] == 19 and item["normal_values"] == 1_208_340 for item in pairs),
        "all_sampler_leases_published_and_consumed": all(item["lease_publishes"] == 19 and item["lease_consumes"] == 19 for item in pairs),
        "phase54_destination_service_wins_all_three_pairs": all(item["phase54_destination_service_seconds"] < item["phase53_destination_service_seconds"] for item in pairs),
        "phase54_packet_preparation_wins_all_three_pairs": all(item["phase54_packet_prepare_seconds"] < item["phase53_packet_prepare_seconds"] for item in pairs),
        "phase54_five_components_win_all_three_pairs": all(item["phase54_five_destination_components_seconds"] < item["phase53_five_destination_components_seconds"] for item in pairs),
        "single_hash_verified_destination_program": len(schema_hashes) == 1,
        "median_destination_service_below_four_seconds": c_service < 4.0,
        "median_packet_preparation_reduced_at_least_seventy_percent": 100.0 * (b_prepare - c_prepare) / b_prepare >= 70.0,
        "three_successful_monolithic_runs": len(full_run_evidence) == 3 and all(item["all_proof_gates_pass"] for item in full_run_evidence),
        "complete_public_destination_workload_covered_every_full_run": all(item["calls"] == 19 and item["rows"] == 4_696_676 and item["owners"] == 201_390 for item in full_run_evidence),
        "all_full_runs_generate_exact_controlled_normal_streams_on_gpu": all(item["normal_calls"] == 19 and item["normal_values"] == 1_208_340 for item in full_run_evidence),
        "all_full_runs_publish_and_consume_sampler_leases": all(item["lease_publishes"] == 19 and item["lease_consumes"] == 19 for item in full_run_evidence),
        "measured_complete_model_beats_phase52": measured_all < phase52_full_seconds,
        "measured_complete_model_beats_regular_activitysim_cpu_by_thirty_percent": 100.0 * (regular_cpu_seconds - measured_all) / regular_cpu_seconds >= 30.0,
    }
    report["success"] = all(report["proof_gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
