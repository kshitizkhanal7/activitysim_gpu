"""Hash-chain the Phase 30 native bootstrap and independent oracle proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--live", type=Path, action="append", required=True)
    parser.add_argument("--native-hashes", type=Path, required=True)
    parser.add_argument("--legacy-hashes", type=Path, required=True)
    parser.add_argument("--native-hash-resident", type=Path, required=True)
    parser.add_argument("--native-hash-live", type=Path, required=True)
    parser.add_argument("--legacy-resident", type=Path, required=True)
    parser.add_argument("--legacy-live", type=Path, required=True)
    parser.add_argument("--arithmetic", type=Path, required=True)
    parser.add_argument("--phase29", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 3 or len(args.live) != 3:
        raise SystemExit("Phase 30 requires exactly three resident and three live reports")

    resident_pairs = [read(path) for path in args.input]
    live_pairs = [read(path) for path in args.live]
    native_hashes, native_hash_digest = read(args.native_hashes)
    legacy_hashes, legacy_hash_digest = read(args.legacy_hashes)
    native_hash_resident, native_hash_resident_digest = read(
        args.native_hash_resident
    )
    native_hash_live, native_hash_live_digest = read(args.native_hash_live)
    legacy_resident, legacy_resident_digest = read(args.legacy_resident)
    legacy_live, legacy_live_digest = read(args.legacy_live)
    arithmetic, arithmetic_digest = read(args.arithmetic)
    phase29, phase29_digest = read(args.phase29)
    resident = [item[0] for item in resident_pairs]
    live = [item[0] for item in live_pairs]

    graph_medians = [item["median_seconds"] for item in resident]
    bootstrap_seconds = [item["native_bootstrap_seconds"] for item in resident]
    cold_seconds = [item["elapsed_seconds_including_resume_overhead"] for item in live]
    graph_median = statistics.median(graph_medians)
    cold_median = statistics.median(cold_seconds)
    phase29_graph = phase29["median_of_process_medians_seconds"]
    phase29_cold = phase29["cold_activitysim_median_seconds"]

    manifests = [
        manifest for report in resident for manifest in report["native_abi_programs"]
    ]
    schemas_by_purpose: dict[str, set[str]] = {}
    ir_by_purpose: dict[str, set[str]] = {}
    for manifest in manifests:
        schemas_by_purpose.setdefault(manifest["purpose"], set()).add(
            manifest["schema_sha256"]
        )
        ir_by_purpose.setdefault(manifest["purpose"], set()).add(
            manifest["ir_sha256"]
        )

    native_programs = native_hashes["programs"]
    legacy_programs = legacy_hashes["programs"]
    exact_program_hashes = (
        len(native_programs) == len(legacy_programs) == 6
        and all(
            left["rows"] == right["rows"]
            and left["dtype"] == right["dtype"]
            and left["sha256"] == right["sha256"]
            for left, right in zip(native_programs, legacy_programs)
        )
    )

    proof = {
        "three_independent_native_processes": len(resident) == 3,
        "all_native_process_gates_passed": all(
            all(item["proof_gates"].values()) for item in resident
        ),
        "all_live_gates_passed": all(
            all(item["proof_gates"].values()) for item in live
        ),
        "all_15_resident_replays_bit_exact": sum(
            item["measured_runs"] for item in resident
        ) == 15
        and all(
            item["final_tdd_mismatches"] == 0
            and all(replay["logsum_bit_mismatches"] == 0 for replay in item["replays"])
            for item in resident
        ),
        "all_1_210_124_dense_rows_avoided": all(
            item["dense_preprocessor_rows_avoided"] == item["mode_logsum_rows"]
            == 1_210_124
            for item in resident
        ),
        "zero_dense_preprocessor_reads": all(
            manifest["dense_preprocessor_rows_read"] == 0
            and manifest["dense_preprocessor_values_read"] == 0
            for manifest in manifests
        ),
        "three_stable_purpose_schemas_and_irs": (
            set(schemas_by_purpose) == {"work", "school", "univ"}
            and set(ir_by_purpose) == {"work", "school", "univ"}
            and all(len(values) == 1 for values in schemas_by_purpose.values())
            and all(len(values) == 1 for values in ir_by_purpose.values())
        ),
        "declared_native_abi_complete": all(
            manifest["terms"] == 315
            and len(manifest["alternatives"]) == 21
            and manifest["float_row_sources"] == 10
            and manifest["int_row_sources"] == 31
            and manifest["scalar_sources"] == 48
            and manifest["skim_sources"] == 209
            and manifest["skim_coordinate_groups"] == 6
            for manifest in manifests
        ),
        "native_and_legacy_logsums_byte_identical": (
            exact_program_hashes
            and native_hashes["aggregate_sha256"]
            == legacy_hashes["aggregate_sha256"]
        ),
        "independent_hash_run_gates_passed": (
            all(native_hash_resident["proof_gates"].values())
            and all(native_hash_live["proof_gates"].values())
            and all(legacy_resident["proof_gates"].values())
            and all(legacy_live["proof_gates"].values())
        ),
        "both_exponential_policies_reference_exact": (
            all(arithmetic["proof_gates"].values())
            and all(item["choice_mismatches"] == 0 for item in arithmetic["policies"])
        ),
        "no_postseal_host_traffic_or_cpu_fallback": all(
            item["modeled_host_to_device_bytes_after_seal"] == 0
            and item["intermediate_modeled_device_to_host_bytes"] == 0
            and item["runtime_telemetry"]["modeled_cpu_fallbacks"] == 0
            for item in resident
        ),
        "cold_runtime_not_regressed": cold_median <= phase29_cold,
    }

    result = {
        "phase": 30,
        "scope": (
            "native reviewed-IR/raw-metadata bootstrap with no ActivitySim dense "
            "logsum preprocessor in the production path"
        ),
        "workload": {
            "households": 50_000,
            "mode_logsum_rows": resident[0]["mode_logsum_rows"],
            "scheduled_tours": resident[0]["scheduled_tours"],
            "programs": resident[0]["programs"],
            "terms_per_program": 315,
            "alternatives": 21,
        },
        "complete_graph_process_medians_seconds": graph_medians,
        "median_of_process_medians_seconds": graph_median,
        "minimum_complete_graph_seconds": min(
            min(item["seconds"]) for item in resident
        ),
        "native_bootstrap_process_seconds": bootstrap_seconds,
        "native_bootstrap_median_seconds": statistics.median(bootstrap_seconds),
        "cold_activitysim_process_seconds": cold_seconds,
        "cold_activitysim_median_seconds": cold_median,
        "cold_seconds_change_vs_phase29": cold_median - phase29_cold,
        "cold_percent_change_vs_phase29": (cold_median / phase29_cold - 1) * 100,
        "resident_percent_change_vs_phase29": (
            graph_median / phase29_graph - 1
        ) * 100,
        "compact_input_state_bytes": resident[0]["compact_input_state_bytes"],
        "removed_captured_row_bytes": resident[0]["removed_captured_row_bytes"],
        "compact_state_reduction_ratio": resident[0][
            "compact_state_reduction_ratio"
        ],
        "dense_preprocessor_rows_read": 0,
        "dense_preprocessor_values_read": 0,
        "dense_preprocessor_rows_avoided": resident[0][
            "dense_preprocessor_rows_avoided"
        ],
        "schema_sha256_by_purpose": {
            key: next(iter(value)) for key, value in sorted(schemas_by_purpose.items())
        },
        "ir_sha256_by_purpose": {
            key: next(iter(value)) for key, value in sorted(ir_by_purpose.items())
        },
        "qualification": {
            "full_public_process_replays": sum(
                item["measured_runs"] for item in resident
            ),
            "native_legacy_logsum_programs_hashed": len(native_programs),
            "native_legacy_aggregate_sha256": native_hashes["aggregate_sha256"],
            "arithmetic_policies": [
                {
                    "exp_policy": item["exp_policy"],
                    "choice_mismatches": item["choice_mismatches"],
                    "detected_ambiguities": item["detected_ambiguities"],
                }
                for item in arithmetic["policies"]
            ],
            "production_boundary_map_entries": resident[0][
                "qualified_boundary_map_entries"
            ],
        },
        "proof_gates": proof,
        "source_hashes": {
            **{
                str(path): digest
                for path, (_, digest) in zip(args.input, resident_pairs)
            },
            **{
                str(path): digest for path, (_, digest) in zip(args.live, live_pairs)
            },
            str(args.native_hashes): native_hash_digest,
            str(args.legacy_hashes): legacy_hash_digest,
            str(args.native_hash_resident): native_hash_resident_digest,
            str(args.native_hash_live): native_hash_live_digest,
            str(args.legacy_resident): legacy_resident_digest,
            str(args.legacy_live): legacy_live_digest,
            str(args.arithmetic): arithmetic_digest,
            str(args.phase29): phase29_digest,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(proof.values()):
        raise SystemExit("Phase 30 summary proof gate failed")


if __name__ == "__main__":
    main()
