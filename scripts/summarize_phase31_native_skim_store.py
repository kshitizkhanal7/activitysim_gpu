"""Hash-chain the Phase 31 persistent native skim-store proof."""

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
    parser.add_argument("--resident", type=Path, action="append", required=True)
    parser.add_argument("--live", type=Path, action="append", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--phase31-hashes", type=Path, required=True)
    parser.add_argument("--phase30-native-hashes", type=Path, required=True)
    parser.add_argument("--phase30-legacy-hashes", type=Path, required=True)
    parser.add_argument("--phase30", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.resident) != 3 or len(args.live) != 3:
        raise SystemExit("Phase 31 requires exactly three resident and live reports")

    resident_pairs = [read(path) for path in args.resident]
    live_pairs = [read(path) for path in args.live]
    build, build_digest = read(args.build)
    phase31_hashes, phase31_hash_digest = read(args.phase31_hashes)
    phase30_native_hashes, phase30_native_hash_digest = read(
        args.phase30_native_hashes
    )
    phase30_legacy_hashes, phase30_legacy_hash_digest = read(
        args.phase30_legacy_hashes
    )
    phase30, phase30_digest = read(args.phase30)
    resident = [item[0] for item in resident_pairs]
    live = [item[0] for item in live_pairs]

    activitysim_seconds = [
        item["elapsed_seconds_including_resume_overhead"] for item in live
    ]
    cold_boundary_seconds = [
        item["cold_component_seconds_including_scheduler_initialization"]
        for item in live
    ]
    scheduler_seconds = [item["scheduler_initialization_seconds"] for item in live]
    store_seconds = [item["native_skim_store_load_seconds"] for item in live]
    verified_read_seconds = [
        item["native_skim_store"]["verified_read_seconds"] for item in resident
    ]
    upload_seconds = [
        item["native_skim_store"]["device_upload_seconds"] for item in resident
    ]
    resident_seconds = [item["median_seconds"] for item in resident]
    activitysim_median = statistics.median(activitysim_seconds)
    cold_boundary_median = statistics.median(cold_boundary_seconds)
    resident_median = statistics.median(resident_seconds)
    phase30_cold = phase30["cold_activitysim_median_seconds"]
    phase30_resident = phase30["median_of_process_medians_seconds"]
    exact_hash = phase31_hashes["aggregate_sha256"]

    proof = {
        "three_independent_full_processes": len(resident) == len(live) == 3,
        "all_live_and_resident_gates_passed": all(
            all(item["proof_gates"].values()) for item in resident + live
        ),
        "all_15_resident_replays_bit_exact": sum(
            item["measured_runs"] for item in resident
        ) == 15
        and all(
            item["final_tdd_mismatches"] == 0
            and all(
                replay["logsum_bit_mismatches"] == 0
                for replay in item["replays"]
            )
            for item in resident
        ),
        "all_live_outputs_exact": all(
            item["tdd_mismatches"] == 0
            and item["start_mismatches"] == 0
            and item["end_mismatches"] == 0
            and item["integrated_tdd_mismatches"] == 0
            for item in live
        ),
        "every_payload_byte_verified_in_every_process": all(
            item["native_skim_store"]["verified_payload_bytes"]
            == item["native_skim_store"]["payload_bytes"]
            == 6_198_588_112
            for item in resident
        ),
        "complete_deduplicated_skim_contract": all(
            item["native_skim_store"]["logical_bindings"] == 209
            and item["native_skim_store"]["physical_cubes"] == 149
            and item["native_skim_store"]["zone_count"] == 1454
            for item in resident
        ),
        "stable_store_and_contract_hashes": len({
            (
                item["native_skim_store"]["payload_sha256"],
                item["native_skim_store"]["skim_contract_sha256"],
            )
            for item in resident
        }) == 1,
        "sharrow_and_omx_inventory_never_materialized": all(
            item["native_skim_stub_calls"] == 6
            and item["native_network_info_bypass_calls"] == 1
            and item["native_network_load_bypass_calls"] == 1
            for item in live
        ),
        "all_boundary_choices_adjudicated_on_device": all(
            item["device_boundary_adjudications"] == item["exact_boundary_rows"] == 57
            and item["boundary_logsum_download_bytes"] == 0
            for item in live
        ),
        "phase31_phase30_native_and_legacy_logsums_byte_identical": (
            phase31_hashes["phase"] == 31
            and len(phase31_hashes["programs"]) == 6
            and exact_hash
            == phase30_native_hashes["aggregate_sha256"]
            == phase30_legacy_hashes["aggregate_sha256"]
        ),
        "immutable_store_build_gates_passed": all(build["proof_gates"].values()),
        "wins_every_cold_process_against_phase30_median": all(
            value < phase30_cold for value in cold_boundary_seconds
        ),
        "phase30_proof_remains_green": all(phase30["proof_gates"].values()),
    }

    result = {
        "phase": 31,
        "scope": (
            "persistent, byte-verified native skim artifact through exact public "
            "MTC mandatory scheduling, with no Sharrow skim materialization"
        ),
        "workload": {
            "households": 50_000,
            "mode_logsum_rows": resident[0]["mode_logsum_rows"],
            "scheduled_tours": resident[0]["scheduled_tours"],
            "programs": resident[0]["programs"],
            "terms_per_program": 315,
            "alternatives": 21,
            "logical_skim_bindings": 209,
            "physical_skim_cubes": 149,
            "zones": 1454,
        },
        "cold_activitysim_process_seconds": activitysim_seconds,
        "cold_activitysim_median_seconds": activitysim_median,
        "scheduler_initialization_process_seconds": scheduler_seconds,
        "scheduler_initialization_median_seconds": statistics.median(
            scheduler_seconds
        ),
        "cold_component_with_scheduler_process_seconds": cold_boundary_seconds,
        "cold_component_with_scheduler_median_seconds": cold_boundary_median,
        "cold_component_seconds_saved_vs_phase30": phase30_cold - cold_boundary_median,
        "cold_component_percent_reduction_vs_phase30": (
            1 - cold_boundary_median / phase30_cold
        ) * 100,
        "cold_component_speedup_vs_phase30": phase30_cold / cold_boundary_median,
        "native_store_load_process_seconds": store_seconds,
        "native_store_load_median_seconds": statistics.median(store_seconds),
        "verified_payload_read_process_seconds": verified_read_seconds,
        "verified_payload_read_median_seconds": statistics.median(
            verified_read_seconds
        ),
        "overlapped_device_upload_process_seconds": upload_seconds,
        "overlapped_device_upload_median_seconds": statistics.median(upload_seconds),
        "resident_graph_process_medians_seconds": resident_seconds,
        "resident_graph_median_seconds": resident_median,
        "resident_percent_change_vs_phase30": (
            resident_median / phase30_resident - 1
        ) * 100,
        "artifact": {
            "format": build["manifest"]["format"],
            "payload_bytes": build["manifest"]["payload_bytes"],
            "payload_sha256": build["manifest"]["payload_sha256"],
            "skim_contract_sha256": build["manifest"]["skim_contract_sha256"],
            "source_omx_sha256": build["manifest"]["source_omx_sha256"],
            "source_land_use_sha256": build["manifest"]["source_land_use_sha256"],
        },
        "qualification": {
            "full_public_resident_replays": sum(
                item["measured_runs"] for item in resident
            ),
            "aggregate_logsum_sha256": exact_hash,
            "device_boundary_adjudications_per_process": [
                item["device_boundary_adjudications"] for item in live
            ],
            "boundary_download_bytes_per_process": [
                item["boundary_logsum_download_bytes"] for item in live
            ],
        },
        "proof_gates": proof,
        "source_hashes": {
            **{
                str(path): digest
                for path, (_, digest) in zip(args.resident, resident_pairs)
            },
            **{
                str(path): digest for path, (_, digest) in zip(args.live, live_pairs)
            },
            str(args.build): build_digest,
            str(args.phase31_hashes): phase31_hash_digest,
            str(args.phase30_native_hashes): phase30_native_hash_digest,
            str(args.phase30_legacy_hashes): phase30_legacy_hash_digest,
            str(args.phase30): phase30_digest,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(proof.values()):
        raise SystemExit("Phase 31 summary proof gate failed")


if __name__ == "__main__":
    main()
