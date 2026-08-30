"""Assemble and validate the Phase 42 promotion evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase42-p42final-qualification.json",
    )
    args = parser.parse_args()
    results = ROOT / "benchmark-results"
    summary = read(results / "phase42-p42final-summary.json")
    candidates = [read(results / f"phase42-p42final-gpu-{trial}.json") for trial in range(1, 4)]
    exact = [read(results / f"phase42-p42final-exact-{trial}.json") for trial in range(1, 4)]
    compiler_probe = read(results / "phase42-numeric-compiler-probe.json")
    telemetry = [item["phase42_numeric_compiler"] for item in candidates]
    stages = [item["phase35_trip_destination_stages"] for item in candidates]
    compiler_hashes = {item["numeric_abi_sha256"] for item in telemetry}
    compiler_versions = {item["compiler_version"] for item in telemetry}
    gates = {
        "numeric_compiler_probe_passes": compiler_probe["success"],
        "three_candidate_reports_pass_every_gate": all(
            all(item["proof_gates"].values()) for item in candidates
        ),
        "one_versioned_numeric_abi_in_all_candidates": (
            len(compiler_hashes) == 1
            and len(compiler_versions) == 1
            and next(iter(compiler_hashes)) == compiler_probe["qualified_numeric_abi_sha256"]
        ),
        "all_candidates_use_thirty_compact_directional_bundles": all(
            item["compact_directional_bundles"] == 30 for item in telemetry
        ),
        "all_candidates_compile_ten_logsum_contracts_then_reuse_twenty": all(
            item["logsum_contract_cache_misses"] == 10
            and item["logsum_contract_cache_hits"] == 20
            for item in telemetry
        ),
        "all_candidates_compile_ten_simulation_specs_then_reuse_twenty": all(
            item["simulation_spec_cache_misses"] == 10
            and item["simulation_spec_cache_hits"] == 20
            for item in telemetry
        ),
        "all_candidates_compile_ten_native_abis_then_reuse_twenty": all(
            item["native_codegen_cache"]["misses"] == 10
            and item["native_codegen_cache"]["hits"] == 20
            for item in telemetry
        ),
        "all_candidates_cover_complete_public_trip_workload": all(
            item["calls"] == 3
            and item["purposes"] == 30
            and item["trip_rows"] == 91_524
            and item["sample_rows"] == 2_094_156
            for item in stages
        ),
        "three_independent_output_verifiers_pass": all(
            item["success"] and item["decision_cells_different"] == 0
            for item in exact
        ),
        "candidate_wins_every_matched_pair": summary["candidate_won_every_pair"],
        "summary_declares_every_pair_exact": summary["every_pair_exact"],
    }
    trip = next(
        item for item in summary["component_comparison"]
        if item["model_name"] == "trip_destination"
    )
    stage_names = (
        "sampling_seconds",
        "preparation_seconds",
        "preprocessor_seconds",
        "logsums_seconds",
        "simulation_seconds",
        "total_seconds",
    )
    median_stages = {
        name: statistics.median(item[name] for item in stages) for name in stage_names
    }
    document = {
        "phase": 42,
        "benchmark": summary["benchmark"],
        "households": summary["households"],
        "alternatives": 1_454,
        "matched_pairs": summary["repetitions"],
        "compiler_version": next(iter(compiler_versions)),
        "numeric_abi_sha256": next(iter(compiler_hashes)),
        "compiler_probe_reduction_shapes": [
            item["term_count"] for item in compiler_probe["reduction_probes"]
        ],
        "compiler_probe_probability_shapes": [
            item["alternative_count"] for item in compiler_probe["probability_probes"]
        ],
        "decision_cell_differences": [item["decision_cells_different"] for item in exact],
        "destination_logsum_max_abs": [item["diagnostic_max_abs"] for item in exact],
        "mode_logsum_max_abs": [item["mode_choice_logsum_max_abs"] for item in exact],
        "median_phase41_all_model_seconds": summary["median_baseline_all_model_seconds"],
        "median_phase42_all_model_seconds": summary["median_candidate_all_model_seconds"],
        "median_all_model_seconds_saved": summary["median_seconds_saved"],
        "median_all_model_reduction_percent": summary["median_reduction_percent"],
        "median_all_model_speedup": summary["median_speedup"],
        "median_trip_destination": trip,
        "median_phase42_trip_destination_stages": median_stages,
        "aspirational_targets": {
            "trip_destination_under_8_seconds": trip["median_candidate_seconds"] < 8.0,
            "all_model_under_150_seconds": summary["median_candidate_all_model_seconds"] < 150.0,
            "note": "targets were optimization goals, not proof gates, and are reported without weakening them",
        },
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 42 qualification failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
