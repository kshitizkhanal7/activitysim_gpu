"""Build and fail-closed validate the Phase 56 qualification artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results"
OUTPUT = RESULTS / "phase56-p56formal-qualification.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    build_path = RESULTS / "phase56-cache-build.json"
    manifest_path = RESULTS / "phase56-skim-cache-manifest.json"
    summary_path = RESULTS / "phase56-p56formal-summary.json"
    cpu_summary_path = RESULTS / "phase52-p52cpu-summary.json"
    build = load(build_path)
    manifest = load(manifest_path)
    summary = load(summary_path)
    cpu_summary = load(cpu_summary_path)

    manifest_payload = dict(manifest)
    recorded_artifact_digest = manifest_payload.pop("artifact_sha256")
    manifest_payload.pop("qualification_seconds")
    computed_artifact_digest = hashlib.sha256(canonical(manifest_payload)).hexdigest()

    runs = []
    for trial, timing in enumerate(summary["runs"], start=1):
        candidate_path = RESULTS / f"phase56-p56formal-gpu-{trial}.json"
        exact_path = RESULTS / f"phase56-p56formal-exact-{trial}.json"
        candidate = load(candidate_path)
        exact = load(exact_path)
        cache = candidate["phase56_modelwide_resident_runtime"]
        runs.append(
            {
                "trial": trial,
                "phase55_seconds": timing["baseline_all_model_seconds"],
                "phase56_seconds_including_validation": timing[
                    "candidate_all_model_seconds"
                ],
                "phase56_model_steps_seconds": timing[
                    "candidate_model_steps_seconds"
                ],
                "cache_validation_seconds": cache["runtime_validation_seconds"],
                "initialize_landuse_seconds": timing["candidate_component_seconds"][
                    "initialize_landuse"
                ],
                "seconds_saved": timing["all_model_seconds_saved"],
                "speedup": timing["all_model_speedup"],
                "all_candidate_proof_gates_pass": all(
                    candidate["proof_gates"].values()
                ),
                "exact_output_verification_pass": bool(exact["success"]),
                "decision_cells_different": int(exact["decision_cells_different"]),
                "cache_contract_valid": bool(cache["all_checks_pass"]),
                "cache_artifact_sha256": cache["artifact_sha256"],
            }
        )

    gates = {
        "out_of_band_cache_build_and_full_digest_pass": (
            build["exit_code"] == 0 and all(build["proof_gates"].values())
        ),
        "checked_in_manifest_self_digest_valid": (
            recorded_artifact_digest == computed_artifact_digest
        ),
        "runtime_and_build_use_same_artifact": all(
            run["cache_artifact_sha256"] == recorded_artifact_digest for run in runs
        ),
        "all_three_runtime_cache_contracts_valid": all(
            run["cache_contract_valid"] for run in runs
        ),
        "all_three_initializations_below_three_seconds": all(
            run["initialize_landuse_seconds"] < 3.0 for run in runs
        ),
        "all_three_full_runs_pass_every_proof_gate": all(
            run["all_candidate_proof_gates_pass"] for run in runs
        ),
        "all_three_full_runs_are_output_exact": all(
            run["exact_output_verification_pass"]
            and run["decision_cells_different"] == 0
            for run in runs
        ),
        "whole_model_wins_all_three_pairs": all(
            run["seconds_saved"] > 0 for run in runs
        ),
        "phase56_lifecycle_median_below_130_seconds": (
            summary["median_candidate_all_model_seconds"] < 130.0
        ),
    }
    output = {
        "phase": 56,
        "contract": "choiceforge-phase56-replicated-modelwide-runtime-v1",
        "status": "qualified" if all(gates.values()) else "not_qualified",
        "benchmark": summary["benchmark"],
        "hardware_scope": "NVIDIA RTX A4000, Windows, 50,000 households",
        "implementation": {
            "cache_contract": manifest["contract"],
            "cache_bytes": manifest["cache"]["bytes"],
            "cache_sha256": manifest["cache"]["sha256"],
            "artifact_sha256": recorded_artifact_digest,
            "source_sha256": {
                item["name"]: item["sha256"] for item in manifest["sources"]
            },
            "runtime_validation_is_charged_to_candidate": True,
            "upstream_compatibility_bridge": (
                "narrow Sharrow 2.13 NumPy-memmap ownership truth-value fix"
            ),
        },
        "matched_full_model": {
            "summary": str(summary_path.relative_to(ROOT)),
            "runs": runs,
            "median_phase55_seconds": summary["median_baseline_all_model_seconds"],
            "median_phase56_seconds_including_validation": summary[
                "median_candidate_all_model_seconds"
            ],
            "median_seconds_saved": summary["median_seconds_saved"],
            "median_speedup": summary["median_speedup"],
            "median_reduction_percent": summary["median_reduction_percent"],
            "median_initialization_seconds": median(
                run["initialize_landuse_seconds"] for run in runs
            ),
            "median_cache_validation_seconds": median(
                run["cache_validation_seconds"] for run in runs
            ),
            "component_comparison": summary["component_comparison"],
        },
        "historical_regular_activitysim_context": {
            "source": str(cpu_summary_path.relative_to(ROOT)),
            "comparison_design": (
                "separately measured three-run median on the same machine and workload; "
                "not interleaved with Phase 56"
            ),
            "regular_activitysim_median_seconds": cpu_summary[
                "median_baseline_all_model_seconds"
            ],
            "phase56_median_seconds": summary["median_candidate_all_model_seconds"],
            "cumulative_speedup": cpu_summary["median_baseline_all_model_seconds"]
            / summary["median_candidate_all_model_seconds"],
            "seconds_saved": cpu_summary["median_baseline_all_model_seconds"]
            - summary["median_candidate_all_model_seconds"],
            "reduction_percent": 100
            * (
                cpu_summary["median_baseline_all_model_seconds"]
                - summary["median_candidate_all_model_seconds"]
            )
            / cpu_summary["median_baseline_all_model_seconds"],
        },
        "acceptance_gates": gates,
        "all_acceptance_gates_pass": all(gates.values()),
        "claim_boundary": (
            "Phase 56 is a replicated whole-model lifecycle improvement over "
            "Phase 55 on this fixed public workload. It removes repeated Sharrow "
            "skim expansion with a verified persistent data image. This is a "
            "startup/data-plane gain layered on the GPU runtime, not a new claim "
            "that an individual CUDA kernel became faster."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["all_acceptance_gates_pass"]:
        raise SystemExit("Phase 56 qualification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
