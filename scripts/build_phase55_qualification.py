"""Build and validate the Phase 55 replication/acceptance artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results"
KERNELS = ROOT / "src/choiceforge/kernels"
OUTPUT = RESULTS / "phase55-p55final3-qualification.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    atlas_path = KERNELS / "phase55_public_destination_plans.json"
    cubin_path = KERNELS / "phase55_public_destination_sm86.cubin"
    cubin_manifest_path = KERNELS / "phase55_public_destination_sm86.json"
    atlas = load(atlas_path)
    atlas_digest = atlas.pop("artifact_sha256")
    computed_atlas_digest = hashlib.sha256(
        json.dumps(atlas, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cubin_manifest = load(cubin_manifest_path)
    cubin_digest = hashlib.sha256(cubin_path.read_bytes()).hexdigest()

    shard_runs = []
    for trial in range(1, 4):
        baseline_seconds = 0.0
        candidate_seconds = 0.0
        shard_gates = []
        for shard in ("pre", "post"):
            baseline_path = RESULTS / f"phase54-p54final-{shard}-gpu-{trial}.json"
            candidate_path = RESULTS / f"phase55-p55final2-{shard}-gpu-{trial}.json"
            baseline = load(baseline_path)
            candidate = load(candidate_path)
            baseline_seconds += baseline["phase54_device_owned_destination_packet"][
                "total_seconds"
            ]
            candidate_seconds += candidate[
                "phase55_device_entity_execution_runtime"
            ]["total_seconds"]
            shard_gates.append(all(candidate["proof_gates"].values()))
        shard_runs.append(
            {
                "trial": trial,
                "phase54_seconds": baseline_seconds,
                "phase55_seconds": candidate_seconds,
                "seconds_saved": baseline_seconds - candidate_seconds,
                "speedup": baseline_seconds / candidate_seconds,
                "reduction_percent": 100 * (baseline_seconds - candidate_seconds)
                / baseline_seconds,
                "all_proof_gates_pass": all(shard_gates),
            }
        )

    full_summary_path = RESULTS / "phase55-p55final3-summary.json"
    full_summary = load(full_summary_path)
    full_runs = []
    monolithic_services = []
    baseline_sampling = []
    candidate_sampling = []
    for run in full_summary["runs"]:
        trial = int(run["trial"])
        baseline_report_path = RESULTS / f"phase55-p55final3-base-{trial}.json"
        candidate_report_path = RESULTS / f"phase55-p55final3-gpu-{trial}.json"
        exact_path = RESULTS / f"phase55-p55final3-exact-{trial}.json"
        baseline_report = load(baseline_report_path)
        candidate_report = load(candidate_report_path)
        exact = load(exact_path)
        runtime = candidate_report["phase55_device_entity_execution_runtime"]
        monolithic_services.append(runtime["total_seconds"])
        baseline_sampling.append(
            sum(item["total_seconds"] for item in baseline_report["phase45_modelwide_sampling"])
        )
        candidate_sampling.append(
            sum(item["total_seconds"] for item in candidate_report["phase45_modelwide_sampling"])
        )
        full_runs.append(
            {
                "trial": trial,
                "phase54_all_model_seconds": run["baseline_all_model_seconds"],
                "phase55_all_model_seconds": run["candidate_all_model_seconds"],
                "all_model_seconds_saved": run["all_model_seconds_saved"],
                "phase54_five_destination_seconds": run["baseline_destination_seconds"],
                "phase55_five_destination_seconds": run["candidate_destination_seconds"],
                "destination_seconds_saved": run["destination_seconds_saved"],
                "phase55_service_seconds": runtime["total_seconds"],
                "aot_plan_hits": runtime["phase55_plan_atlas_hits"],
                "codegen_bypasses": runtime["phase55_codegen_bypasses"],
                "device_compactions": sum(
                    bool(item["phase55_device_compaction"])
                    for item in candidate_report["phase45_modelwide_sampling"]
                ),
                "all_proof_gates_pass": all(candidate_report["proof_gates"].values()),
                "exact_output_verification_pass": bool(exact["success"]),
            }
        )

    gates = {
        "atlas_digest_and_ten_plan_contract_valid": (
            atlas_digest == computed_atlas_digest and len(atlas["plans"]) == 10
        ),
        "sm86_cubin_digest_valid": cubin_digest == cubin_manifest["cubin_sha256"],
        "all_six_cold_shards_pass_proof_gates": all(
            run["all_proof_gates_pass"] for run in shard_runs
        ),
        "phase55_service_wins_all_three_shard_pairs": all(
            run["seconds_saved"] > 0 for run in shard_runs
        ),
        "all_three_full_runs_pass_every_proof_gate": all(
            run["all_proof_gates_pass"] for run in full_runs
        ),
        "all_three_full_runs_are_output_exact": all(
            run["exact_output_verification_pass"] for run in full_runs
        ),
        "all_three_full_runs_have_19_aot_hits_and_compactions": all(
            run["aot_plan_hits"] == 19
            and run["codegen_bypasses"] == 19
            and run["device_compactions"] == 19
            for run in full_runs
        ),
        "five_destination_components_win_all_three_pairs": all(
            run["destination_seconds_saved"] > 0 for run in full_runs
        ),
        "whole_model_wins_all_three_pairs": all(
            run["all_model_seconds_saved"] > 0 for run in full_runs
        ),
        "monolithic_destination_service_median_below_2_5_seconds": (
            median(monolithic_services) < 2.5
        ),
    }
    stretch = {
        "five_destination_component_median_below_10_seconds": (
            full_summary["median_candidate_destination_seconds"] < 10.0
        ),
        "full_model_median_below_130_seconds": (
            full_summary["median_candidate_all_model_seconds"] < 130.0
        ),
    }
    output = {
        "phase": 55,
        "contract": "choiceforge-phase55-replicated-qualification-v1",
        "status": "qualified_with_two_absolute_time_stretch_targets_open",
        "benchmark": full_summary["benchmark"],
        "hardware_scope": "NVIDIA RTX A4000, sm_86 cubin",
        "implementation": {
            "plan_atlas": str(atlas_path.relative_to(ROOT)),
            "plan_atlas_sha256": atlas_digest,
            "plan_count": len(atlas["plans"]),
            "cubin": str(cubin_path.relative_to(ROOT)),
            "cubin_sha256": cubin_digest,
            "source_sha256": cubin_manifest["source_sha256"],
        },
        "cold_sharded_service": {
            "runs": shard_runs,
            "median_phase54_seconds": median(r["phase54_seconds"] for r in shard_runs),
            "median_phase55_seconds": median(r["phase55_seconds"] for r in shard_runs),
            "median_speedup": median(r["phase54_seconds"] for r in shard_runs)
            / median(r["phase55_seconds"] for r in shard_runs),
        },
        "matched_full_model": {
            "summary": str(full_summary_path.relative_to(ROOT)),
            "runs": full_runs,
            "median_phase54_all_model_seconds": full_summary[
                "median_baseline_all_model_seconds"
            ],
            "median_phase55_all_model_seconds": full_summary[
                "median_candidate_all_model_seconds"
            ],
            "median_all_model_speedup": full_summary["median_speedup"],
            "median_all_model_reduction_percent": full_summary[
                "median_reduction_percent"
            ],
            "median_phase54_five_destination_seconds": full_summary[
                "median_baseline_destination_seconds"
            ],
            "median_phase55_five_destination_seconds": full_summary[
                "median_candidate_destination_seconds"
            ],
            "median_five_destination_speedup": full_summary[
                "median_destination_speedup"
            ],
            "median_phase55_service_seconds": median(monolithic_services),
            "median_phase54_sampling_seconds": median(baseline_sampling),
            "median_phase55_sampling_seconds": median(candidate_sampling),
        },
        "acceptance_gates": gates,
        "stretch_targets": stretch,
        "core_qualification_pass": all(gates.values()),
        "all_stretch_targets_pass": all(stretch.values()),
        "claim_boundary": (
            "Phase 55 is a replicated speedup over the already GPU-accelerated "
            "Phase 54 runtime on this RTX A4000 public 50,000-household workload. "
            "It does not yet satisfy the two remaining absolute-time stretch targets."
        ),
    }
    if not output["core_qualification_pass"]:
        raise SystemExit("Phase 55 core qualification failed")
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
