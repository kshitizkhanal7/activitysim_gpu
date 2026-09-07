"""Consolidate Phase 57 live-computation evidence without hiding missed targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    summary = load(RESULTS / "phase57-p57formal-summary.json")
    micro = load(RESULTS / "phase57-live-pairs-microbenchmark.json")
    runs = []
    for timing in summary["runs"]:
        trial = timing["trial"]
        base = load(RESULTS / f"phase57-p57formal-base-{trial}.json")
        candidate = load(RESULTS / f"phase57-p57formal-gpu-{trial}.json")
        exact = load(RESULTS / f"phase57-p57formal-exact-{trial}.json")
        runs.append({
            "trial": trial,
            "baseline_seconds": timing["baseline_all_model_seconds"],
            "candidate_seconds": timing["candidate_all_model_seconds"],
            "baseline_process_wall_seconds": timing["baseline_wall_seconds"],
            "candidate_process_wall_seconds": timing["candidate_wall_seconds"],
            "seconds_saved": timing["all_model_seconds_saved"],
            "baseline_mandatory_seconds": timing["baseline_mandatory_seconds"],
            "candidate_mandatory_seconds": timing["candidate_mandatory_seconds"],
            "baseline_proof_pass": all(base["proof_gates"].values()),
            "candidate_proof_pass": all(candidate["proof_gates"].values()),
            "exact_decisions": exact["success"] and exact["decision_cells_different"] == 0,
            "live_compaction": candidate["phase57_live_scheduling"],
            "native_abi_programs": candidate["native_abi_programs"],
            "timed_model_steps": len(candidate["model_timing_seconds"]),
        })
    gates = {
        "three_complete_matched_pairs": len(runs) == 3,
        "all_baseline_and_candidate_proof_gates_pass": all(
            r["baseline_proof_pass"] and r["candidate_proof_pass"] for r in runs),
        "all_published_decisions_exact": all(r["exact_decisions"] for r in runs),
        "all_six_logsums_recomputed_in_every_run": all(r["native_abi_programs"] == 6 for r in runs),
        "all_34_steps_and_six_live_calls_in_every_run": all(
            r["timed_model_steps"] == 34 and r["live_compaction"]["calls"] == 6
            and r["live_compaction"]["representative_rows"] == 1_210_124 for r in runs),
        "all_15242743_host_rows_avoided": all(
            r["live_compaction"]["full_interaction_rows_avoided"] == 15_242_743 for r in runs),
        "no_new_result_replay": all(not r["live_compaction"]["result_replay_enabled"] for r in runs),
        "whole_model_wins_every_pair": all(r["seconds_saved"] > 0 for r in runs),
        "mandatory_scheduling_wins_every_pair": all(
            r["candidate_mandatory_seconds"] < r["baseline_mandatory_seconds"] for r in runs),
        "compiled_cpu_gpu_reduction_outputs_exact": micro["all_outputs_exact"],
        "gpu_beats_every_tested_cpu_configuration_in_every_micro_trial": all(
            trial["seconds"]["gpu"] < min(value for key, value in trial["seconds"].items()
                                           if key.startswith("cpu_"))
            for trial in micro["runs"]),
        "microbenchmark_matches_live_cardinalities": (
            micro["batches"] == 6 and micro["chooser_rows"] == 81_983
            and micro["feasible_rows"] == 15_242_743
            and micro["representative_rows"] == 1_210_124),
    }
    target_pass = summary["median_candidate_all_model_seconds"] < 105
    source_paths = [
        "src/choiceforge/phase57_live_scheduling.py",
        "scripts/run_phase22_integrated_scheduling.py",
        "scripts/run_phase32_full_model_ab.ps1",
        "scripts/benchmark_phase57_live_pairs.py",
        "tests/test_phase57_live_scheduling.py",
    ]
    evidence_paths = [RESULTS / "phase57-p57formal-summary.json",
                      RESULTS / "phase57-live-pairs-microbenchmark.json"]
    for run in runs:
        evidence_paths.extend(RESULTS / f"phase57-p57formal-{kind}-{run['trial']}.json"
                              for kind in ("base", "gpu", "exact"))
    report = {
        "phase": 57,
        "contract": "choiceforge-phase57-live-scheduling-qualification-v1",
        "status": ("target_met" if target_pass else "replicated_improvement_target_not_met")
            if all(gates.values()) else "not_qualified",
        "correctness_and_replication_gates": gates,
        "all_correctness_and_replication_gates_pass": all(gates.values()),
        "sub_105_second_target_met": target_pass,
        "gap_to_105_seconds": max(0, summary["median_candidate_all_model_seconds"]-105),
        "matched_full_model": {
            "runs": runs,
            "median_baseline_seconds": summary["median_baseline_all_model_seconds"],
            "median_candidate_seconds": summary["median_candidate_all_model_seconds"],
            "speedup": summary["median_speedup"],
            "reduction_percent": summary["median_reduction_percent"],
            "median_baseline_process_wall_seconds": median(r["baseline_process_wall_seconds"] for r in runs),
            "median_candidate_process_wall_seconds": median(r["candidate_process_wall_seconds"] for r in runs),
            "components": summary["component_comparison"],
        },
        "compiled_cpu_comparison": micro,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        "evidence_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_paths},
        "limitations": [
            "Fixed public 50000-household workload with inherited scheduling and sparse boundary reference artifacts.",
            "Three pairs measure repeatability; they do not prove universal speed or broad statistical significance.",
            "Model-step totals plus validation/prewarm are distinct from process wall time.",
            "Earlier result-capsule and final-output replay experiments were rejected and removed from the release candidate.",
            "The 105-second target is reported separately and is never waived by a positive relative speedup.",
        ],
    }
    output = RESULTS / "phase57-p57formal-qualification.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status", "all_correctness_and_replication_gates_pass", "sub_105_second_target_met", "gap_to_105_seconds")}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
