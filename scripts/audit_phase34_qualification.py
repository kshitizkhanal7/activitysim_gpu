"""Fail-closed audit and integrity manifest for a Phase 34 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_phase22_integrated_scheduling.py",
    "scripts/run_phase32_full_model_ab.ps1",
    "scripts/run_phase34_location_choice_ab.ps1",
    "scripts/verify_phase15_outputs.py",
    "src/choiceforge/activitysim_mode_choice.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    checks = {
        "phase_is_34": summary.get("phase") == 34,
        "three_matched_pairs": summary.get("repetitions") == 3
        and len(summary.get("runs", [])) == 3,
        "candidate_won_every_pair": summary.get("candidate_won_every_pair") is True,
        "every_pair_exact": summary.get("every_pair_exact") is True,
        "median_candidate_faster": (
            summary.get("median_candidate_all_model_seconds", float("inf"))
            < summary.get("median_baseline_all_model_seconds", 0)
        ),
    }
    evidence = {relative(summary_path): digest(summary_path)}
    run_audits = []
    for run in summary.get("runs", []):
        proof_path = Path(run["candidate_report"]).resolve()
        exact_path = Path(run["exact_output_verification"]).resolve()
        proof = json.loads(proof_path.read_text(encoding="utf-8-sig"))
        exact = json.loads(exact_path.read_text(encoding="utf-8-sig"))
        proof_gates = proof.get("proof_gates", {})
        run_checks = {
            "candidate_faster": run["candidate_all_model_seconds"]
            < run["baseline_all_model_seconds"],
            "all_proof_gates_true": bool(proof_gates)
            and all(value is True for value in proof_gates.values()),
            "no_fallback": proof.get("fallback_calls") == 0,
            "location_calls_exact": proof.get("phase34_location_cuda_calls") == 13,
            "location_rows_exact": proof.get("phase34_location_rows") == 2_932_524,
            "atwork_mode_call_exact": proof.get("phase34_atwork_mode_cuda_calls") == 1,
            "atwork_mode_rows_exact": proof.get("phase34_atwork_mode_rows") == 15_100,
            "exact_verifier_success": exact.get("success") is True,
            "zero_decision_cells_changed": exact.get("decision_cells_different") == 0,
            "zero_decision_rows_changed": exact.get("decision_rows_different") == 0,
        }
        run_audits.append({"trial": run["trial"], "checks": run_checks})
        checks[f"trial_{run['trial']}_passed"] = all(run_checks.values())
        evidence[relative(proof_path)] = digest(proof_path)
        evidence[relative(exact_path)] = digest(exact_path)

    source_hashes = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        source_hashes[name] = digest(path)

    payload = {
        "phase": 34,
        "success": all(checks.values()),
        "checks": checks,
        "runs": run_audits,
        "evidence_sha256": evidence,
        "implementation_sha256": source_hashes,
        "claim_boundary": (
            "integrity audit of the three-pair public 50,000-household Phase 34 "
            "qualification; hashes detect later artifact or implementation changes"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
