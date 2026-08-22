"""Make the Phase 15 50k non-promotion result reproducible and explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from verify_phase15_outputs import verify


def timing(output: Path, log: Path) -> dict:
    frame = pd.read_csv(output / "timing_log.csv")
    destination = frame.loc[frame["model_name"] == "trip_destination", "seconds"]
    if len(destination) != 1:
        raise RuntimeError(f"expected one trip_destination timing in {output}")
    text = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Time to execute all models\s*:\s*([0-9.]+) seconds", text)
    if not match:
        raise RuntimeError(f"missing all-model timing in {log}")
    return {
        "all_models_seconds": float(match.group(1)),
        "trip_destination_seconds": float(destination.iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = timing(args.baseline_output, args.baseline_log)
    candidate = timing(args.candidate_output, args.candidate_log)
    candidate_text = args.candidate_log.read_text(encoding="utf-8", errors="replace")
    correctness = verify(args.baseline_output, args.candidate_output)
    result = {
        "phase": 15,
        "benchmark": "public Prototype MTC Extended, 50,000 households, 1,454 zones",
        "design": "one fresh-process diagnostic pair; sufficient to reject, not to claim superiority",
        "policy": "unbounded strict candidate on all destination utility batches",
        "baseline": baseline,
        "candidate": candidate,
        "all_models_speedup": baseline["all_models_seconds"] / candidate["all_models_seconds"],
        "trip_destination_speedup": (
            baseline["trip_destination_seconds"] / candidate["trip_destination_seconds"]
        ),
        "candidate_policy_fallbacks": len(
            re.findall(r"strict CUDA candidate policy fallback", candidate_text)
        ),
        "candidate_batches": len(
            re.findall(r"ChoiceForge strict candidate rows=", candidate_text)
        ),
        "correctness": correctness,
        "promotion_gate_passed": False,
        "decision": (
            "reject unbounded Phase 15 at 50k; retain Phase 11 production path "
            "until skim gathers and compact inputs are device-native"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
