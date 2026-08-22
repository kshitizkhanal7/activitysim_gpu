"""Verify exact modeled decisions and explicitly bounded diagnostic logsums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED = {"final_checkpoints.csv"}
DIAGNOSTIC_LIMITS = {
    "destination_logsum": 1e-4,
    "mode_choice_logsum": 1e-5,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(reference: Path, candidate: Path) -> dict:
    names = sorted(
        path.name for path in reference.glob("final_*.csv")
        if path.name not in EXCLUDED
    )
    missing = [name for name in names if not (candidate / name).exists()]
    if missing:
        raise RuntimeError(f"candidate is missing outputs: {missing}")
    byte_mismatches = [
        name for name in names
        if name != "final_trips.csv" and sha256(reference / name) != sha256(candidate / name)
    ]
    if byte_mismatches:
        raise RuntimeError(f"non-trip substantive outputs differ: {byte_mismatches}")

    left_text = pd.read_csv(reference / "final_trips.csv", dtype=str, keep_default_na=False)
    right_text = pd.read_csv(candidate / "final_trips.csv", dtype=str, keep_default_na=False)
    if list(left_text.columns) != list(right_text.columns) or left_text.shape != right_text.shape:
        raise RuntimeError("final_trips schema or row count differs")
    diagnostic_columns = [
        column for column in DIAGNOSTIC_LIMITS if column in left_text
    ]
    decision_columns = [
        column for column in left_text if column not in diagnostic_columns
    ]
    decision_mask = left_text[decision_columns].ne(right_text[decision_columns])
    decision_cells = int(decision_mask.to_numpy().sum())
    decision_rows = int(decision_mask.any(axis=1).sum())
    if decision_cells:
        raise RuntimeError(
            f"Phase 15 changed {decision_cells} decision cells across {decision_rows} trips"
        )

    diagnostics = {}
    for column in diagnostic_columns:
        left = pd.to_numeric(left_text[column], errors="coerce").to_numpy()
        right = pd.to_numeric(right_text[column], errors="coerce").to_numpy()
        both_nan = np.isnan(left) & np.isnan(right)
        one_nan = np.isnan(left) ^ np.isnan(right)
        if one_nan.any():
            raise RuntimeError(f"{column} missing-value pattern differs")
        delta = np.abs(left[~both_nan] - right[~both_nan])
        maximum = float(delta.max(initial=0.0))
        limit = DIAGNOSTIC_LIMITS[column]
        if maximum > limit:
            raise RuntimeError(f"{column} max abs {maximum} exceeds {limit}")
        diagnostics[column] = {
            "text_cells_different": int(
                left_text[column].ne(right_text[column]).sum()
            ),
            "max_abs": maximum,
            "mean_abs": float(delta.mean()) if delta.size else 0.0,
            "gate": limit,
        }
    destination = diagnostics.get("destination_logsum", {})
    mode_choice = diagnostics.get("mode_choice_logsum", {})
    return {
        "decision_columns_exact": True,
        "decision_cells_different": decision_cells,
        "decision_rows_different": decision_rows,
        "byte_identical_non_trip_outputs": [
            name for name in names if name != "final_trips.csv"
        ],
        "diagnostic_columns": diagnostics,
        "diagnostic_column": "destination_logsum",
        "diagnostic_text_cells_different": destination.get("text_cells_different", 0),
        "diagnostic_max_abs": destination.get("max_abs", 0.0),
        "diagnostic_mean_abs": destination.get("mean_abs", 0.0),
        "diagnostic_gate": destination.get(
            "gate", DIAGNOSTIC_LIMITS["destination_logsum"]
        ),
        "mode_choice_logsum_max_abs": mode_choice.get("max_abs", 0.0),
        "mode_choice_logsum_gate": mode_choice.get(
            "gate", DIAGNOSTIC_LIMITS["mode_choice_logsum"]
        ),
        "success": True,
        "claim_boundary": (
            "all modeled decisions and non-trip substantive files are exact; "
            "declared logsum diagnostics are bounded, not byte-identical"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.reference, args.candidate)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
