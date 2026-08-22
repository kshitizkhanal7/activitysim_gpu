"""Verify Phase 15 decision replication and bounded diagnostic logsum drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED = {"final_checkpoints.csv"}
DIAGNOSTIC_COLUMN = "destination_logsum"
MAX_DIAGNOSTIC_ABS = 1e-4


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
    decision_columns = [column for column in left_text if column != DIAGNOSTIC_COLUMN]
    decision_mask = left_text[decision_columns].ne(right_text[decision_columns])
    decision_cells = int(decision_mask.to_numpy().sum())
    decision_rows = int(decision_mask.any(axis=1).sum())
    if decision_cells:
        raise RuntimeError(
            f"Phase 15 changed {decision_cells} decision cells across {decision_rows} trips"
        )

    left = pd.to_numeric(left_text[DIAGNOSTIC_COLUMN], errors="coerce").to_numpy()
    right = pd.to_numeric(right_text[DIAGNOSTIC_COLUMN], errors="coerce").to_numpy()
    both_nan = np.isnan(left) & np.isnan(right)
    one_nan = np.isnan(left) ^ np.isnan(right)
    if one_nan.any():
        raise RuntimeError("destination_logsum missing-value pattern differs")
    delta = np.abs(left[~both_nan] - right[~both_nan])
    maximum = float(delta.max(initial=0.0))
    if maximum > MAX_DIAGNOSTIC_ABS:
        raise RuntimeError(
            f"destination_logsum max abs {maximum} exceeds {MAX_DIAGNOSTIC_ABS}"
        )
    text_differences = int(
        left_text[DIAGNOSTIC_COLUMN].ne(right_text[DIAGNOSTIC_COLUMN]).sum()
    )
    return {
        "decision_columns_exact": True,
        "decision_cells_different": decision_cells,
        "decision_rows_different": decision_rows,
        "byte_identical_non_trip_outputs": [
            name for name in names if name != "final_trips.csv"
        ],
        "diagnostic_column": DIAGNOSTIC_COLUMN,
        "diagnostic_text_cells_different": text_differences,
        "diagnostic_max_abs": maximum,
        "diagnostic_mean_abs": float(delta.mean()) if delta.size else 0.0,
        "diagnostic_gate": MAX_DIAGNOSTIC_ABS,
        "success": True,
        "claim_boundary": (
            "all modeled decisions and non-trip substantive files are exact; "
            "destination_logsum is diagnostic and bounded, not byte-identical"
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
