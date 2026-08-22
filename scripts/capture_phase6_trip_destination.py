"""Capture deterministic trip-destination simulation batches from ActivitySim.

The capture observes the real ``interaction_sample_simulate`` boundary after
sampling and mode-choice logsums. It records ActivitySim's evaluated terms,
ragged sampled destinations, random draws, selected positions, and outputs.
Nothing in the model is replaced during capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def _is_destination_simulation(label: object) -> bool:
    text = str(label)
    return text.startswith("trip_destination") and "trip_dest_simulate" in text and "compute_logsums" not in text


def _expression_values(spec, df, locals_d):
    """Evaluate each utility term with the same expression contract as ActivitySim."""
    from activitysim.core import simulate
    from activitysim.core.fast_eval import fast_eval

    local = dict(locals_d or {})
    local["df"] = df
    if isinstance(spec.index, pd.MultiIndex):
        expressions = spec.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
        labels = spec.index.get_level_values(simulate.SPEC_LABEL_NAME)
    else:
        expressions = spec.index
        labels = spec.index

    columns = []
    kept_expressions = []
    kept_labels = []
    kept_coefficients = []

    def as_array(value):
        if np.isscalar(value):
            return np.full(len(df), value, dtype=np.float64)
        return np.asarray(value)

    for expression, label, coefficient in zip(expressions, labels, spec.iloc[:, 0]):
        expression = str(expression)
        if expression.startswith("_") and "@" in expression:
            target, rhs = expression.split("@", 1)
            local[target] = pd.Series(as_array(eval(rhs, globals(), local)), index=df.index)
            continue
        if expression.startswith("@"):
            value = eval(expression[1:], globals(), local)
        else:
            value = fast_eval(df, expression, resolvers=[local])
        columns.append(as_array(value).astype(np.float32, copy=False))
        kept_expressions.append(expression)
        kept_labels.append(str(label))
        kept_coefficients.append(float(coefficient))

    return (
        np.ascontiguousarray(np.column_stack(columns), dtype=np.float32),
        kept_expressions,
        kept_labels,
        np.asarray(kept_coefficients, dtype=np.float32),
    )


def _stable_logsums(utilities: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.empty(offsets.size - 1, dtype=np.float64)
    for chooser, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row = utilities[begin:end]
        row_max = row.max()
        result[chooser] = row_max + np.log(np.exp(row - row_max).sum())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config", action="append", type=Path, default=[])
    args = parser.parse_args()

    args.capture.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    from activitysim.core import interaction_simulate, logit

    original_eval = interaction_simulate.eval_interaction_utilities
    original_make_choices = logit.make_choices
    pending: dict[str, list[dict]] = {}
    batches: list[dict] = []

    def capture_eval(state, spec, df, locals_d, trace_label, trace_rows, *pos, **kwargs):
        started = time.perf_counter()
        result = original_eval(
            state, spec, df, locals_d, trace_label, trace_rows, *pos, **kwargs
        )
        elapsed = time.perf_counter() - started
        if _is_destination_simulation(trace_label):
            terms, expressions, labels, coefficients = _expression_values(spec, df, locals_d)
            chooser_ids = np.asarray(df.index)
            starts = np.flatnonzero(np.r_[True, chooser_ids[1:] != chooser_ids[:-1]])
            offsets = np.ascontiguousarray(np.r_[starts, len(df)], dtype=np.int64)
            record = {
                "trace_label": str(trace_label),
                "terms": terms,
                "coefficients": coefficients,
                "expressions": expressions,
                "labels": labels,
                "utilities": np.asarray(result[0].utility, dtype=np.float64),
                "chooser_ids": chooser_ids,
                "offsets": offsets,
                "alternative_ids": np.asarray(df["dest_taz"], dtype=np.int32),
                "eval_seconds": elapsed,
            }
            pending.setdefault(str(trace_label), []).append(record)
        return result

    def capture_choices(state, probabilities, trace_label=None, trace_choosers=None):
        result = original_make_choices(
            state,
            probabilities,
            trace_label=trace_label,
            trace_choosers=trace_choosers,
        )
        if _is_destination_simulation(trace_label):
            label = str(trace_label)
            candidates = [key for key, queue in pending.items() if queue and label.startswith(key)]
            if not candidates:
                raise RuntimeError(f"no captured destination utility batch matches {label}")
            key = max(candidates, key=len)
            record = pending[key].pop(0)
            positions = np.asarray(result[0], dtype=np.int32)
            row_positions = record["offsets"][:-1] + positions
            record.update(
                {
                    "positions": positions,
                    "draws": np.asarray(result[1], dtype=np.float64),
                    "probabilities": np.asarray(probabilities, dtype=np.float64),
                    "choices": record["alternative_ids"][row_positions],
                    "logsums": _stable_logsums(record["utilities"], record["offsets"]),
                }
            )
            batches.append(record)
        return result

    interaction_simulate.eval_interaction_utilities = capture_eval
    logit.make_choices = capture_choices

    from activitysim.cli import main as activitysim_main

    cli = ["activitysim", "run"]
    for config in args.config:
        cli.extend(["-c", str(config.resolve())])
    cli.extend(
        [
            "-c",
            str((args.project / "configs").resolve()),
            "-d",
            str((args.project / "data").resolve()),
            "-o",
            str(args.output.resolve()),
        ]
    )
    old_argv = sys.argv
    exit_code = 0
    try:
        sys.argv = cli
        try:
            exit_code = activitysim_main.main()
        except SystemExit as exc:
            exit_code = exc.code or 0
    finally:
        sys.argv = old_argv
        interaction_simulate.eval_interaction_utilities = original_eval
        logit.make_choices = original_make_choices

    if pending and any(pending.values()):
        raise RuntimeError("unmatched destination utility captures remain")

    manifest = {"format_version": 1, "component": "trip_destination", "batches": []}
    for number, batch in enumerate(batches):
        filename = f"batch_{number:03d}.npz"
        np.savez_compressed(
            args.capture / filename,
            terms=batch["terms"],
            coefficients=batch["coefficients"],
            utilities=batch["utilities"],
            chooser_ids=batch["chooser_ids"],
            offsets=batch["offsets"],
            alternative_ids=batch["alternative_ids"],
            positions=batch["positions"],
            choices=batch["choices"],
            draws=batch["draws"],
            probabilities=batch["probabilities"],
            logsums=batch["logsums"],
        )
        counts = np.diff(batch["offsets"])
        manifest["batches"].append(
            {
                "file": filename,
                "trace_label": batch["trace_label"],
                "interaction_rows": int(batch["terms"].shape[0]),
                "terms": int(batch["terms"].shape[1]),
                "choosers": int(batch["positions"].size),
                "min_alternatives": int(counts.min()),
                "max_alternatives": int(counts.max()),
                "expanded_megabytes": batch["terms"].nbytes / 1_000_000,
                "eval_seconds": batch["eval_seconds"],
                "expressions": batch["expressions"],
                "labels": batch["labels"],
            }
        )
    (args.capture / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"captured {len(batches)} trip-destination batches in {args.capture}")
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
