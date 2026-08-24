"""Capture deterministic mandatory-tour-scheduling replay data from ActivitySim.

This script runs the real prototype_mtc component and records the exact numeric
expression values at ActivitySim's interaction-simulation boundary.  Random
draws and selected positions are captured from ActivitySim itself.  The output
is intentionally a replay artifact, not a replacement model.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def _expression_values(spec, df, locals_d, *, retain_terms=True):
    """Evaluate the spec once, retaining terms instead of immediately summing."""
    from activitysim.core.fast_eval import fast_eval
    from activitysim.core import simulate

    local = dict(locals_d or {})
    local["df"] = df
    if isinstance(spec.index, pd.MultiIndex):
        exprs = spec.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
        labels = spec.index.get_level_values(simulate.SPEC_LABEL_NAME)
    else:
        exprs = spec.index
        labels = spec.index

    columns = []
    kept_exprs = []
    kept_labels = []
    kept_coefficients = []

    def as_array(value):
        if np.isscalar(value):
            return np.full(len(df), value, dtype=np.float64)
        return np.asarray(value)

    for expr, label, coefficient in zip(exprs, labels, spec.iloc[:, 0]):
        if expr.startswith("_"):
            target, rhs = expr.split("@", 1)
            local[target] = pd.Series(as_array(eval(rhs, globals(), local)), index=df.index)
            continue
        stateful = "tt." in expr or "_adjacent_window" in expr or expr.startswith("@")
        if retain_terms or stateful:
            if expr.startswith("@"):
                value = eval(expr[1:], globals(), local)
            else:
                value = fast_eval(df, expr, resolvers=[local])
            values = as_array(value).astype(np.float32, copy=False)
        else:
            values = None
        if retain_terms:
            columns.append(values)
        kept_exprs.append(str(expr))
        kept_labels.append(str(label))
        kept_coefficients.append(float(coefficient))
        # The compact packer needs only stateful row primitives. Pure
        # expressions remain syntax and are evaluated later by generated code.
        if not retain_terms:
            columns.append(values if stateful else None)

    return (
        (
            np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
            if retain_terms
            else None
        ),
        kept_exprs,
        kept_labels,
        np.asarray(kept_coefficients, dtype=np.float32),
        columns,
    )


def _compact_inputs(df, expressions, expression_columns):
    """Build a compact ABI instead of repeating every evaluated term.

    Pure arithmetic/Boolean expressions remain as syntax for the Phase 3 CUDA
    compiler. Stateful timetable expressions are captured as row primitives;
    reproducing the timetable itself is outside this kernel boundary.
    """
    lowered = []
    stateful_columns = []
    for expr, values in zip(expressions, expression_columns):
        if "tt." in expr or "_adjacent_window" in expr or expr.startswith("@"):
            if values is None:
                raise RuntimeError(f"stateful expression was not captured: {expr}")
            name = f"stateful_{len(stateful_columns)}"
            lowered.append(name)
            stateful_columns.append(np.asarray(values, dtype=np.float32))
        else:
            lowered.append(
                expr.replace(
                    "mandatory_tour_frequency == 'work_and_school'",
                    "mandatory_tour_frequency_work_and_school",
                )
            )

    names = set()
    for expr in lowered:
        names.update(
            node.id for node in ast.walk(ast.parse(expr, mode="eval"))
            if isinstance(node, ast.Name)
        )

    alternative_names = [x for x in ("start", "end", "duration") if x in names]
    row_names = ["mode_choice_logsum"] if "mode_choice_logsum" in names else []
    row_names.extend(f"stateful_{i}" for i in range(len(stateful_columns)))
    chooser_names = sorted(names - set(alternative_names) - set(row_names))

    chooser_ids = np.asarray(df.index)
    starts = np.flatnonzero(np.r_[True, chooser_ids[1:] != chooser_ids[:-1]])
    chooser_values = []
    for name in chooser_names:
        if name == "mandatory_tour_frequency_work_and_school":
            values = np.asarray(df["mandatory_tour_frequency"] == "work_and_school")
        else:
            values = np.asarray(df[name])
        chooser_values.append(values[starts].astype(np.float32, copy=False))

    alt_ids = np.asarray(df["tdd"], dtype=np.int16)
    n_alts = int(alt_ids.max()) + 1
    alternative_values = np.full((n_alts, len(alternative_names)), np.nan, dtype=np.float32)
    for alt_id in np.unique(alt_ids):
        first = int(np.flatnonzero(alt_ids == alt_id)[0])
        for column, name in enumerate(alternative_names):
            alternative_values[int(alt_id), column] = np.float32(df[name].iloc[first])

    rows = []
    if "mode_choice_logsum" in row_names:
        rows.append(np.asarray(df["mode_choice_logsum"], dtype=np.float32))
    rows.extend(stateful_columns)
    return {
        "expressions": lowered,
        "chooser_names": chooser_names,
        "row_names": row_names,
        "alternative_names": alternative_names,
        "chooser_values": np.ascontiguousarray(np.column_stack(chooser_values), dtype=np.float32),
        "row_values": np.ascontiguousarray(np.column_stack(rows), dtype=np.float32),
        "alternative_values": alternative_values,
        "alternative_ids": np.ascontiguousarray(alt_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="input-data directory (defaults to PROJECT/data)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config-overlay", type=Path)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="omit the expanded term, utility, and probability matrices",
    )
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="retain ActivitySim probability matrices for precision diagnostics",
    )
    parser.add_argument(
        "--only-next-model",
        action="store_true",
        help="when resuming, stop after the single model following the checkpoint",
    )
    args = parser.parse_args()

    args.capture.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    from activitysim.core import interaction_simulate, logit

    original_eval = interaction_simulate.eval_interaction_utilities
    original_make_choices = logit.make_choices
    pending: dict[str, list[dict]] = {}
    batches: list[dict] = []

    def capture_eval(state, spec, df, locals_d, trace_label, trace_rows, *pos, **kw):
        start = time.perf_counter()
        result = original_eval(
            state, spec, df, locals_d, trace_label, trace_rows, *pos, **kw
        )
        elapsed = time.perf_counter() - start
        if str(trace_label).startswith("mandatory_tour_scheduling") and "logsums" not in str(trace_label):
            terms, expressions, labels, coefficients, expression_columns = _expression_values(
                spec, df, locals_d, retain_terms=not args.compact_only
            )
            compact = _compact_inputs(df, expressions, expression_columns)
            chooser_ids = np.asarray(df.index)
            starts = np.flatnonzero(
                np.r_[True, chooser_ids[1:] != chooser_ids[:-1]]
            )
            record = {
                "trace_label": str(trace_label),
                "terms": None if args.compact_only else terms,
                "coefficients": coefficients,
                "expressions": expressions,
                "labels": labels,
                "utilities": (
                    None
                    if args.compact_only
                    else np.asarray(result[0].utility, dtype=np.float64)
                ),
                "chooser_ids": chooser_ids[starts],
                "offsets": np.r_[starts, chooser_ids.size].astype(np.int64),
                "eval_seconds": elapsed,
                "compact": compact,
            }
            pending.setdefault(str(trace_label), []).append(record)
        return result

    def capture_choices(state, probs, trace_label=None, trace_choosers=None):
        result = original_make_choices(
            state, probs, trace_label=trace_label, trace_choosers=trace_choosers
        )
        label = str(trace_label)
        if label.startswith("mandatory_tour_scheduling") and "logsums" not in label:
            # eval_interaction_utilities receives the parent trace label, whereas
            # make_choices receives that label extended by make_choices.
            candidates = [k for k, q in pending.items() if q and label.startswith(k)]
            if not candidates:
                raise RuntimeError(f"no captured utility batch matches {label}")
            key = max(candidates, key=len)
            record = pending[key].pop(0)
            record["positions"] = np.asarray(result[0], dtype=np.int32)
            record["draws"] = np.asarray(result[1], dtype=np.float64)
            record["probabilities"] = (
                np.asarray(probs)
                if (args.include_probabilities or not args.compact_only)
                else None
            )
            batches.append(record)
        return result

    interaction_simulate.eval_interaction_utilities = capture_eval
    logit.make_choices = capture_choices

    original_runner_call = None
    if args.only_next_model:
        if not args.resume:
            raise ValueError("--only-next-model requires --resume")
        from activitysim.core.workflow.runner import Runner

        original_runner_call = Runner.__call__

        def run_one_model(self, models, resume_after=None, memory_sidecar_process=None):
            if isinstance(models, list) and resume_after in models:
                checkpoint = models.index(resume_after)
                models = models[: checkpoint + 2]
            return original_runner_call(
                self,
                models,
                resume_after=resume_after,
                memory_sidecar_process=memory_sidecar_process,
            )

        Runner.__call__ = run_one_model

    from activitysim.cli import main as activitysim_main

    cli = ["activitysim", "run"]
    if args.config_overlay:
        cli.extend(["-c", str(args.config_overlay.resolve())])
    cli.extend(
        [
            "-c",
            str((args.project / "configs").resolve()),
            "-d",
            str((args.data or (args.project / "data")).resolve()),
            "-o",
            str(args.output.resolve()),
        ]
    )
    if args.resume:
        cli.extend(["-r", args.resume])
    old_argv = sys.argv
    exit_code = 0
    try:
        sys.argv = cli
        try:
            exit_code = activitysim_main.main()
        except SystemExit as exc:
            # The console entry point exits even after a successful run.  Keep
            # control long enough to serialize the in-memory capture.
            exit_code = exc.code or 0
    finally:
        sys.argv = old_argv
        interaction_simulate.eval_interaction_utilities = original_eval
        logit.make_choices = original_make_choices
        if original_runner_call is not None:
            Runner.__call__ = original_runner_call

    manifest = {
        "format_version": 3 if args.compact_only else 2,
        "compact_only": bool(args.compact_only),
        "batches": [],
    }
    for number, batch in enumerate(batches):
        stem = f"batch_{number:03d}"
        arrays = {
            "coefficients": batch["coefficients"],
            "chooser_ids": batch["chooser_ids"],
            "offsets": batch["offsets"],
            "positions": batch["positions"],
            "draws": batch["draws"],
            "chooser_values": batch["compact"]["chooser_values"],
            "row_values": batch["compact"]["row_values"],
            "alternative_values": batch["compact"]["alternative_values"],
            "alternative_ids": batch["compact"]["alternative_ids"],
        }
        if not args.compact_only:
            arrays.update(
                terms=batch["terms"],
                utilities=batch["utilities"],
            )
        if batch["probabilities"] is not None:
            arrays["probabilities"] = batch["probabilities"]
        np.savez_compressed(args.capture / f"{stem}.npz", **arrays)
        manifest["batches"].append(
            {
                "file": f"{stem}.npz",
                "trace_label": batch["trace_label"],
                "interaction_rows": int(batch["compact"]["row_values"].shape[0]),
                "terms": int(batch["coefficients"].size),
                "choosers": int(batch["positions"].size),
                "eval_seconds": batch["eval_seconds"],
                "expressions": batch["expressions"],
                "labels": batch["labels"],
                "compact_expressions": batch["compact"]["expressions"],
                "chooser_columns": batch["compact"]["chooser_names"],
                "row_columns": batch["compact"]["row_names"],
                "alternative_columns": batch["compact"]["alternative_names"],
            }
        )
    (args.capture / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"captured {len(batches)} scheduling batches in {args.capture}")
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
