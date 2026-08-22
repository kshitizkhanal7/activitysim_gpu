"""Capture real trip-destination nested-logit reduction inputs.

The model runs normally. This observer records the 21 evaluated mode utilities,
the purpose-specific numeric nest, and ActivitySim's resulting logsum for every
Phase 7 destination-logsum call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def _wanted(label: object) -> bool:
    text = str(label)
    return "trip_destination" in text and "compute_logsums_tripnum_batched" in text


def _jsonable(value):
    """Convert ActivitySim/Pydantic nest objects to plain JSON values."""
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config", action="append", type=Path, default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.capture.mkdir(parents=True, exist_ok=True)

    from activitysim.core import simulate

    original_eval_utilities = simulate.eval_utilities
    original_eval_nl_logsums = simulate.eval_nl_logsums
    pending: dict[str, list[dict]] = {}
    captures: list[dict] = []

    def capture_utilities(*pos, **kwargs):
        started = time.perf_counter()
        result = original_eval_utilities(*pos, **kwargs)
        label = kwargs.get("trace_label")
        if label is None and len(pos) > 4:
            label = pos[4]
        if _wanted(label):
            pending.setdefault(str(label), []).append(
                {
                    "utilities": np.ascontiguousarray(result.to_numpy(), dtype=np.float64),
                    "alternatives": [str(column) for column in result.columns],
                    "utility_eval_seconds": time.perf_counter() - started,
                }
            )
        return result

    def capture_logsums(
        state,
        choosers,
        spec,
        nest_spec,
        locals_d,
        trace_label=None,
        **kwargs,
    ):
        result = original_eval_nl_logsums(
            state,
            choosers,
            spec,
            nest_spec,
            locals_d,
            trace_label=trace_label,
            **kwargs,
        )
        eval_label = f"{trace_label}.eval_nl_logsums"
        if _wanted(trace_label):
            queue = pending.get(eval_label, [])
            if not queue:
                raise RuntimeError(f"missing raw utilities for {eval_label}")
            record = queue.pop(0)
            record.update(
                {
                    "trace_label": str(trace_label),
                    "nest_spec": nest_spec,
                    "logsums": np.asarray(result, dtype=np.float64),
                }
            )
            captures.append(record)
        return result

    simulate.eval_utilities = capture_utilities
    simulate.eval_nl_logsums = capture_logsums

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
        simulate.eval_utilities = original_eval_utilities
        simulate.eval_nl_logsums = original_eval_nl_logsums

    if any(pending.values()):
        raise RuntimeError("unmatched nested-logit utility captures remain")

    manifest = {"format_version": 1, "component": "trip_destination_logsums", "batches": []}
    for number, record in enumerate(captures):
        filename = f"batch_{number:03d}.npz"
        np.savez_compressed(
            args.capture / filename,
            utilities=record["utilities"],
            alternatives=np.asarray(record["alternatives"]),
            logsums=record["logsums"],
        )
        manifest["batches"].append(
            {
                "file": filename,
                "trace_label": record["trace_label"],
                "rows": int(record["utilities"].shape[0]),
                "alternatives": int(record["utilities"].shape[1]),
                "utility_eval_seconds": record["utility_eval_seconds"],
                "nest_spec": _jsonable(record["nest_spec"]),
            }
        )
    (args.capture / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"captured {len(captures)} real nested-logit batches")
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
