"""Benchmark the exact Phase 48 host round trip against Phase 49 residency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from choiceforge.cuda_backend import _cupy


# (sampled alternatives, selected published location outputs).  These are the
# complete 19-call, 50k-household public Prototype MTC extended dimensions.
PUBLIC_CALLS = (
    (44_292, 8_093), (73_712, 5_654), (567_911, 23_509),
    (256_620, 10_099), (379_658, 14_918), (511_991, 19_025),
    (710_813, 26_496), (14_556, 0), (15_958, 0), (25_711, 0),
    (11_392, 0), (8_942, 0), (442_368, 0), (267_686, 0),
    (408_285, 0), (188_566, 0), (122_209, 0), (335_038, 0),
    (310_968, 0),
)


def run_boundary(cp, repetitions: int) -> dict:
    maximum = max(rows for rows, _ in PUBLIC_CALLS)
    source = cp.linspace(-20.0, 20.0, maximum, dtype=cp.float64)
    selectors = {
        (rows, selected): cp.asarray(
            np.linspace(0, rows - 1, selected, dtype=np.int64)
        )
        for rows, selected in PUBLIC_CALLS if selected
    }

    def baseline():
        outputs = []
        for rows, _ in PUBLIC_CALLS:
            host64 = cp.asnumpy(source[:rows])
            outputs.append(cp.asarray(host64.astype(np.float32)))
        cp.cuda.Stream.null.synchronize()
        return outputs

    def candidate():
        outputs = []
        selected_outputs = []
        for rows, selected in PUBLIC_CALLS:
            view = source[:rows]
            outputs.append(view.astype(cp.float32, copy=True))
            if selected:
                selected_outputs.append(cp.asnumpy(view[selectors[(rows, selected)]]))
        cp.cuda.Stream.null.synchronize()
        return outputs, selected_outputs

    baseline_outputs = baseline()
    candidate_outputs, selected_outputs = candidate()
    mismatches = sum(
        int(cp.count_nonzero(left.view(cp.uint32) != right.view(cp.uint32)).get())
        for left, right in zip(baseline_outputs, candidate_outputs)
    )
    selected_rows = sum(len(values) for values in selected_outputs)

    baseline_seconds = []
    candidate_seconds = []
    for _ in range(repetitions):
        started = time.perf_counter()
        baseline()
        baseline_seconds.append(time.perf_counter() - started)
        started = time.perf_counter()
        candidate()
        candidate_seconds.append(time.perf_counter() - started)
    baseline_median = statistics.median(baseline_seconds)
    candidate_median = statistics.median(candidate_seconds)
    source_bytes = sum(rows * 8 for rows, _ in PUBLIC_CALLS)
    upload_bytes = sum(rows * 4 for rows, _ in PUBLIC_CALLS)
    selected_bytes = selected_rows * 8
    return {
        "phase": 49,
        "benchmark": "exact inter-stage logsum handoff on all public 50k call shapes",
        "repetitions": repetitions,
        "calls_per_repetition": len(PUBLIC_CALLS),
        "alternative_rows": sum(rows for rows, _ in PUBLIC_CALLS),
        "selected_output_rows": selected_rows,
        "baseline_device_to_host_bytes": source_bytes,
        "baseline_host_to_device_bytes": upload_bytes,
        "candidate_selected_device_to_host_bytes": selected_bytes,
        "round_trip_bytes_avoided": source_bytes + upload_bytes - selected_bytes,
        "float32_bit_mismatches": mismatches,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "median_baseline_seconds": baseline_median,
        "median_candidate_seconds": candidate_median,
        "median_speedup": baseline_median / candidate_median,
        "median_reduction_percent": 100.0 * (baseline_median - candidate_median) / baseline_median,
        "exact_gate_passed": mismatches == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    result = run_boundary(_cupy(), args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["exact_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
