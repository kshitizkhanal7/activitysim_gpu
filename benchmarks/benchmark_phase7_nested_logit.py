"""Benchmark the captured real MTC nested-logit reduction boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd

from choiceforge.nested_logit import mtc21_nested_logsums_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture", type=Path,
        default=Path("benchmark-results/phase7-nested-logsum-capture"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark-results/phase7-nested-logsum-summary.json"),
    )
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--phase", default="7B")
    parser.add_argument("--date", default="2026-08-11")
    args = parser.parse_args()

    from activitysim.core.simulate import compute_nested_exp_utilities
    from choiceforge.cuda_backend import _cupy

    cp = _cupy()

    manifest = json.loads((args.capture / "manifest.json").read_text())
    batches = []
    for item in manifest["batches"]:
        data = np.load(args.capture / item["file"])
        batches.append(
            (
                np.asarray(data["utilities"], dtype=np.float64),
                [str(x) for x in data["alternatives"]],
                np.asarray(data["logsums"], dtype=np.float64),
                item["nest_spec"],
            )
        )

    # Compile and establish the CUDA context outside timed trials.
    mtc21_nested_logsums_cuda(*[batches[0][i] for i in (0, 3, 1)])
    cpu_times, gpu_times = [], []
    max_error = 0.0
    for repetition in range(args.repetitions):
        order = ("cpu", "gpu") if repetition % 2 == 0 else ("gpu", "cpu")
        for backend in order:
            started = time.perf_counter()
            outputs = []
            if backend == "cpu":
                for utilities, alternatives, _, nest in batches:
                    frame = pd.DataFrame(utilities, columns=alternatives)
                    outputs.append(np.log(compute_nested_exp_utilities(frame, nest)["root"].to_numpy()))
                cpu_times.append(time.perf_counter() - started)
            else:
                for utilities, alternatives, _, nest in batches:
                    outputs.append(mtc21_nested_logsums_cuda(utilities, nest, alternatives))
                cp.cuda.get_current_stream().synchronize()
                gpu_times.append(time.perf_counter() - started)
            if repetition == 0:
                for output, (_, _, reference, _) in zip(outputs, batches):
                    max_error = max(max_error, float(np.max(np.abs(output - reference))))

    cpu_median = statistics.median(cpu_times)
    gpu_median = statistics.median(gpu_times)
    rows = sum(batch[0].shape[0] for batch in batches)
    summary = {
        "phase": args.phase,
        "date": args.date,
        "hardware": "NVIDIA RTX A4000 16 GB",
        "boundary": f"{len(batches)} real trip-destination 21-mode nested-logit reductions",
        "design": f"{args.repetitions} interleaved warmed repetitions; GPU includes H2D, kernels, and D2H",
        "batches": len(batches),
        "rows": rows,
        "utility_matrix_megabytes": rows * 21 * 8 / 1_000_000,
        "cpu_activitysim_pandas_seconds": cpu_times,
        "gpu_transfer_inclusive_seconds": gpu_times,
        "cpu_median_seconds": cpu_median,
        "gpu_median_seconds": gpu_median,
        "speedup": cpu_median / gpu_median,
        "median_seconds_saved": cpu_median - gpu_median,
        "max_abs_logsum_error": max_error,
        "live_integration_decision": "integrate only if end-to-end A/B improves and outputs remain equivalent",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
