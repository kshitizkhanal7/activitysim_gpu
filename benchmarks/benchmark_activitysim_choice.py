"""Compare ActivitySim's Numba choice_maker with the compatible CUDA kernel.

This isolates the final inverse-CDF sampling stage. It is not a benchmark of
the fused utility/logsum kernel and should not be used as an end-to-end model
speedup claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from activitysim.core.logit import choice_maker

from choiceforge.cuda_backend import CudaChoiceBackend


def samples(fn, repetitions):
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--choosers", type=int, default=100_000)
    parser.add_argument("--alternatives", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = rng.random((args.choosers, args.alternatives), dtype=np.float64)
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    draws = rng.random(args.choosers, dtype=np.float64)

    choice_maker(probabilities[:100], draws[:100])  # Numba warm-up
    cpu = samples(lambda: choice_maker(probabilities, draws), args.repetitions)

    backend = CudaChoiceBackend()
    backend.choose_from_probabilities(probabilities, draws)
    gpu_e2e = samples(
        lambda: backend.choose_from_probabilities(probabilities, draws), args.repetitions
    )

    import cupy as cp

    d_probabilities = cp.asarray(probabilities)
    d_draws = cp.asarray(draws)
    backend.choose_from_probabilities(d_probabilities, d_draws, return_device=True)
    cp.cuda.Stream.null.synchronize()
    gpu_resident = []
    for _ in range(args.repetitions):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        backend.choose_from_probabilities(d_probabilities, d_draws, return_device=True)
        stop.record()
        stop.synchronize()
        gpu_resident.append(cp.cuda.get_elapsed_time(start, stop) / 1000.0)

    expected = choice_maker(probabilities, draws)
    actual = backend.choose_from_probabilities(probabilities, draws)
    result = {
        "schema_version": 1,
        "scope": "ActivitySim choice_maker only; excludes utility evaluation and logsums",
        "workload": vars(args) | {"output": str(args.output) if args.output else None},
        "versions": {
            "activitysim": __import__("activitysim").__version__,
            "numpy": np.__version__,
            "cupy": cp.__version__,
        },
        "activitysim_numba_seconds": cpu,
        "activitysim_numba_median_seconds": statistics.median(cpu),
        "gpu_end_to_end_seconds": gpu_e2e,
        "gpu_end_to_end_median_seconds": statistics.median(gpu_e2e),
        "gpu_resident_seconds": gpu_resident,
        "gpu_resident_median_seconds": statistics.median(gpu_resident),
        "end_to_end_speedup": statistics.median(cpu) / statistics.median(gpu_e2e),
        "resident_speedup": statistics.median(cpu) / statistics.median(gpu_resident),
        "choice_mismatches": int(np.count_nonzero(expected != actual)),
    }
    payload = json.dumps(result, indent=2, default=str)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

