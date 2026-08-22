"""Phase 1 benchmark against strong CPU implementations.

The workload and random draws are identical for every backend. Timings exclude
JIT compilation through explicit warm-up. GPU transfer-inclusive and resident
measurements are kept separate. All raw samples and correctness comparisons are
written to JSON so summary claims can be independently recomputed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from choiceforge.cuda_backend import CudaChoiceBackend, cuda_available
from choiceforge.numba_backend import linear_choice_numba, numba_available
from choiceforge.reference import linear_choice


def timed_samples(fn: Callable[[], object], repetitions: int) -> list[float]:
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start) / 1e9)
    finally:
        gc.enable()
    return samples


def warm(fn: Callable[[], object], repetitions: int) -> None:
    for _ in range(repetitions):
        fn()


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def correctness(actual, expected) -> dict[str, float | int]:
    finite = np.isfinite(actual.logsums) & np.isfinite(expected.logsums)
    if finite.any():
        errors = np.abs(actual.logsums[finite] - expected.logsums[finite])
        max_error = float(errors.max())
        p999_error = float(np.quantile(errors, 0.999))
    else:
        max_error = 0.0
        p999_error = 0.0
    invalid_mismatches = int(
        np.count_nonzero(np.isfinite(actual.logsums) != np.isfinite(expected.logsums))
    )
    return {
        "choice_mismatches": int(np.count_nonzero(actual.choices != expected.choices)),
        "invalid_row_mismatches": invalid_mismatches,
        "max_logsum_absolute_error": max_error,
        "p999_logsum_absolute_error": p999_error,
    }


def workload(size: int, alternatives: int, features: int, seed: int):
    rng = np.random.default_rng(seed + size)
    x = rng.normal(size=(size, features)).astype(np.float32)
    beta = rng.normal(scale=0.2, size=(alternatives, features)).astype(np.float32)
    constants = rng.normal(scale=0.2, size=alternatives).astype(np.float32)
    availability = rng.random((size, alternatives)) > 0.05
    availability[:, 0] = True
    draws = rng.random(size, dtype=np.float32)
    return x, beta, constants, draws, availability


def benchmark_size(args, size: int, gpu_backend) -> dict[str, object]:
    x, beta, constants, draws, availability = workload(
        size, args.alternatives, args.features, args.seed
    )
    short = min(size, 1024)
    expected = linear_choice(x, beta, constants, draws, availability)
    methods: dict[str, dict[str, object]] = {}
    checks: dict[str, dict[str, float | int]] = {}

    # NumPy uses optimized OpenBLAS matrix multiplication, then materialized
    # vectorized logsum and inverse-CDF operations. Both single-core and the
    # practical physical-core configuration are measured.
    for threads in sorted(set((1, args.cpu_threads))):
        name = f"numpy_materialized_{threads}t"
        with threadpool_limits(limits=threads, user_api="blas"):
            linear_choice(x[:short], beta, constants, draws[:short], availability[:short])
            warm(
                lambda: linear_choice(x, beta, constants, draws, availability),
                args.warmups,
            )
            samples = timed_samples(
                lambda: linear_choice(x, beta, constants, draws, availability),
                args.repetitions,
            )
            actual = linear_choice(x, beta, constants, draws, availability)
        methods[name] = summarize(samples)
        checks[name] = correctness(actual, expected)

    if numba_available():
        configurations = [("numba_fused_serial", False, None)]
        configurations.extend(
            (f"numba_fused_parallel_{threads}t", True, threads)
            for threads in sorted(set((1, args.cpu_threads)))
        )
        for name, parallel, threads in configurations:
            linear_choice_numba(
                x[:short], beta, constants, draws[:short], availability[:short],
                parallel=parallel, threads=threads,
            )
            warm(
                lambda p=parallel, t=threads: linear_choice_numba(
                    x, beta, constants, draws, availability, parallel=p, threads=t
                ),
                args.warmups,
            )
            samples = timed_samples(
                lambda p=parallel, t=threads: linear_choice_numba(
                    x, beta, constants, draws, availability, parallel=p, threads=t
                ),
                args.repetitions,
            )
            actual = linear_choice_numba(
                x, beta, constants, draws, availability,
                parallel=parallel, threads=threads,
            )
            methods[name] = summarize(samples)
            checks[name] = correctness(actual, expected)

    if gpu_backend is not None:
        import cupy as cp

        gpu_backend.linear_choice(
            x[:short], beta, constants, draws[:short], availability[:short]
        )
        warm(
            lambda: gpu_backend.linear_choice(x, beta, constants, draws, availability),
            args.warmups,
        )
        samples = timed_samples(
            lambda: gpu_backend.linear_choice(x, beta, constants, draws, availability),
            args.repetitions,
        )
        actual = gpu_backend.linear_choice(x, beta, constants, draws, availability)
        methods["gpu_transfer_inclusive"] = summarize(samples)
        checks["gpu_transfer_inclusive"] = correctness(actual, expected)

        dx, db, dc, dd, da = map(cp.asarray, (x, beta, constants, draws, availability))
        def resident():
            gpu_backend.linear_choice(dx, db, dc, dd, da, return_device=True)
            cp.cuda.Stream.null.synchronize()

        warm(resident, args.warmups)
        resident_samples = timed_samples(resident, args.repetitions)
        methods["gpu_resident"] = summarize(resident_samples)
        checks["gpu_resident"] = checks["gpu_transfer_inclusive"]

    cpu_candidates = [
        result["median_seconds"]
        for name, result in methods.items()
        if not name.startswith("gpu_")
    ]
    best_cpu = min(cpu_candidates)
    for name, result in methods.items():
        result["speedup_vs_best_cpu"] = best_cpu / result["median_seconds"]

    return {
        "choosers": size,
        "alternatives": args.alternatives,
        "features": args.features,
        "materialized_utility_bytes": size * args.alternatives * 4,
        "methods": methods,
        "correctness": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--alternatives", type=int, default=32)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu_backend = CudaChoiceBackend() if cuda_available() else None
    environment: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "configured_physical_cpu_threads": args.cpu_threads,
        "threadpools": threadpool_info(),
    }
    if numba_available():
        import numba

        environment["numba"] = numba.__version__
    if gpu_backend is not None:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        environment.update({"cupy": cp.__version__, "gpu": gpu_name})

    results = [benchmark_size(args, size, gpu_backend) for size in args.sizes]
    payload = {
        "schema_version": 2,
        "benchmark": "phase1_linear_choice",
        "environment": environment,
        "settings": {
            "sizes": args.sizes,
            "alternatives": args.alternatives,
            "features": args.features,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "cpu_threads": args.cpu_threads,
            "seed": args.seed,
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, default=str)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
