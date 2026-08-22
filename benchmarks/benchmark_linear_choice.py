"""Reproducible CPU/GPU benchmark for the fused linear-choice kernel."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from choiceforge.cuda_backend import CudaChoiceBackend, cuda_available
from choiceforge.reference import linear_choice


def elapsed_samples(fn, repetitions: int) -> list[float]:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choosers", type=int, default=100_000)
    parser.add_argument("--alternatives", type=int, default=32)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.choosers, args.features)).astype(np.float32)
    beta = rng.normal(scale=0.2, size=(args.alternatives, args.features)).astype(np.float32)
    constants = rng.normal(scale=0.2, size=args.alternatives).astype(np.float32)
    availability = rng.random((args.choosers, args.alternatives)) > 0.05
    availability[:, 0] = True
    draws = rng.random(args.choosers, dtype=np.float32)

    # Warm CPU libraries before measuring.
    linear_choice(x[: min(1024, len(x))], beta, constants, draws[: min(1024, len(x))], availability[: min(1024, len(x))])
    cpu_samples = elapsed_samples(
        lambda: linear_choice(x, beta, constants, draws, availability), args.repetitions
    )

    result = {
        "schema_version": 1,
        "workload": vars(args) | {"output": str(args.output) if args.output else None},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "cpu_materialized_seconds": cpu_samples,
        "cpu_median_seconds": statistics.median(cpu_samples),
        "materialized_utility_bytes": args.choosers * args.alternatives * 4,
    }

    if cuda_available():
        import cupy as cp

        backend = CudaChoiceBackend()
        # Compile and validate before timing.
        backend.linear_choice(x[:1024], beta, constants, draws[:1024], availability[:1024])
        expected = linear_choice(x, beta, constants, draws, availability)
        # Warm the exact workload size as allocation pools and transfer staging
        # can have size-dependent one-time costs that are not steady-state work.
        backend.linear_choice(x, beta, constants, draws, availability)
        end_to_end_samples = elapsed_samples(
            lambda: backend.linear_choice(x, beta, constants, draws, availability),
            args.repetitions,
        )

        dx, db, dc, dd, da = map(cp.asarray, (x, beta, constants, draws, availability))
        backend.linear_choice(dx, db, dc, dd, da, return_device=True)
        cp.cuda.Stream.null.synchronize()
        resident_samples = []
        for _ in range(args.repetitions):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            device_result = backend.linear_choice(dx, db, dc, dd, da, return_device=True)
            stop.record()
            stop.synchronize()
            resident_samples.append(cp.cuda.get_elapsed_time(start, stop) / 1000.0)
        actual = backend.linear_choice(dx, db, dc, dd, da)
        choice_mismatches = int(np.count_nonzero(actual.choices != expected.choices))
        max_logsum_error = float(np.max(np.abs(actual.logsums - expected.logsums)))
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu_props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        result["environment"].update({"cupy": cp.__version__, "gpu": gpu_name})
        result.update(
            {
                "gpu_end_to_end_seconds": end_to_end_samples,
                "gpu_end_to_end_median_seconds": statistics.median(end_to_end_samples),
                "gpu_resident_seconds": resident_samples,
                "gpu_resident_median_seconds": statistics.median(resident_samples),
                "end_to_end_speedup": statistics.median(cpu_samples) / statistics.median(end_to_end_samples),
                "resident_speedup": statistics.median(cpu_samples) / statistics.median(resident_samples),
                "choice_mismatches": choice_mismatches,
                "max_logsum_absolute_error": max_logsum_error,
            }
        )

    payload = json.dumps(result, indent=2, default=str)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
