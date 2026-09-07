"""Compare identical current-window reductions on compiled CPU and CUDA.

Six frozen public input snapshots are reconstructed for measurement; neither
backend receives reference outputs. All input/output transfer and allocations
are included in the GPU timer. Compilation is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
import time

import numba
import numpy as np
from choiceforge.gpu_scheduling_pipeline import build_tdd_footprints
from choiceforge.phase57_live_scheduling import period_pairs_cuda


@numba.njit(cache=True, parallel=True)
def cpu_pairs(windows, footprints, slots):
    n, periods = windows.shape
    a = len(footprints)
    first = np.full((n, 25), a, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    for owner in numba.prange(n):
        for alt in range(a):
            available = True
            for p in range(periods):
                c, w = footprints[alt, p], windows[owner, p]
                if ((c == 2 and (w == 2 or w == 7))
                    or (c == 4 and (w == 4 or w == 7))
                    or (c == 7 and (w == 2 or w == 4 or w == 6 or w == 7))
                    or (c == 6 and w == 7)):
                    available = False
                    break
            if available:
                first[owner, slots[alt]] = min(first[owner, slots[alt]], alt)
                counts[owner] += 1
    return first, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("benchmark-results/phase21-scheduling-inputs"))
    parser.add_argument("--report", type=Path, default=Path("benchmark-results/phase57-live-pairs-microbenchmark.json"))
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    with np.load(args.inputs / manifest["common_file"]) as common:
        alternatives = common["alternative_values"].copy()
    footprints, _ = build_tdd_footprints(alternatives)
    footprints = np.ascontiguousarray(footprints, dtype=np.int8)
    periods = np.searchsorted([5, 9, 14, 18], alternatives[:, :2], side="left")
    slots = np.ascontiguousarray(periods[:, 0]*5+periods[:, 1], dtype=np.int32)
    windows = np.zeros((manifest["person_count"], footprints.shape[1]), dtype=np.int8)
    snapshots = []
    for record in manifest["batches"]:
        with np.load(args.inputs / record["file"]) as batch:
            people = batch["person_rows"]
            snapshots.append(np.ascontiguousarray(windows[people]))
            windows[people] |= footprints[batch["expected_tdd"]]
    max_threads = numba.config.NUMBA_NUM_THREADS
    thread_counts = sorted({1, min(4, max_threads), min(8, max_threads), min(24, max_threads), max_threads})
    started = time.perf_counter()
    numba.set_num_threads(1)
    cpu_pairs(snapshots[0][:10], footprints, slots)
    cpu_compile = time.perf_counter()-started
    started = time.perf_counter()
    period_pairs_cuda(snapshots[0][:10], footprints, slots)
    gpu_compile = time.perf_counter()-started
    runs = []
    for trial in range(args.repetitions):
        measured = {}
        modes = [*(f"cpu_{n}" for n in thread_counts), "gpu"]
        if trial % 2:
            modes.reverse()
        results = {}
        for mode in modes:
            if mode != "gpu":
                numba.set_num_threads(int(mode.split("_")[1]))
            started = time.perf_counter()
            fn = period_pairs_cuda if mode == "gpu" else cpu_pairs
            results[mode] = [fn(window, footprints, slots) for window in snapshots]
            measured[mode] = time.perf_counter()-started
        for mode, outputs in results.items():
            for actual, expected in zip(outputs, results["cpu_1"]):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
        runs.append({"trial": trial+1, "seconds": measured, "exact": True})
    medians = {mode: median(run["seconds"][mode] for run in runs) for mode in modes}
    best_cpu = min((mode for mode in modes if mode.startswith("cpu_")), key=medians.get)
    from choiceforge.cuda_backend import _cupy
    cp = _cupy()
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    input_paths = [args.inputs / "manifest.json", args.inputs / manifest["common_file"],
                   *(args.inputs / record["file"] for record in manifest["batches"])]
    report = {
        "contract": "phase57-current-window-cpu-cuda-reduction-v1",
        "platform": platform.platform(), "numba_version": numba.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "cupy_version": cp.__version__,
        "activitysim_version": version("activitysim"),
        "cpu_description": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "gpu_name": device["name"].decode() if isinstance(device["name"], bytes) else device["name"],
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
        "cpu_thread_counts": thread_counts, "batches": len(snapshots),
        "chooser_rows": sum(len(w) for w in snapshots),
        "feasible_rows": sum(int(count.sum()) for _, count in results["gpu"]),
        "representative_rows": sum(int((first < len(footprints)).sum()) for first, _ in results["gpu"]),
        "repetitions": args.repetitions, "runs": runs, "median_seconds": medians,
        "strongest_measured_cpu": best_cpu,
        "speedup_vs_strongest_cpu": medians[best_cpu]/medians["gpu"],
        "cold_cpu_compile_seconds": cpu_compile, "cold_gpu_compile_seconds": gpu_compile,
        "gpu_includes_all_host_device_transfers_and_allocations": True,
        "all_outputs_exact": True,
        "scope": "six public timetable snapshots, warmed compiled functions; separate from full ActivitySim timing",
    }
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
