"""Reproducible microbenchmark for the Phase 12 device utility boundary.

This is intentionally not an ActivitySim end-to-end claim.  It measures only
the numeric portion after a model-specific compiler has lowered expressions and
skim lookups into a feature matrix.  The CPU equivalence check is mandatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from choiceforge.destination_utility import (
    LoweredDestinationUtility,
    mtc21_logsums_from_lowered_cuda,
)
from choiceforge.nested_logit import MTC21_ALTERNATIVES


NEST = {
    "name": "root", "coefficient": 1.0, "alternatives": [
        {"name": "AUTO", "coefficient": 0.72, "alternatives": [
            {"name": "DRIVEALONE", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[:2])},
            {"name": "SHAREDRIDE2", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[2:4])},
            {"name": "SHAREDRIDE3", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[4:6])},
        ]},
        {"name": "NONMOTORIZED", "coefficient": 0.72, "alternatives": list(MTC21_ALTERNATIVES[6:8])},
        {"name": "TRANSIT", "coefficient": 0.72, "alternatives": [
            {"name": "WALKACCESS", "coefficient": 0.5, "alternatives": list(MTC21_ALTERNATIVES[8:13])},
            {"name": "DRIVEACCESS", "coefficient": 0.5, "alternatives": list(MTC21_ALTERNATIVES[13:18])},
        ]},
        {"name": "RIDEHAIL", "coefficient": 0.36, "alternatives": list(MTC21_ALTERNATIVES[18:])},
    ],
}


def median_ms(fn, repetitions):
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return value, float(np.median(samples)), samples


def _lse(values, scale):
    high = np.max(values / scale, axis=1)
    return high + np.log(np.exp(values / scale - high[:, None]).sum(axis=1))


def numpy_mtc21_logsums(utility):
    """Independent vectorized float64 reference for the canonical MTC nest."""
    auto_c, auto_sub_c, nm_c, transit_c, transit_sub_c, ridehail_c = (.72, .35, .72, .72, .5, .36)
    da = auto_sub_c * _lse(utility[:, 0:2], auto_c * auto_sub_c)
    sr2 = auto_sub_c * _lse(utility[:, 2:4], auto_c * auto_sub_c)
    sr3 = auto_sub_c * _lse(utility[:, 4:6], auto_c * auto_sub_c)
    auto_high = np.maximum(np.maximum(da, sr2), sr3)
    auto_log = auto_c * (auto_high + np.log(np.exp(da-auto_high) + np.exp(sr2-auto_high) + np.exp(sr3-auto_high)))
    nm_log = nm_c * _lse(utility[:, 6:8], nm_c)
    walk_access = transit_sub_c * _lse(utility[:, 8:13], transit_c * transit_sub_c)
    drive_access = transit_sub_c * _lse(utility[:, 13:18], transit_c * transit_sub_c)
    transit_high = np.maximum(walk_access, drive_access)
    transit_log = transit_c * (transit_high + np.log(np.exp(walk_access-transit_high) + np.exp(drive_access-transit_high)))
    ridehail_log = ridehail_c * _lse(utility[:, 18:21], ridehail_c)
    root_high = np.maximum(np.maximum(auto_log, nm_log), np.maximum(transit_log, ridehail_log))
    return root_high + np.log(np.exp(auto_log-root_high) + np.exp(nm_log-root_high) + np.exp(transit_log-root_high) + np.exp(ridehail_log-root_high))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.features < 1 or args.repetitions < 3:
        raise ValueError("rows/features must be positive and repetitions must be at least 3")

    rng = np.random.default_rng(20260817)
    features = rng.normal(size=(args.rows, args.features))
    model = LoweredDestinationUtility(
        tuple(f"feature_{i}" for i in range(args.features)), MTC21_ALTERNATIVES,
        rng.normal(size=(args.features, 21)), rng.normal(size=21),
    )
    # Warm GPU compilation and allocations before collecting timing samples.
    mtc21_logsums_from_lowered_cuda(model, features[:256], NEST)
    cpu_result, cpu_ms, cpu_samples = median_ms(
        lambda: numpy_mtc21_logsums(model.cpu_reference(features)), args.repetitions
    )
    gpu_result, gpu_ms, gpu_samples = median_ms(
        lambda: mtc21_logsums_from_lowered_cuda(model, features, NEST), args.repetitions
    )
    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-11, atol=1e-11)
    _, telemetry = mtc21_logsums_from_lowered_cuda(
        model, features, NEST, return_telemetry=True
    )
    payload = {
        "phase": "12",
        "claim_scope": "lowered numeric utility plus MTC-21 logsum microbenchmark; not ActivitySim end-to-end",
        "rows": args.rows, "features": args.features, "alternatives": 21,
        "repetitions": args.repetitions,
        "cpu_reference_pipeline_median_ms": cpu_ms,
        "cpu_reference_pipeline_samples_ms": cpu_samples,
        "gpu_pipeline_median_ms": gpu_ms,
        "gpu_pipeline_samples_ms": gpu_samples,
        "gpu_pipeline_speedup": cpu_ms / gpu_ms,
        "correctness": {"rtol": 1e-11, "atol": 1e-11, "passed": True},
        "telemetry_last_run": {
            "utility": telemetry.utility.__dict__,
            "nested_logsum": telemetry.nested_logsum.__dict__,
        },
        "note": "The CPU reference is a vectorized NumPy implementation of the same lowered linear utility and canonical MTC-21 nesting.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
