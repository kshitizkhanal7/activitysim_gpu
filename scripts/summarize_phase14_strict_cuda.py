"""Summarize real-batch Phase 14 strict IR CPU/CUDA reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--households", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.reports.glob("batch_*.json"))
    if not paths:
        raise SystemExit(f"no batch reports found in {args.reports}")
    reports = [json.loads(path.read_text(encoding="utf8")) for path in paths]
    failures = [path.name for path, report in zip(paths, reports) if not report["exact_gate_passed"]]
    if failures:
        raise SystemExit(f"strict CPU/CUDA gate failed: {failures[:5]}")
    if any(report["terms"] != 379 or report["alternatives"] != 21 for report in reports):
        raise SystemExit("not every real batch covered all 379 terms and 21 alternatives")
    if any(not report.get("activitysim_authoritative") for report in reports):
        raise SystemExit("a real batch did not preserve ActivitySim authority")
    kernels = [report["kernel"] for report in reports]
    summary = {
        "phase": 14,
        "benchmark": "public Prototype MTC Extended generated strict-IR CUDA real-batch gate",
        "households": args.households,
        "batches": len(reports),
        "rows": sum(report["rows"] for report in reports),
        "terms_per_batch": 379,
        "alternatives_per_batch": 21,
        "exact_batches": len(reports),
        "exact_feature_cells": sum(report["feature_comparison"]["exact_cells"] for report in reports),
        "feature_cells": sum(report["feature_comparison"]["total_cells"] for report in reports),
        "exact_utility_cells": sum(report["utility_comparison"]["exact_cells"] for report in reports),
        "utility_cells": sum(report["utility_comparison"]["total_cells"] for report in reports),
        "max_feature_abs": max(report["feature_comparison"]["max_abs"] for report in reports),
        "max_utility_abs": max(report["utility_comparison"]["max_abs"] for report in reports),
        "ir_hashes": sorted({report["ir_sha256"] for report in reports}),
        "kernel_cache_keys": sorted({kernel["cache_key"] for kernel in kernels}),
        "kernel_source_sha256": sorted({kernel["source_sha256"] for kernel in kernels}),
        "compiled_calls": sum(kernel["compiled_this_call"] for kernel in kernels),
        "timing_diagnostic_only": {
            "median_host_to_device_ms": statistics.median(kernel["host_to_device_ms"] for kernel in kernels),
            "median_kernel_ms": statistics.median(kernel["kernel_ms"] for kernel in kernels),
            "median_device_to_host_ms": statistics.median(kernel["device_to_host_ms"] for kernel in kernels),
        },
        "activitysim_authoritative": True,
        "report_tree_sha256": _tree_hash(paths),
        "conclusion": (
            "The generated strict-IR CUDA target matched the strict CPU oracle exactly "
            "for every feature and utility cell in every real batch. Timings are diagnostic "
            "because input packing and feature downloads remain qualification scaffolding."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(
        f"batches={summary['batches']} rows={summary['rows']} "
        f"feature_cells={summary['feature_cells']} utility_cells={summary['utility_cells']} "
        f"sha256={summary['report_tree_sha256']}"
    )


def _tree_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
