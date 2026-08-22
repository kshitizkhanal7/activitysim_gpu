"""Summarize the Phase 15 device-resident strict CUDA candidate gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def tree_sha256(path: Path) -> str:
    lines = []
    for item in sorted(path.glob("*.json")):
        lines.append(f"{item.name}:{hashlib.sha256(item.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def median(records, key):
    return statistics.median(float(record[key]) for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--exact-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--households", type=int, required=True)
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in sorted(args.reports.glob("*.json"))]
    exact = [json.loads(path.read_text()) for path in sorted(args.exact_reports.glob("*.json"))]
    if not records or not exact:
        raise RuntimeError("Phase 15 needs nonempty candidate and exact report sets")
    if any(not record.get("candidate_used") or record.get("fallback_used") for record in records):
        raise RuntimeError("Phase 15 candidate used a fallback")
    if any(not record.get("exact_gate_passed") for record in exact):
        raise RuntimeError("Phase 15 strict CPU/CUDA exact gate failed")
    if len(records) != len(exact):
        raise RuntimeError("candidate and exact report counts differ")

    summary = {
        "phase": 15,
        "benchmark": "public Prototype MTC Extended device-resident generated strict-CUDA candidate",
        "households": args.households,
        "batches": len(records),
        "rows": sum(record["rows"] for record in records),
        "terms_per_batch": sorted({record["terms"] for record in records}),
        "alternatives_per_batch": sorted({record["alternatives"] for record in records}),
        "candidate_batches": sum(bool(record["candidate_used"]) for record in records),
        "fallback_batches": sum(bool(record["fallback_used"]) for record in records),
        "exact_cpu_cuda_batches": sum(bool(record["exact_gate_passed"]) for record in exact),
        "exact_feature_cells": sum(record["feature_comparison"]["exact_cells"] for record in exact),
        "feature_cells": sum(record["feature_comparison"]["total_cells"] for record in exact),
        "exact_utility_cells": sum(record["utility_comparison"]["exact_cells"] for record in exact),
        "utility_cells": sum(record["utility_comparison"]["total_cells"] for record in exact),
        "max_feature_abs": max(record["feature_comparison"]["max_abs"] for record in exact),
        "max_utility_abs": max(record["utility_comparison"]["max_abs"] for record in exact),
        "device_resident_utility_batches": sum(
            bool(record["device_resident_utility_handoff"]) for record in records
        ),
        "utility_device_to_host_bytes": sum(record["utility_device_to_host_bytes"] for record in records),
        "nested_host_to_device_bytes": sum(record["nested_host_to_device_bytes"] for record in records),
        "timing_medians_ms": {
            "binding_resolve": median(records, "binding_resolve_ms") if "binding_resolve_ms" in records[0] else None,
            "host_pack": median(records, "host_pack_ms"),
            "input_upload": median(records, "input_upload_ms"),
            "coefficient_upload": median(records, "coefficient_upload_ms"),
            "generated_utility_kernel": median(records, "kernel_ms"),
            "nested_kernel": median(records, "nested_kernel_ms"),
            "nested_download": median(records, "nested_download_ms"),
        },
        "compiled_calls": sum(bool(record["compiled_this_call"]) for record in records),
        "coefficient_cache_hits": sum(bool(record["coefficient_cache_hit"]) for record in records),
        "source_sha256": sorted({record["source_sha256"] for record in records}),
        "candidate_report_tree_sha256": tree_sha256(args.reports),
        "exact_report_tree_sha256": tree_sha256(args.exact_reports),
        "success": True,
        "claim_boundary": "correctness and device-residency gate; full-model superiority requires repeated interleaved A/B trials",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
