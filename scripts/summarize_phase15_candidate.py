"""Summarize a device-resident strict CUDA candidate gate."""

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
    parser.add_argument("--mode-reports", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--households", type=int, required=True)
    parser.add_argument("--phase", type=int, default=15)
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in sorted(args.reports.glob("*.json"))]
    exact = [json.loads(path.read_text()) for path in sorted(args.exact_reports.glob("*.json"))]
    if not records or not exact:
        raise RuntimeError(f"Phase {args.phase} needs nonempty candidate and exact report sets")
    if any(not record.get("candidate_used") or record.get("fallback_used") for record in records):
        raise RuntimeError(f"Phase {args.phase} candidate used a fallback")
    if any(not record.get("exact_gate_passed") for record in exact):
        raise RuntimeError(f"Phase {args.phase} strict CPU/CUDA exact gate failed")
    if len(records) != len(exact):
        raise RuntimeError("candidate and exact report counts differ")
    mode_records = []
    if args.mode_reports:
        mode_records = [
            json.loads(path.read_text())
            for path in sorted(args.mode_reports.glob("*.json"))
        ]
        if not mode_records:
            raise RuntimeError("mode report directory is empty")
        if any(
            not record.get("candidate_used") or record.get("fallback_used")
            for record in mode_records
        ):
            raise RuntimeError("trip-mode generated utility candidate used a fallback")

    summary = {
        "phase": args.phase,
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
            "plan_build": median(records, "plan_build_ms") if "plan_build_ms" in records[0] else None,
            "generated_utility_kernel": median(records, "kernel_ms"),
            "nested_kernel": median(records, "nested_kernel_ms"),
            "nested_download": median(records, "nested_download_ms"),
        },
        "compiled_calls": sum(bool(record["compiled_this_call"]) for record in records),
        "coefficient_cache_hits": sum(bool(record["coefficient_cache_hit"]) for record in records),
        "persistent_plan": all(bool(record.get("persistent_plan")) for record in records),
        "plan_cache_hits": sum(bool(record.get("plan_cache_hit")) for record in records),
        "reusable_workspace": all(
            bool(record.get("reusable_workspace")) for record in records
        ),
        "workspace_cache_hits": sum(
            bool(record.get("workspace_cache_hit")) for record in records
        ),
        "tile_rows": sorted({int(record.get("tile_rows", 1)) for record in records}),
        "dense_row_inputs": sorted(
            {int(record.get("dense_row_inputs", 0)) for record in records}
        ),
        "scalar_inputs": sorted(
            {int(record.get("scalar_inputs", 0)) for record in records}
        ),
        "unique_skim_bindings": sorted(
            {int(record.get("unique_skim_bindings", 0)) for record in records}
        ),
        "skim_index_groups": sorted(
            {int(record.get("skim_index_groups", 0)) for record in records}
        ),
        "grouped_skim_indices": all(
            bool(record.get("grouped_skim_indices")) for record in records
        ),
        "sparse_zero_coefficients": all(
            bool(record.get("sparse_zero_coefficients")) for record in records
        ),
        "expression_dtypes": sorted(
            {record.get("expression_dtype", "float64") for record in records}
        ),
        "active_coefficients": sorted(
            {int(record.get("active_coefficients", 0)) for record in records}
        ),
        "zero_coefficient_ops_skipped": sum(
            int(record.get("zero_coefficient_ops_skipped_per_row", 0))
            * int(record["rows"])
            for record in records
        ),
        "skim_loads_avoided": sum(
            int(record.get("skim_loads_avoided_per_row", 0)) * int(record["rows"])
            for record in records
        ),
        "ir_cache_hits": sum(bool(record.get("ir_cache_hit")) for record in records),
        "skim_binding_cache_hits": sum(
            int(record.get("skim_binding_cache_hits", 0)) for record in records
        ),
        "skim_binding_cache_misses": sum(
            int(record.get("skim_binding_cache_misses", 0)) for record in records
        ),
        "skim_array_uploads": sum(
            int(record.get("skim_array_uploads", 0)) for record in records
        ),
        "source_sha256": sorted({record["source_sha256"] for record in records}),
        "candidate_report_tree_sha256": tree_sha256(args.reports),
        "exact_report_tree_sha256": tree_sha256(args.exact_reports),
        "mode_candidate": (
            {
                "reports": len(mode_records),
                "rows": sum(int(record["rows"]) for record in mode_records),
                "fallbacks": 0,
                "plan_cache_hits": sum(
                    bool(record.get("plan_cache_hit")) for record in mode_records
                ),
                "workspace_cache_hits": sum(
                    bool(record.get("workspace_cache_hit"))
                    for record in mode_records
                ),
                "source_sha256": sorted(
                    {record["source_sha256"] for record in mode_records}
                ),
                "report_tree_sha256": tree_sha256(args.mode_reports),
            }
            if mode_records else None
        ),
        "success": True,
        "claim_boundary": "correctness and device-residency gate; full-model superiority requires repeated interleaved A/B trials",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
