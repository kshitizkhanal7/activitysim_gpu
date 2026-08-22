"""Summarize strict-CUDA candidate A/B runs with replication gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_phase9_mtc_full import COMPONENTS, SCHEDULING, parse_run, speed_metrics
from verify_phase15_outputs import verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-reports", type=Path)
    parser.add_argument("--require-promotion", action="store_true")
    parser.add_argument("--require-component-promotion", action="store_true")
    parser.add_argument("--phase", type=int, default=15)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    runs = [parse_run(run) for run in manifest["runs"]]
    baseline = [run for run in runs if run["condition"] == "baseline"]
    optimized = [run for run in runs if run["condition"] == "choiceforge"]
    if len(baseline) != len(optimized) or not baseline:
        raise RuntimeError("need equal nonzero baseline and candidate run counts")

    reference = Path(baseline[0]["output"])
    correctness = {
        run["name"]: verify(reference, Path(run["output"]))
        for run in optimized
    }
    keys = ("all_models", "scheduling_suite") + COMPONENTS
    metrics = {
        key: speed_metrics(
            [run[key] for run in baseline],
            [run[key] for run in optimized],
        )
        for key in keys
    }
    repeated = len(baseline) >= 3
    exact_decisions = all(
        result["decision_columns_exact"] for result in correctness.values()
    )
    promotion_gate = {
        "requires_at_least_three_interleaved_pairs": repeated,
        "all_modeled_decisions_exact": exact_decisions,
        "all_candidate_model_times_below_all_baselines": bool(
            repeated and metrics["all_models"].get("all_optimized_faster_than_all_baseline")
        ),
        "all_candidate_destination_times_below_all_baselines": bool(
            repeated and metrics["trip_destination"].get("all_optimized_faster_than_all_baseline")
        ),
        "whole_model_median_speedup_above_one": metrics["all_models"]["speedup"] > 1.0,
        "destination_median_speedup_above_one": metrics["trip_destination"]["speedup"] > 1.0,
    }
    promotion_gate["passed"] = all(promotion_gate.values())
    component_promotion_gate = {
        "requires_at_least_three_interleaved_pairs": repeated,
        "all_modeled_decisions_exact": exact_decisions,
        "all_candidate_destination_times_below_all_baselines": bool(
            repeated
            and metrics["trip_destination"].get(
                "all_optimized_faster_than_all_baseline"
            )
        ),
        "destination_median_speedup_above_one": (
            metrics["trip_destination"]["speedup"] > 1.0
        ),
    }
    component_promotion_gate["passed"] = all(component_promotion_gate.values())
    telemetry = None
    if args.candidate_reports:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.candidate_reports.glob("*.json"))
        ]
        if not records:
            raise RuntimeError("candidate report directory is empty")
        used = [record for record in records if record.get("candidate_used")]
        telemetry = {
            "reports": len(records),
            "candidate_batches": len(used),
            "fallback_batches": sum(bool(record.get("fallback_used")) for record in records),
            "device_resident_utility_batches": sum(
                bool(record.get("device_resident_utility_handoff")) for record in used
            ),
            "utility_device_to_host_bytes": sum(
                int(record.get("utility_device_to_host_bytes", 0)) for record in used
            ),
            "nested_host_to_device_bytes": sum(
                int(record.get("nested_host_to_device_bytes", 0)) for record in used
            ),
            "rows": sum(int(record["rows"]) for record in used),
            "tile_rows": sorted({int(record.get("tile_rows", 1)) for record in used}),
            "dense_row_inputs": sorted(
                {int(record.get("dense_row_inputs", 0)) for record in used}
            ),
            "scalar_inputs": sorted(
                {int(record.get("scalar_inputs", 0)) for record in used}
            ),
            "unique_skim_bindings": sorted(
                {int(record.get("unique_skim_bindings", 0)) for record in used}
            ),
            "skim_index_groups": sorted(
                {int(record.get("skim_index_groups", 0)) for record in used}
            ),
            "grouped_skim_indices": all(
                bool(record.get("grouped_skim_indices")) for record in used
            ),
            "sparse_zero_coefficients": all(
                bool(record.get("sparse_zero_coefficients")) for record in used
            ),
            "expression_dtypes": sorted(
                {record.get("expression_dtype", "float64") for record in used}
            ),
            "active_coefficients": sorted(
                {int(record.get("active_coefficients", 0)) for record in used}
            ),
            "zero_coefficient_ops_skipped": sum(
                int(record.get("zero_coefficient_ops_skipped_per_row", 0))
                * int(record["rows"])
                for record in used
            ),
            "skim_loads_avoided": sum(
                int(record.get("skim_loads_avoided_per_row", 0))
                * int(record["rows"])
                for record in used
            ),
            "ir_cache_hits": sum(bool(record.get("ir_cache_hit")) for record in used),
            "skim_binding_cache_hits": sum(
                int(record.get("skim_binding_cache_hits", 0)) for record in used
            ),
            "skim_binding_cache_misses": sum(
                int(record.get("skim_binding_cache_misses", 0)) for record in used
            ),
            "skim_array_uploads": sum(
                int(record.get("skim_array_uploads", 0)) for record in used
            ),
            "timing_totals_ms": {
                key: sum(float(record.get(key, 0.0)) for record in used)
                for key in (
                    "ir_compile_ms", "binding_resolve_ms", "host_pack_ms",
                    "input_upload_ms", "kernel_ms", "nested_kernel_ms",
                    "nested_download_ms",
                )
            },
        }
    summary = {
        "phase": args.phase,
        "manifest": str(args.manifest),
        "benchmark": {
            key: manifest[key]
            for key in (
                "benchmark", "full_population_households", "zones",
                "data_sha256", "design", "reproducibility",
            )
        },
        "runs": [
            {key: value for key, value in run.items() if key not in {"output", "stdout", "stderr"}}
            for run in runs
        ],
        "metrics": metrics,
        "correctness": {
            "policy": (
                "exact modeled decisions and byte-identical non-trip outputs; "
                "destination_logsum diagnostic max absolute difference <= 1e-4"
            ),
            "candidate_runs": correctness,
            "all_decisions_exact": exact_decisions,
            "maximum_destination_logsum_abs": max(
                result["diagnostic_max_abs"] for result in correctness.values()
            ),
        },
        "candidate_telemetry": telemetry,
        "component_promotion_gate": component_promotion_gate,
        "promotion_gate": promotion_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.require_promotion and not promotion_gate["passed"]:
        raise SystemExit(f"Phase {args.phase} promotion gate failed")
    if args.require_component_promotion and not component_promotion_gate["passed"]:
        raise SystemExit(f"Phase {args.phase} component promotion gate failed")


if __name__ == "__main__":
    main()
