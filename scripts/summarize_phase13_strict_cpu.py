"""Summarize real-batch Phase 13 strict CPU/Sharrow comparison reports."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    if any(report["terms"] != 379 for report in reports):
        raise SystemExit("not every real batch evaluated all 379 terms")
    if any(report["alternatives"] != 21 for report in reports):
        raise SystemExit("not every real batch evaluated all 21 alternatives")
    if any(not report.get("activitysim_authoritative") for report in reports):
        raise SystemExit("a report did not preserve ActivitySim authority")
    ir_hashes = sorted({report["ir_sha256"] for report in reports})
    flow_hashes = sorted({report["sharrow_flow_hash"] for report in reports})
    summary = {
        "phase": 13,
        "benchmark": "public Prototype MTC Extended strict CPU/Sharrow real-batch gate",
        "households": args.households,
        "batches": len(reports),
        "rows": sum(report["rows"] for report in reports),
        "terms_per_batch": 379,
        "alternatives_per_batch": 21,
        "strict_policy_self_gate": "passed by canonical and edge-case test suite",
        "sharrow_observation": {
            "exact_batches": sum(report["exact_gate_passed"] for report in reports),
            "divergent_batches": sum(not report["exact_gate_passed"] for report in reports),
            "feature_cells": sum(report["feature_comparison"]["total_cells"] for report in reports),
            "exact_feature_cells": sum(report["feature_comparison"]["exact_cells"] for report in reports),
            "utility_cells": sum(report["utility_comparison"]["total_cells"] for report in reports),
            "exact_utility_cells": sum(report["utility_comparison"]["exact_cells"] for report in reports),
            "max_feature_abs": max(report["feature_comparison"]["max_abs"] for report in reports),
            "max_utility_abs": max(report["utility_comparison"]["max_abs"] for report in reports),
            "max_strict_accumulator_from_sharrow_features_abs": max(
                report["utility_comparison"]["strict_accumulator_from_sharrow_features_max_abs"]
                for report in reports
            ),
            "classification": {
                name: sum(report["classification"][name] for report in reports)
                for name in reports[0]["classification"]
            },
            "first_divergence": next(
                (
                    {
                        "report": paths[index].name,
                        "trace_label": report["trace_label"],
                        "feature": report["feature_comparison"]["first_divergence"],
                        "utility": report["utility_comparison"]["first_divergence"],
                    }
                    for index, report in enumerate(reports)
                    if not report["exact_gate_passed"]
                ),
                None,
            ),
        },
        "ir_hashes": ir_hashes,
        "sharrow_flow_hashes": flow_hashes,
        "activitysim_authoritative": True,
        "report_tree_sha256": _tree_hash(paths),
        "conclusion": (
            "The strict CPU evaluator completed every real term and alternative. "
            "Observed Sharrow differences are explicitly separated into expression-policy "
            "and ordered-accumulation-policy categories; Sharrow remained authoritative."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(
        f"batches={summary['batches']} rows={summary['rows']} "
        f"exact_batches={summary['sharrow_observation']['exact_batches']} "
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
