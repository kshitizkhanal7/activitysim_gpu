"""Test device scheduling arithmetic policies against the public reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_policy(artifact, exp_policy):
    from choiceforge.cuda_backend import _cupy
    from choiceforge.gpu_scheduling_integration import IntegratedGpuMandatoryScheduler
    from choiceforge.scheduling_compiler import CompiledCudaSchedulingModel

    cp = _cupy()
    runtime = IntegratedGpuMandatoryScheduler(
        artifact, qualify_against_artifact=False, device_boundary_reference=False
    )
    batches = []
    total_errors = total_boundary = boundary_errors = 0
    for number, batch in enumerate(runtime.batches):
        meta, data, host = batch["meta"], batch["device"], batch["host"]
        prepared = runtime.preparer.prepare(
            data["person_rows"],
            data["chooser_values"],
            data["mode_logsum_cache"],
            **runtime._columns(meta),
        )
        model = CompiledCudaSchedulingModel(
            meta["expressions"],
            host["coefficients"],
            batch["model"].schema,
            overflow_protection=False,
            chooser_float64=True,
            dot_policy="sharrow65_lane4",
            exp_policy=exp_policy,
        )
        result = model.choose(
            prepared.chooser_values,
            prepared.row_values,
            runtime.alternative_values,
            prepared.alternative_ids,
            prepared.offsets,
            data["draws"],
            return_device=True,
        )
        selected = prepared.alternative_ids[prepared.offsets[:-1] + result.choices]
        different = selected != data["expected_tdd"]
        boundary = result.boundary_distances <= runtime.boundary_tolerance
        errors = int(cp.count_nonzero(different).item())
        boundaries = int(cp.count_nonzero(boundary).item())
        boundary_error_count = int(cp.count_nonzero(different & boundary).item())
        batches.append(
            {
                "batch": number,
                "choosers": int(selected.size),
                "choice_mismatches": errors,
                "detected_ambiguities": boundaries,
                "ambiguity_mismatches": boundary_error_count,
            }
        )
        total_errors += errors
        total_boundary += boundaries
        boundary_errors += boundary_error_count
        runtime.preparer.assign(data["person_rows"], selected)
    cp.cuda.Stream.null.synchronize()
    return {
        "exp_policy": exp_policy,
        "choice_mismatches": total_errors,
        "detected_ambiguities": total_boundary,
        "ambiguity_mismatches": boundary_errors,
        "all_mismatches_detected_as_ambiguous": total_errors == boundary_errors,
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = [
        run_policy(args.artifact, "libdevice_f32"),
        run_policy(args.artifact, "libdevice_f64_to_f32"),
    ]
    best = min(policies, key=lambda item: item["choice_mismatches"])
    report = {
        "phase": 30,
        "scope": "shared scheduling dot/sum/probability contract with two device exponential lowerings",
        "policies": policies,
        "selected_policy": best["exp_policy"],
        "proof_gates": {
            "six_batches_per_policy": all(len(item["batches"]) == 6 for item in policies),
            "all_mismatches_fail_closed_in_detected_set": all(
                item["all_mismatches_detected_as_ambiguous"] for item in policies
            ),
            "selected_policy_minimizes_reference_mismatches": best["choice_mismatches"]
                == min(item["choice_mismatches"] for item in policies),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(report["proof_gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
