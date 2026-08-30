"""Exercise Phase 42's general numeric policies on the qualified GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from choiceforge.arithmetic_abi import (
    Float32ProbabilityPolicy,
    NumericPolicyCompiler,
    grouped_left_reduction,
    ordered_left_reduction,
    reduce_float32,
)
from choiceforge.cuda_backend import _cupy


ROOT = Path(__file__).resolve().parents[1]


def reduction_probe(cp, term_count: int) -> dict:
    rng = np.random.default_rng(42_100_000 + term_count)
    rows = 257
    features = rng.normal(0.0, 40.0, size=(rows, term_count)).astype(np.float32)
    coefficients = rng.normal(0.0, 8.0, size=term_count).astype(np.float32)
    policy = grouped_left_reduction(term_count)
    expected = reduce_float32(features, coefficients, policy)
    compiler = NumericPolicyCompiler(policy, Float32ProbabilityPolicy(17))
    source = (
        'extern "C" __global__ void general_reduce('
        'const float* features, const float* coefficients, float* output, int rows)\n'
        '{\n'
        '    int row = blockIdx.x * blockDim.x + threadIdx.x;\n'
        '    if (row >= rows) return;\n'
        f'    const float* intermediate = features + (long long)row * {term_count};\n'
        f'    {compiler.cuda_reduction()}\n'
        '    output[row] = utility;\n'
        '}\n'
    )
    kernel = cp.RawKernel(
        source,
        "general_reduce",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    device_output = cp.empty(rows, dtype=cp.float32)
    kernel(
        ((rows + 127) // 128,),
        (128,),
        (cp.asarray(features), cp.asarray(coefficients), device_output, np.int32(rows)),
    )
    actual = cp.asnumpy(device_output)
    mismatch = actual.view(np.uint32) != expected.view(np.uint32)
    return {
        "term_count": term_count,
        "rows": rows,
        "policy_sha256": policy.sha256,
        "bit_mismatches": int(np.count_nonzero(mismatch)),
        "max_abs_difference": float(np.max(np.abs(actual - expected))),
    }


def probability_probe(cp, alternative_count: int) -> dict:
    rng = np.random.default_rng(42_200_000 + alternative_count)
    rows = 257
    values = rng.uniform(-80.0, 8.0, size=(rows, alternative_count)).astype(np.float32)
    expected = np.exp(values).sum(axis=1)
    compiler = NumericPolicyCompiler(
        grouped_left_reduction(3), Float32ProbabilityPolicy(alternative_count)
    )
    function_name = f"numpy_pairwise_exp_sum_{alternative_count}"
    source = compiler.cuda_probability_helpers() + f'''
extern "C" __global__ void general_probability(
    const float* values, float* output, int rows)
{{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    output[row] = {function_name}(
        values + (long long)row * {alternative_count});
}}
'''
    kernel = cp.RawKernel(
        source,
        "general_probability",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    device_output = cp.empty(rows, dtype=cp.float32)
    kernel(
        ((rows + 127) // 128,),
        (128,),
        (cp.asarray(values), device_output, np.int32(rows)),
    )
    actual = cp.asnumpy(device_output)
    mismatch = actual.view(np.uint32) != expected.view(np.uint32)
    return {
        "alternative_count": alternative_count,
        "rows": rows,
        "weight_cells": int(values.size),
        "policy_sha256": compiler.probability.sha256,
        "bit_mismatches": int(np.count_nonzero(mismatch)),
        "max_abs_difference": float(np.max(np.abs(actual - expected))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase42-numeric-compiler-probe.json",
    )
    args = parser.parse_args()
    cp = _cupy()
    reductions = [reduction_probe(cp, count) for count in (1, 3, 5, 15, 17, 31)]
    probabilities = [
        probability_probe(cp, count) for count in (1, 7, 8, 129, 257, 1_454)
    ]
    grouped = NumericPolicyCompiler(
        grouped_left_reduction(15), Float32ProbabilityPolicy(1_454)
    )
    changed_reduction = NumericPolicyCompiler(
        ordered_left_reduction(15), Float32ProbabilityPolicy(1_454)
    )
    changed_alternatives = NumericPolicyCompiler(
        grouped_left_reduction(15), Float32ProbabilityPolicy(1_453)
    )
    gates = {
        "six_reduction_shapes_are_bit_exact": all(
            item["bit_mismatches"] == 0 for item in reductions
        ),
        "six_probability_shapes_are_bit_exact": all(
            item["bit_mismatches"] == 0 for item in probabilities
        ),
        "reduction_mutation_changes_abi_hash": (
            grouped.abi_sha256 != changed_reduction.abi_sha256
        ),
        "alternative_count_mutation_changes_abi_hash": (
            grouped.abi_sha256 != changed_alternatives.abi_sha256
        ),
    }
    document = {
        "phase": 42,
        "compiler_version": grouped.compiler_version,
        "qualified_numeric_abi_sha256": grouped.abi_sha256,
        "reduction_probes": reductions,
        "probability_probes": probabilities,
        "mutation_hashes": {
            "qualified": grouped.abi_sha256,
            "ordered_left": changed_reduction.abi_sha256,
            "1453_alternatives": changed_alternatives.abi_sha256,
        },
        "proof_gates": gates,
        "success": all(gates.values()),
    }
    if not document["success"]:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Phase 42 numeric compiler probe failed: {failed}")
    rendered = json.dumps(document, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
