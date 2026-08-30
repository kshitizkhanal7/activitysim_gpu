"""Compare CUDA exponential policies with NumPy's AVX2 float32 ufunc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from choiceforge.cuda_backend import _cupy
from choiceforge.arithmetic_abi import numpy_float32_choice_cuda_helpers


SOURCE = r'''
__device__ __forceinline__ float numpy_avx2_expf(float value)
{
    if (isnan(value)) return value;
    if (value >= 88.72283935546875f) return __int_as_float(0x7f800000);
    if (value <= -103.97208404541015625f) return 0.0f;
    float quadrant = value * 1.4426950408889634074f;
    quadrant = (quadrant + 12582912.0f) - 12582912.0f;
    float reduced = fmaf(quadrant, -6.93145752e-1f, value);
    reduced = fmaf(quadrant, -1.42860677e-6f, reduced);
    float numerator = fmaf(5.082762527590693718096e-04f, reduced,
                           6.757896990527504603057e-03f);
    numerator = fmaf(numerator, reduced, 5.114512081637298353406e-02f);
    numerator = fmaf(numerator, reduced, 2.473615434895520810817e-01f);
    numerator = fmaf(numerator, reduced, 7.257664613233124478488e-01f);
    numerator = fmaf(numerator, reduced, 9.999999999980870924916e-01f);
    float denominator = fmaf(2.159509375685829852307e-02f, reduced,
                             -2.742335390411667452936e-01f);
    denominator = fmaf(denominator, reduced, 1.0f);
    float polynomial = numerator / denominator;
    int q = __float2int_rn(quadrant);
    if (q <= -125) {
        int difference = -q - 125;
        int scale_divisor = 1 << difference;
        q = -125;
        polynomial = __int_as_float(__float_as_int(polynomial) + (q << 23));
        return polynomial / (float)scale_divisor;
    }
    return __int_as_float(__float_as_int(polynomial) + (q << 23));
}

extern "C" __global__ void exp_policies(
    const float* values, float* outputs, long long count)
{
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = values[index];
    outputs[index * 3] = expf(value);
    outputs[index * 3 + 1] = (float)exp((double)value);
    outputs[index * 3 + 2] = numpy_avx2_expf(value);
}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cp = _cupy()
    rng = np.random.default_rng(410041)
    values = np.concatenate(
        (
            rng.uniform(-104.0, 20.0, size=1_000_000).astype(np.float32),
            np.linspace(-104.0, 89.0, 100_000, dtype=np.float32),
        )
    )
    expected = np.exp(values)
    device_values = cp.asarray(values)
    device_outputs = cp.empty((len(values), 3), dtype=cp.float32)
    kernel = cp.RawKernel(
        SOURCE,
        "exp_policies",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    block = 256
    kernel(
        ((len(values) + block - 1) // block,),
        (block,),
        (device_values, device_outputs, np.int64(len(values))),
    )
    outputs = cp.asnumpy(device_outputs)
    comparisons = {}
    for column, name in enumerate(("libdevice_f32", "libdevice_f64_to_f32", "numpy_avx2")):
        actual = outputs[:, column]
        mismatch = actual.view(np.uint32) != expected.view(np.uint32)
        finite = np.isfinite(actual) & np.isfinite(expected)
        comparisons[name] = {
            "bit_mismatches": int(np.count_nonzero(mismatch)),
            "mismatch_input_min": float(values[mismatch].min(initial=np.inf)),
            "mismatch_input_max": float(values[mismatch].max(initial=-np.inf)),
            "mismatches_ge_minus_80": int(np.count_nonzero(mismatch & (values >= -80.0))),
            "mismatches_ge_minus_20": int(np.count_nonzero(mismatch & (values >= -20.0))),
            "max_abs_difference": float(np.max(np.abs(actual[finite] - expected[finite]))),
        }

    rows = rng.uniform(-100.0, 5.0, size=(5_000, 1_454)).astype(np.float32)
    expected_sums = np.exp(rows).sum(axis=1)
    pairwise_source = numpy_float32_choice_cuda_helpers() + r'''
extern "C" __global__ void pairwise_rows(
    const float* values, float* sums, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) sums[row] = numpy_pairwise_exp_sum_1454(values + (long long)row * 1454);
}
'''
    device_rows = cp.asarray(rows)
    device_sums = cp.empty(len(rows), dtype=cp.float32)
    pairwise = cp.RawKernel(
        pairwise_source,
        "pairwise_rows",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    pairwise(((len(rows) + 127) // 128,), (128,), (device_rows, device_sums, np.int32(len(rows))))
    actual_sums = cp.asnumpy(device_sums)
    sum_mismatch = actual_sums.view(np.uint32) != expected_sums.view(np.uint32)
    document = {
        "values": len(values),
        "comparisons": comparisons,
        "pairwise_rows": len(rows),
        "pairwise_cells": int(rows.size),
        "pairwise_bit_mismatches": int(np.count_nonzero(sum_mismatch)),
        "pairwise_max_abs_difference": float(
            np.max(np.abs(actual_sums - expected_sums))
        ),
    }
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
