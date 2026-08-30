"""Versioned arithmetic schedules shared by CPU qualification and CUDA codegen.

The public MTC trip-destination Sharrow program materializes 15 float32
features and evaluates a 15-by-1 ``np.dot``.  Numba lowers that operation to
OpenBLAS SGEMV.  For the qualified OpenBLAS 0.3.30 Haswell path, the one-row
fallback reduces three four-term groups from left to right and then consumes
the final three products one at a time.

Keeping the schedule declarative prevents the CPU oracle and CUDA backend from
quietly acquiring different association rules.  The ABI identifier is emitted
in benchmark telemetry and is intentionally changed whenever the schedule or
rounding contract changes.
"""

from __future__ import annotations

import hashlib
import json

import numba as nb
import numpy as np


SHARROW15_ABI_VERSION = "sharrow15-openblas-sgemv-group4-left-v1"
SHARROW15_REDUCTION_GROUPS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12,),
    (13,),
    (14,),
)
_SHARROW15_GROUP_MEMBERS = np.arange(15, dtype=np.int32)
_SHARROW15_GROUP_OFFSETS = np.asarray((0, 4, 8, 12, 13, 14, 15), dtype=np.int32)
SHARROW15_ABI_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": SHARROW15_ABI_VERSION,
            "dtype": "float32",
            "fma": False,
            "groups": SHARROW15_REDUCTION_GROUPS,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

NUMPY_FLOAT32_CHOICE_ABI_VERSION = "numpy246-avx2-exp-pairwise128-v1"


def _pairwise_tree(start: int, count: int):
    if count <= 128:
        return (start, count)
    left_count = (count // 2) & ~7
    return (
        _pairwise_tree(start, left_count),
        _pairwise_tree(start + left_count, count - left_count),
    )


_NUMPY1454_PAIRWISE_TREE = _pairwise_tree(0, 1454)
NUMPY_FLOAT32_CHOICE_ABI_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": NUMPY_FLOAT32_CHOICE_ABI_VERSION,
            "numpy": "2.4.6",
            "simd": "AVX2_FMA3",
            "pairwise_blocksize": 128,
            "alternative_count": 1454,
            "tree": _NUMPY1454_PAIRWISE_TREE,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


@nb.njit(cache=True, fastmath=False)
def sharrow15_cpu_reduce(features, coefficients):
    """Apply the canonical schedule with a float32 rounding point per add."""
    accumulator = np.float32(0.0)
    for group_index in range(len(_SHARROW15_GROUP_OFFSETS) - 1):
        partial = np.float32(0.0)
        for member_index in range(
            _SHARROW15_GROUP_OFFSETS[group_index],
            _SHARROW15_GROUP_OFFSETS[group_index + 1],
        ):
            position = _SHARROW15_GROUP_MEMBERS[member_index]
            product = np.float32(features[position] * coefficients[position])
            partial = np.float32(partial + product)
        accumulator = np.float32(accumulator + partial)
    return accumulator


def sharrow15_cuda_reduction(
    *, intermediate: str = "intermediate", coefficients: str = "coefficients"
) -> str:
    """Generate the CUDA statements for the canonical reduction schedule."""
    lines = ["float utility = 0.0f;", "float absolute_product_sum = 0.0f;"]
    for group_number, group in enumerate(SHARROW15_REDUCTION_GROUPS):
        lines.append(f"float abi_group_{group_number} = 0.0f;")
        for position in group:
            lines.extend(
                (
                    f"const float abi_product_{position} = "
                    f"{coefficients}[{position}] * {intermediate}[{position}];",
                    f"abi_group_{group_number} += abi_product_{position};",
                    f"absolute_product_sum += fabsf(abi_product_{position});",
                )
            )
        lines.append(f"utility += abi_group_{group_number};")
    return "\n    ".join(lines)


def numpy_float32_choice_cuda_helpers(alternative_count: int = 1454) -> str:
    """Emit NumPy 2.4 AVX2 exp and pairwise-sum compatibility helpers.

    The public benchmark has a fixed dense 1,454-zone universe.  Generating
    its reduction tree at compile time avoids CUDA device recursion while
    preserving NumPy's exact association order.
    """
    if alternative_count != 1454:
        raise ValueError("the qualified NumPy choice ABI requires 1,454 alternatives")
    tree = _pairwise_tree(0, alternative_count)
    leaves = []

    def expression(node):
        if isinstance(node[0], int):
            leaf = len(leaves)
            leaves.append(node)
            return f"leaf_{leaf}"
        return f"({expression(node[0])} + {expression(node[1])})"

    root_expression = expression(tree)
    leaf_lines = [
        f"const float leaf_{number} = numpy_pairwise_exp_small(values, {start}, {count});"
        for number, (start, count) in enumerate(leaves)
    ]
    return r'''
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

__device__ __forceinline__ float numpy_pairwise_exp_small(
    const float* values, int start, int count)
{
    float lanes[8];
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        lanes[lane] = numpy_avx2_expf(values[start + lane]);
    }
    const int stop = count - (count % 8);
    for (int offset = 8; offset < stop; offset += 8) {
        #pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            lanes[lane] += numpy_avx2_expf(values[start + offset + lane]);
        }
    }
    float result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                 + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (int offset = stop; offset < count; ++offset) {
        result += numpy_avx2_expf(values[start + offset]);
    }
    return result;
}

__device__ __forceinline__ float numpy_pairwise_exp_sum_1454(const float* values)
{
    __LEAVES__
    return __ROOT__;
}
'''.replace("__LEAVES__", "\n    ".join(leaf_lines)).replace(
        "__ROOT__", root_expression
    )
