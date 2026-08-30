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
from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class Float32ReductionPolicy:
    """Portable description of every float32 utility addition.

    A policy is data rather than handwritten CUDA.  Its hash changes when the
    term count, grouping, association, or FMA rule changes, so a cached kernel
    cannot silently inherit another model's arithmetic semantics.
    """

    term_count: int
    groups: tuple[tuple[int, ...], ...]
    association: Literal["grouped-left", "ordered-left"]
    contract_fma: bool = False

    def __post_init__(self):
        if self.term_count <= 0:
            raise ValueError("a reduction policy requires at least one term")
        flattened = tuple(position for group in self.groups for position in group)
        if flattened != tuple(range(self.term_count)):
            raise ValueError(
                "reduction groups must cover every term exactly once in source order"
            )
        if self.association == "ordered-left" and any(
            len(group) != 1 for group in self.groups
        ):
            raise ValueError("ordered-left requires one term per group")
        if self.contract_fma:
            raise ValueError("the qualified portable compiler forbids contracted FMA")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": "choiceforge-float32-reduction-v1",
                    "term_count": self.term_count,
                    "groups": self.groups,
                    "association": self.association,
                    "contract_fma": self.contract_fma,
                    "rounding": "ieee754-nearest-even-after-each-operation",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class Float32ProbabilityPolicy:
    """Versioned exponential and probability-reduction contract."""

    alternative_count: int
    exp_policy: Literal["numpy-2.4.6-avx2-fma3"] = "numpy-2.4.6-avx2-fma3"
    pairwise_block_size: int = 128

    def __post_init__(self):
        if self.alternative_count <= 0:
            raise ValueError("a probability policy requires at least one alternative")
        if self.pairwise_block_size != 128:
            raise ValueError("only NumPy's qualified 128-value block is supported")

    @property
    def tree(self):
        return _pairwise_tree(0, self.alternative_count)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": "choiceforge-float32-probability-v1",
                    "alternative_count": self.alternative_count,
                    "exp_policy": self.exp_policy,
                    "pairwise_block_size": self.pairwise_block_size,
                    "tree": self.tree,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class NumericPolicyCompiler:
    """One shared, hash-addressed CPU/CUDA arithmetic compiler."""

    reduction: Float32ReductionPolicy
    probability: Float32ProbabilityPolicy
    compiler_version: str = "choiceforge-numeric-policy-compiler-v1"

    @property
    def abi_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "compiler_version": self.compiler_version,
                    "reduction_sha256": self.reduction.sha256,
                    "probability_sha256": self.probability.sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def cuda_reduction(
        self, *, intermediate: str = "intermediate", coefficients: str = "coefficients"
    ) -> str:
        return float32_reduction_cuda(
            self.reduction,
            intermediate=intermediate,
            coefficients=coefficients,
        )

    def cuda_probability_helpers(self) -> str:
        return numpy_float32_choice_cuda_helpers(
            self.probability.alternative_count
        )


def grouped_left_reduction(term_count: int, group_width: int = 4):
    """Build the explicit OpenBLAS-style group schedule for any term count."""
    if group_width <= 0:
        raise ValueError("group width must be positive")
    term_count = int(term_count)
    groups = []
    full_stop = term_count - (term_count % group_width)
    groups.extend(
        tuple(range(start, start + group_width))
        for start in range(0, full_stop, group_width)
    )
    groups.extend((position,) for position in range(full_stop, term_count))
    return Float32ReductionPolicy(
        term_count=term_count,
        groups=tuple(groups),
        association="grouped-left",
    )


def ordered_left_reduction(term_count: int):
    return Float32ReductionPolicy(
        term_count=int(term_count),
        groups=tuple((position,) for position in range(int(term_count))),
        association="ordered-left",
    )


def reduce_float32(features, coefficients, policy: Float32ReductionPolicy):
    """Reference evaluator generated from the same reduction policy as CUDA."""
    values = np.asarray(features, dtype=np.float32)
    weights = np.asarray(coefficients, dtype=np.float32)
    if values.shape[-1] != policy.term_count or weights.shape != (policy.term_count,):
        raise ValueError("feature/coefficient shape does not match reduction policy")
    rows = values.reshape(-1, policy.term_count)
    output = np.empty(rows.shape[0], dtype=np.float32)
    for row_number, row in enumerate(rows):
        accumulator = np.float32(0.0)
        for group in policy.groups:
            partial = np.float32(0.0)
            for position in group:
                product = np.float32(row[position] * weights[position])
                partial = np.float32(partial + product)
            accumulator = np.float32(accumulator + partial)
        output[row_number] = accumulator
    return output.reshape(values.shape[:-1])


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
    return float32_reduction_cuda(
        Float32ReductionPolicy(
            term_count=15,
            groups=SHARROW15_REDUCTION_GROUPS,
            association="grouped-left",
        ),
        intermediate=intermediate,
        coefficients=coefficients,
    )


def float32_reduction_cuda(
    policy: Float32ReductionPolicy,
    *,
    intermediate: str = "intermediate",
    coefficients: str = "coefficients",
) -> str:
    """Generate CUDA from a validated reduction policy."""
    lines = ["float utility = 0.0f;", "float absolute_product_sum = 0.0f;"]
    for group_number, group in enumerate(policy.groups):
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
    alternative_count = int(alternative_count)
    if alternative_count <= 0:
        raise ValueError("the NumPy choice ABI requires at least one alternative")
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
    if (count < 8) {
        float small = 0.0f;
        for (int offset = 0; offset < count; ++offset) {
            small += numpy_avx2_expf(values[start + offset]);
        }
        return small;
    }
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

__device__ __forceinline__ float __PAIRWISE_FUNCTION__(const float* values)
{
    __LEAVES__
    return __ROOT__;
}
'''.replace("__LEAVES__", "\n    ".join(leaf_lines)).replace(
        "__ROOT__", root_expression
    ).replace("__PAIRWISE_FUNCTION__", f"numpy_pairwise_exp_sum_{alternative_count}")


PHASE42_NUMERIC_COMPILER = NumericPolicyCompiler(
    reduction=grouped_left_reduction(15),
    probability=Float32ProbabilityPolicy(1_454),
)
PHASE42_NUMERIC_COMPILER_VERSION = PHASE42_NUMERIC_COMPILER.compiler_version
PHASE42_NUMERIC_ABI_SHA256 = PHASE42_NUMERIC_COMPILER.abi_sha256
