"""Fused CUDA compiler for small fixed-alternative ActivitySim MNL models."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .api import ChoiceResult
from .cuda_backend import _cupy
from .gpu_native import GpuOnlyViolation, _is_cuda_array
from .scheduling_compiler import compile_cuda_expression


def _source(
    expressions: tuple[str, ...],
    coefficients: tuple[tuple[float, ...], ...],
    columns: tuple[str, ...],
) -> str:
    alternatives = len(coefficients[0])
    declarations = "\n".join(
        f"  const double {name} = values[row * {len(columns)} + {position}];"
        for position, name in enumerate(columns)
    )
    accumulators = "\n".join(f"  double u{alt} = 0.0;" for alt in range(alternatives))
    terms = []
    for term_number, (expression, row) in enumerate(zip(expressions, coefficients)):
        compiled = compile_cuda_expression(expression)
        terms.append(f"  const double term_{term_number} = (double)({compiled});")
        for alternative, coefficient in enumerate(row):
            if coefficient:
                terms.append(
                    f"  u{alternative} += term_{term_number} * {float(coefficient)!r};"
                )
    utility_array = ", ".join(f"u{alt}" for alt in range(alternatives))
    return f'''\
extern "C" __global__ void fused_fixed_mnl(
 const double* values, const double* dynamic_value, const double* draws,
 int n, int* choices, double* logsums)
{{
 const int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= n) return;
{declarations}
  const double auto_ownership = dynamic_value[row];
{accumulators}
{chr(10).join(terms)}
  double utility[{alternatives}] = {{{utility_array}}};
  double maximum = utility[0];
  int largest = 0;
  for (int a=1; a<{alternatives}; ++a) {{
    if (utility[a] > maximum) {{ maximum = utility[a]; largest = a; }}
  }}
  double weight[{alternatives}];
  double total = 0.0;
  for (int a=0; a<{alternatives}; ++a) {{
    double value = exp(utility[a] - maximum);
    if (value <= 1.0e-300) value = 0.0;
    weight[a] = value;
    total += value;
  }}
  logsums[row] = log(total) + maximum;
  double remainder = draws[row];
  int selected = largest;
  for (int a=0; a<{alternatives}; ++a) {{
    double probability = weight[a] / total;
    probability = probability < 0.0 ? 0.0 : (probability > 1.0 ? 1.0 : probability);
    remainder -= probability;
    if (remainder <= 0.0) {{ selected = a; break; }}
  }}
  choices[row] = selected;
}}
'''


@lru_cache(maxsize=8)
def _kernel(
    expressions: tuple[str, ...],
    coefficients: tuple[tuple[float, ...], ...],
    columns: tuple[str, ...],
):
    cp = _cupy()
    return cp.RawKernel(
        _source(expressions, coefficients, columns),
        "fused_fixed_mnl",
        options=("--std=c++11",),
    )


class FusedFixedMnlCudaModel:
    """Evaluate expressions, utilities, probabilities, and choice in one kernel.

    ``auto_ownership`` is a deliberately explicit dynamic column because the
    public mandatory-tour-frequency model consumes the immediately preceding
    auto-ownership result. All other columns are packed once into resident
    float64 state.
    """

    def __init__(
        self,
        expressions: Sequence[str],
        coefficients: Any,
        columns: Sequence[str],
    ):
        matrix = np.ascontiguousarray(coefficients, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(expressions):
            raise ValueError("fused MNL coefficient shape does not match expressions")
        if matrix.shape[1] < 2:
            raise ValueError("fused MNL requires at least two alternatives")
        self.expressions = tuple(str(value) for value in expressions)
        self.coefficients = tuple(tuple(float(value) for value in row) for row in matrix)
        self.columns = tuple(str(value) for value in columns)
        if "auto_ownership" in self.columns:
            raise ValueError("auto_ownership is the dedicated dynamic fused-MNL column")
        self.kernel = _kernel(self.expressions, self.coefficients, self.columns)

    def choose(
        self,
        static_values: Any,
        auto_ownership: Any,
        draws: Any,
        *,
        return_device: bool = True,
    ) -> ChoiceResult:
        inputs = (static_values, auto_ownership, draws)
        if any(not _is_cuda_array(value) for value in inputs):
            raise GpuOnlyViolation("fused MNL inputs must reside on the GPU")
        cp = _cupy()
        values = cp.ascontiguousarray(static_values, dtype=cp.float64)
        dynamic = cp.ascontiguousarray(auto_ownership, dtype=cp.float64)
        uniforms = cp.ascontiguousarray(draws, dtype=cp.float64)
        n = int(values.shape[0])
        if values.shape != (n, len(self.columns)):
            raise ValueError("fused MNL static input shape differs from its schema")
        if dynamic.shape != (n,) or uniforms.shape != (n,):
            raise ValueError("fused MNL dynamic inputs must match chooser rows")
        choices = cp.empty(n, dtype=cp.int32)
        logsums = cp.empty(n, dtype=cp.float64)
        threads = 256
        self.kernel(
            ((n + threads - 1) // threads,),
            (threads,),
            (values, dynamic, uniforms, np.int32(n), choices, logsums),
        )
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))
