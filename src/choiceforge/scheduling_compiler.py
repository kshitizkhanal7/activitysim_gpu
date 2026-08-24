"""Compile a safe ActivitySim scheduling-expression subset to CPU and CUDA.

The compiler consumes the compact Phase 3 ABI: chooser values are stored once
per chooser, alternative attributes once per TDD alternative, and only truly
row-varying values are stored per feasible chooser-alternative pair.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import keyword
from typing import Any, Sequence

import numpy as np

from .api import ChoiceResult
from .cuda_backend import _cupy, _register_pip_cuda_dlls

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None

if numba is not None:
    @numba.njit(parallel=True, cache=True)
    def _ragged_to_dense_f32(utilities, offsets, width):
        dense = np.full((offsets.size - 1, width), -np.inf, dtype=np.float32)
        for chooser in numba.prange(offsets.size - 1):
            begin = offsets[chooser]
            count = offsets[chooser + 1] - begin
            for position in range(count):
                dense[chooser, position] = utilities[begin + position]
        return dense

    @numba.njit(parallel=True, cache=True)
    def _choose_normalized_f32(probabilities, draws):
        choices = np.empty(probabilities.shape[0], dtype=np.int32)
        for chooser in numba.prange(probabilities.shape[0]):
            remainder = draws[chooser]
            selected = probabilities.shape[1] - 1
            for position in range(probabilities.shape[1]):
                remainder -= probabilities[chooser, position]
                if remainder <= 0.0:
                    selected = position
                    break
            choices[chooser] = selected
        return choices


_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.BitAnd: "&&", ast.BitOr: "||",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=",
}


class _CudaExpression(ast.NodeVisitor):
    """Translate the deliberately small numeric/Boolean expression subset."""

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, (int, float)):
            return repr(node.value)
        raise ValueError(f"unsupported constant {node.value!r}")

    def visit_BinOp(self, node):
        operator = _BINOPS.get(type(node.op))
        if operator is None:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        return f"({self.visit(node.left)} {operator} {self.visit(node.right)})"

    def visit_BoolOp(self, node):
        operator = "&&" if isinstance(node.op, ast.And) else "||" if isinstance(node.op, ast.Or) else None
        if operator is None:
            raise ValueError(f"unsupported Boolean operator {type(node.op).__name__}")
        return "(" + f" {operator} ".join(self.visit(x) for x in node.values) + ")"

    def visit_Compare(self, node):
        pieces = []
        left = node.left
        for operator_node, right in zip(node.ops, node.comparators):
            operator = _CMPOPS.get(type(operator_node))
            if operator is None:
                raise ValueError(f"unsupported comparison {type(operator_node).__name__}")
            pieces.append(f"({self.visit(left)} {operator} {self.visit(right)})")
            left = right
        return "(" + " && ".join(pieces) + ")"

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            return f"(-{self.visit(node.operand)})"
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return f"(!{self.visit(node.operand)})"
        raise ValueError(f"unsupported unary operator {type(node.op).__name__}")

    def generic_visit(self, node):
        raise ValueError(f"unsupported syntax {type(node).__name__}")


def compile_cuda_expression(expression: str) -> str:
    return _CudaExpression().visit(ast.parse(expression, mode="eval").body)


class _ScalarBooleanTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.BitAnd):
            return ast.BoolOp(op=ast.And(), values=[node.left, node.right])
        if isinstance(node.op, ast.BitOr):
            return ast.BoolOp(op=ast.Or(), values=[node.left, node.right])
        return node


def compile_python_expression(expression: str) -> str:
    tree = _ScalarBooleanTransformer().visit(ast.parse(expression, mode="eval"))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body)


def _validate_names(expressions: Sequence[str], valid_names: set[str]) -> None:
    for expression in expressions:
        tree = ast.parse(expression, mode="eval")
        compile_cuda_expression(expression)  # validates every syntax node
        names = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
        unknown = names - valid_names
        if unknown:
            raise ValueError(f"unknown names in {expression!r}: {sorted(unknown)}")


@dataclass(frozen=True)
class SchedulingSchema:
    chooser_columns: tuple[str, ...]
    row_columns: tuple[str, ...]
    alternative_columns: tuple[str, ...]


_CUDA_RESERVED = {
    "auto", "bool", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "false", "float", "for", "goto",
    "if", "int", "long", "register", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "true", "typedef", "union", "unsigned", "void",
    "volatile", "while",
}


def _validate_schema(schema: SchedulingSchema) -> None:
    columns = schema.chooser_columns + schema.row_columns + schema.alternative_columns
    if len(columns) != len(set(columns)):
        raise ValueError("compact scheduling column names must be unique across scopes")
    invalid = [
        name for name in columns
        if not name.isidentifier() or keyword.iskeyword(name) or name in _CUDA_RESERVED
    ]
    if invalid:
        raise ValueError(f"invalid generated-source column names: {invalid}")


def _cuda_source(expressions, coefficients, schema: SchedulingSchema) -> str:
    _validate_schema(schema)
    valid = set(schema.chooser_columns + schema.row_columns + schema.alternative_columns)
    _validate_names(expressions, valid)
    assignments = []
    for i, name in enumerate(schema.chooser_columns):
        assignments.append(f"const float {name} = chooser_values[chooser * {len(schema.chooser_columns)} + {i}];")
    for i, name in enumerate(schema.row_columns):
        assignments.append(f"const float {name} = row_values[row * {len(schema.row_columns)} + {i}];")
    for i, name in enumerate(schema.alternative_columns):
        assignments.append(f"const float {name} = alternative_values[alt_id * {len(schema.alternative_columns)} + {i}];")
    sums = []
    for expression, coefficient in zip(expressions, coefficients):
        c_expr = compile_cuda_expression(expression)
        literal = f"{float(np.float32(coefficient)):.9g}"
        if "." not in literal and "e" not in literal.lower():
            literal += ".0"
        literal += "f"
        sums.append(f"acc = fmaf((float)({c_expr}), {literal}, acc);")

    body = "\n        ".join(assignments + ["float acc = 0.0f;"] + sums + ["utility = acc;"])
    return f'''extern "C" __global__
void compact_scheduling_choice(
 const float* chooser_values, const float* row_values,
 const float* alternative_values, const short* alternative_ids,
 const long long* offsets, const double* uniforms, int n_choosers,
 int* choices, float* logsums)
{{
 const int chooser = blockIdx.x; const int lane = threadIdx.x;
 if (chooser >= n_choosers) return;
 const long long begin = offsets[chooser];
 const int count = (int)(offsets[chooser + 1] - begin);
 extern __shared__ float shared[]; float* values = shared; float* scratch = shared + blockDim.x;
 const float neg_inf = -__int_as_float(0x7f800000); float utility = neg_inf;
 if (lane < count) {{
  const long long row = begin + lane; const int alt_id = (int)alternative_ids[row];
        {body}
 }}
 values[lane] = utility; scratch[lane] = utility; __syncthreads();
 for (int stride=blockDim.x/2; stride>0; stride>>=1) {{ if(lane<stride) scratch[lane]=fmaxf(scratch[lane],scratch[lane+stride]); __syncthreads(); }}
 const float row_max=scratch[0]; __syncthreads();
 const float weight=lane<count?expf(values[lane]-row_max):0.0f;
 values[lane]=weight; scratch[lane]=weight; __syncthreads();
 for (int stride=blockDim.x/2; stride>0; stride>>=1) {{ if(lane<stride) scratch[lane]+=scratch[lane+stride]; __syncthreads(); }}
 if(lane==0) {{ const float total=scratch[0]; double remainder=uniforms[chooser]; int selected=-1;
  for(int alt=0;alt<count;++alt) {{ const float probability=values[alt]/total; remainder-=(double)probability; if(selected<0 && remainder<=0.0) selected=alt; }}
  if(selected<0) selected=count-1; choices[chooser]=selected; logsums[chooser]=row_max+logf(total); }}
}}
'''


@lru_cache(maxsize=8)
def _raw_kernel(expressions, coefficients, schema):
    cp = _cupy()
    source = _cuda_source(expressions, coefficients, schema)
    return cp.RawKernel(source, "compact_scheduling_choice", options=("--std=c++11",))


def _threads_for_offsets(offsets) -> int:
    counts = np.diff(np.asarray(offsets))
    maximum = int(counts.max())
    if maximum < 1 or maximum > 1024:
        raise ValueError("each chooser must have 1 to 1,024 alternatives")
    return max(32, 1 << (maximum - 1).bit_length())


class CompiledCudaSchedulingModel:
    def __init__(self, expressions, coefficients, schema: SchedulingSchema):
        _register_pip_cuda_dlls()
        if not expressions or len(expressions) != len(coefficients):
            raise ValueError("expressions and coefficients must be non-empty and equal length")
        self.expressions = tuple(expressions)
        self.coefficients = tuple(float(np.float32(x)) for x in coefficients)
        self.schema = schema
        self.kernel = _raw_kernel(self.expressions, self.coefficients, schema)
        self._threads: int | None = None

    def choose(self, chooser_values, row_values, alternative_values, alternative_ids,
               offsets, uniforms, *, return_device=False):
        cp = _cupy()
        offsets_are_device = hasattr(offsets, "__cuda_array_interface__")
        if not offsets_are_device:
            host_offsets = np.asarray(offsets, dtype=np.int64)
            if host_offsets.ndim != 1 or host_offsets[0] != 0:
                raise ValueError("offsets must be a one-dimensional CSR pointer starting at zero")
            if self._threads is None:
                self._threads = _threads_for_offsets(host_offsets)
        chooser = cp.ascontiguousarray(cp.asarray(chooser_values, dtype=cp.float32))
        rows = cp.ascontiguousarray(cp.asarray(row_values, dtype=cp.float32))
        alternatives = cp.ascontiguousarray(cp.asarray(alternative_values, dtype=cp.float32))
        alt_ids = cp.ascontiguousarray(cp.asarray(alternative_ids, dtype=cp.int16))
        ptr = cp.ascontiguousarray(cp.asarray(offsets, dtype=cp.int64))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float64))
        n = int(chooser.shape[0])
        if ptr.shape != (n + 1,) or draws.shape != (n,) or alt_ids.shape != (rows.shape[0],):
            raise ValueError("compact scheduling inputs have incompatible shapes")
        if not offsets_are_device and int(host_offsets[-1]) != rows.shape[0]:
            raise ValueError("the final offset must equal the number of interaction rows")
        if self._threads is None:
            counts = ptr[1:] - ptr[:-1]
            maximum = int(cp.max(counts).item())
            self._threads = max(32, 1 << (maximum - 1).bit_length())
        threads = self._threads
        choices = cp.empty(n, dtype=cp.int32); logsums = cp.empty(n, dtype=cp.float32)
        self.kernel((n,), (threads,), (chooser, rows, alternatives, alt_ids, ptr, draws, np.int32(n), choices, logsums),
                    shared_mem=2 * threads * np.dtype(np.float32).itemsize)
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))


def _python_utility_source(expressions, coefficients, schema):
    _validate_schema(schema)
    valid = set(schema.chooser_columns + schema.row_columns + schema.alternative_columns)
    _validate_names(expressions, valid)
    lines = ["def calculate(chooser_values, row_values, alternative_values, alternative_ids, offsets):",
             "    out = np.empty(row_values.shape[0], dtype=np.float32)",
             "    for chooser in prange(chooser_values.shape[0]):"]
    for i, name in enumerate(schema.chooser_columns):
        lines.append(f"        {name} = chooser_values[chooser, {i}]")
    lines.extend(["        for row in range(offsets[chooser], offsets[chooser + 1]):",
                  "            alt_id = alternative_ids[row]"])
    for i, name in enumerate(schema.row_columns):
        lines.append(f"            {name} = row_values[row, {i}]")
    for i, name in enumerate(schema.alternative_columns):
        lines.append(f"            {name} = alternative_values[alt_id, {i}]")
    lines.append("            acc = np.float32(0.0)")
    for expression, coefficient in zip(expressions, coefficients):
        scalar_expression = compile_python_expression(expression)
        lines.append(f"            acc += np.float32(({scalar_expression}) * np.float32({float(np.float32(coefficient))!r}))")
    lines.extend(["            out[row] = acc", "    return out"])
    return "\n".join(lines)


@lru_cache(maxsize=8)
def _numba_utility_kernel(expressions, coefficients, schema):
    if numba is None:
        raise RuntimeError("Numba is required for the compiled CPU baseline")
    namespace = {"np": np, "prange": numba.prange}
    exec(_python_utility_source(expressions, coefficients, schema), namespace)
    return numba.njit(parallel=True)(namespace["calculate"])


class CompiledCpuSchedulingModel:
    def __init__(self, expressions, coefficients, schema: SchedulingSchema):
        if numba is None:
            raise RuntimeError("Numba is required for the compiled CPU baseline")
        if not expressions or len(expressions) != len(coefficients):
            raise ValueError("expressions and coefficients must be non-empty and equal length")
        self.expressions = tuple(expressions)
        self.coefficients = tuple(float(np.float32(x)) for x in coefficients)
        self.schema = schema
        self.kernel = _numba_utility_kernel(self.expressions, self.coefficients, schema)

    def utilities(self, chooser_values, row_values, alternative_values, alternative_ids, offsets):
        return self.kernel(
            np.ascontiguousarray(chooser_values, dtype=np.float32),
            np.ascontiguousarray(row_values, dtype=np.float32),
            np.ascontiguousarray(alternative_values, dtype=np.float32),
            np.ascontiguousarray(alternative_ids, dtype=np.int16),
            np.ascontiguousarray(offsets, dtype=np.int64),
        )

    def choose(self, chooser_values, row_values, alternative_values, alternative_ids, offsets, uniforms):
        utilities = self.utilities(chooser_values, row_values, alternative_values, alternative_ids, offsets)
        ptr = np.asarray(offsets, dtype=np.int64)
        dense = _ragged_to_dense_f32(utilities, ptr, np.asarray(alternative_values).shape[0])
        shifts = dense.max(axis=1, keepdims=True)
        dense -= shifts
        np.exp(dense, out=dense)
        np.putmask(dense, dense <= np.float32(1.0e-300), np.float32(0.0))
        totals = dense.sum(axis=1)
        logsums = np.log(totals) + shifts[:, 0]
        np.divide(dense, totals[:, None], out=dense)
        np.clip(dense, np.float32(0.0), np.float32(1.0), out=dense)
        choices = _choose_normalized_f32(
            dense, np.asarray(uniforms, dtype=np.float64)
        )
        return ChoiceResult(choices, logsums)
