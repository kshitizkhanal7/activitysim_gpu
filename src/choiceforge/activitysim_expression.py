"""Strict evaluator for the numeric subset of ActivitySim utility expressions.

This is a deliberately small AST interpreter, not a replacement for Python
``eval``.  It covers the expression forms found in the Prototype MTC Extended
``trip_mode_choice.csv`` and fails closed on any new syntax.  The same parsed
tree runs with NumPy (reference) or CuPy (device) arrays.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import operator

import numpy as np


class ExpressionUnsupported(ValueError):
    """The expression is outside the reviewed GPU-lowering subset."""


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
}
_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}


@lru_cache(maxsize=1024)
def parse_activitysim_expression(expression: str) -> ast.Expression:
    """Parse an ActivitySim expression after removing its local-variable ``@``.

    ActivitySim uses ``@name`` merely to select a local namespace.  In the
    lowered environment all names share one explicit mapping, so stripping the
    marker preserves the numeric meaning while avoiding Python execution.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionUnsupported("expression must be a nonempty string")
    try:
        return ast.parse(expression.replace("@", ""), mode="eval")
    except SyntaxError as err:
        raise ExpressionUnsupported(f"cannot parse expression {expression!r}") from err


def evaluate_activitysim_expression(expression: str, environment, xp):
    """Evaluate one reviewed expression against mapping-like data and ``xp``.

    ``environment`` exposes chooser columns under ``df``, skim mappings under
    ``od_skims``/``odt_skims``, and coefficient/local values by their name.
    Values can be scalars or arrays belonging to NumPy or CuPy.
    """
    return _evaluate(parse_activitysim_expression(expression).body, environment, xp)


def lower_activitysim_utility_spec(spec, environment, xp=np, dtype=None):
    """Lower a coefficient-resolved ActivitySim spec into numeric features.

    ``spec`` must have ActivitySim's ``Expression`` column and numeric
    alternative columns; comment/blank expression rows are ignored.  This
    intentionally rejects unresolved coefficient strings.  The returned
    ``LoweredDestinationUtility`` and feature matrix can be sent directly to
    the CUDA utility pipeline after a CPU equivalence comparison.
    """
    from .destination_utility import LoweredDestinationUtility

    if "Expression" not in spec.columns:
        raise ExpressionUnsupported("spec needs an Expression column")
    alternatives = tuple(column for column in spec.columns if column not in {"Label", "Description", "Expression"})
    if not alternatives:
        raise ExpressionUnsupported("spec has no alternative columns")
    active = spec[spec["Expression"].notna() & spec["Expression"].astype(str).str.strip().ne("")]
    expressions = tuple(active["Expression"].astype(str))
    labels = tuple(active["Label"].astype(str)) if "Label" in active else tuple(f"expression_{i}" for i in range(len(active)))
    if len(set(labels)) != len(labels):
        raise ExpressionUnsupported("spec labels must be unique after comments are removed")
    try:
        coefficients = active.loc[:, alternatives].replace("", 0).fillna(0).astype(float).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ExpressionUnsupported("spec contains unresolved nonnumeric coefficients") from err
    dtype = dtype or xp.float64
    values = [_as_row_array(evaluate_activitysim_expression(expression, environment, xp), environment, xp, dtype) for expression in expressions]
    rows = int(values[0].shape[0]) if values else _environment_rows(environment)
    if any(int(value.shape[0]) != rows for value in values):
        raise ExpressionUnsupported("expressions produced inconsistent row counts")
    feature_values = xp.column_stack(values) if values else xp.empty((rows, 0), dtype=xp.float64)
    return LoweredDestinationUtility(
        labels, alternatives, coefficients, np.zeros(len(alternatives)),
        compute_dtype=np.dtype(dtype).name,
    ), feature_values


def _lookup(value, key):
    if isinstance(value, dict):
        return value[key]
    try:
        return value[key]
    except (KeyError, TypeError, IndexError):
        return getattr(value, key)


def _environment_rows(environment):
    for value in environment.values():
        if isinstance(value, dict):
            for nested in value.values():
                shape = getattr(nested, "shape", ())
                if shape:
                    return int(shape[0])
        else:
            shape = getattr(value, "shape", ())
            if shape:
                return int(shape[0])
    raise ExpressionUnsupported("environment needs at least one row-array value")


def _as_row_array(value, environment, xp, dtype):
    shape = getattr(value, "shape", ())
    if shape:
        if len(shape) != 1:
            raise ExpressionUnsupported("expressions must yield one value per row")
        # A reviewed expression can combine a device chooser column with a
        # NumPy-valued global constant.  Normalize every vector through the
        # selected array namespace before column_stack; CuPy intentionally
        # rejects mixed host/device inputs.
        return xp.asarray(value, dtype=dtype)
    return xp.full(_environment_rows(environment), value, dtype=dtype)


def _evaluate(node, env, xp):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExpressionUnsupported(f"unknown name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            return ("numpy_function", node.attr)
        return _lookup(_evaluate(node.value, env, xp), node.attr)
    if isinstance(node, ast.Subscript):
        key = _evaluate(node.slice, env, xp)
        return _lookup(_evaluate(node.value, env, xp), key)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, env, xp)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Invert):
            return ~value
        raise ExpressionUnsupported(f"unary {type(node.op).__name__} is unsupported")
    if isinstance(node, ast.BinOp):
        operation = _BINOPS.get(type(node.op))
        if operation is None:
            raise ExpressionUnsupported(f"binary {type(node.op).__name__} is unsupported")
        return operation(_evaluate(node.left, env, xp), _evaluate(node.right, env, xp))
    if isinstance(node, ast.BoolOp):
        operation = operator.and_ if isinstance(node.op, ast.And) else operator.or_ if isinstance(node.op, ast.Or) else None
        if operation is None:
            raise ExpressionUnsupported(f"boolean {type(node.op).__name__} is unsupported")
        values = [_evaluate(item, env, xp) for item in node.values]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, env, xp)
        result = True
        for operation_node, right_node in zip(node.ops, node.comparators):
            operation = _CMPOPS.get(type(operation_node))
            if operation is None:
                raise ExpressionUnsupported(f"comparison {type(operation_node).__name__} is unsupported")
            right = _evaluate(right_node, env, xp)
            result = operator.and_(result, operation(left, right))
            left = right
        return result
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clip":
            value = _evaluate(node.func.value, env, xp)
            if len(node.args) > 1 or any(k.arg not in {"lower", "upper"} for k in node.keywords):
                raise ExpressionUnsupported("only clip(lower=, upper=) is supported")
            lower = _evaluate(node.args[0], env, xp) if node.args else None
            upper = None
            for keyword in node.keywords:
                if keyword.arg == "lower": lower = _evaluate(keyword.value, env, xp)
                if keyword.arg == "upper": upper = _evaluate(keyword.value, env, xp)
            return xp.clip(value, lower, upper)
        func = _evaluate(node.func, env, xp)
        if func == ("numpy_function", "maximum") and len(node.args) == 2 and not node.keywords:
            return xp.maximum(_evaluate(node.args[0], env, xp), _evaluate(node.args[1], env, xp))
        raise ExpressionUnsupported("only np.maximum and array.clip are supported")
    raise ExpressionUnsupported(f"AST node {type(node).__name__} is unsupported")
