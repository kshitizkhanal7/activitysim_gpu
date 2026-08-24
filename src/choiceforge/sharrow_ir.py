"""Strict, portable IR and CPU reference for Sharrow-style utilities.

The IR is data-only and canonical JSON serializable. The CPU evaluator is the
semantic oracle for future generated backends: it evaluates expression trees
in source order, casts completed features once, and accumulates utility terms
with separate, ordered IEEE-754 multiply and add operations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .activitysim_expression import ExpressionUnsupported, parse_activitysim_expression


_BINARY = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
    ast.BitAnd: "and", ast.BitOr: "or",
}
_COMPARE = {
    ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
    ast.Gt: "gt", ast.GtE: "ge",
}
_NUMERIC_POLICY = {
    "expression_dtype": "float64",
    "feature_storage_dtype": "float32",
    "coefficient_dtype": "float32",
    "utility_dtype": "float32",
    "rounding": "ieee754-nearest-even",
    "allow_fastmath": False,
    "ordered_terms": True,
    "contract_multiply_add": False,
    "flush_subnormals": True,
    "nonfinite": "preserve",
}


@dataclass(frozen=True)
class StrictCpuResult:
    """Complete, inspectable result of one strict CPU utility evaluation."""

    term_labels: tuple[str, ...]
    expressions: tuple[str, ...]
    alternative_names: tuple[str, ...]
    features: np.ndarray
    coefficients: np.ndarray
    utilities: np.ndarray
    ir_sha256: str
    numeric_policy: Mapping[str, Any]


def expression_ir(expression: str) -> dict:
    """Compile one reviewed expression to a typed, source-order AST."""
    return _node(parse_activitysim_expression(expression).body)


def specification_ir(spec) -> dict:
    """Compile a CSV-style specification DataFrame into a strict IR document."""
    if "Expression" not in spec.columns:
        raise ExpressionUnsupported("spec needs an Expression column")
    alternatives = [c for c in spec.columns if c not in {"Label", "Description", "Expression"}]
    terms = []
    for position, (_, row) in enumerate(spec.iterrows()):
        expression = str(row["Expression"]).strip()
        if not expression or expression.lower() == "nan":
            continue
        terms.append({
            "position": position,
            "label": str(row.get("Label", f"expression_{position}")),
            "expression": expression,
            "tree": expression_ir(expression),
            "coefficients": {alt: _coefficient(row[alt]) for alt in alternatives},
        })
    document = {
        "ir_version": 3,
        "numeric_policy": dict(_NUMERIC_POLICY),
        "alternatives": alternatives,
        "terms": terms,
    }
    document["sha256"] = ir_sha256(document)
    return document


def ir_sha256(document: dict) -> str:
    """Hash the IR excluding its self-referential hash field."""
    payload = {k: v for k, v in document.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_ir(document: dict, filename) -> None:
    with open(filename, "w", encoding="utf8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def evaluate_ir(tree: dict, environment, xp):
    """Generic source-order evaluator for an IR tree (NumPy or CuPy arrays).

    This compatibility evaluator follows the input array namespace's native
    promotion rules. :func:`evaluate_strict_cpu` is the normative evaluator.
    """
    op = tree["op"]
    if op == "const":
        return tree["value"]
    if op == "name":
        return environment[tree["name"]]
    if op == "column":
        return environment["df"][tree["name"]]
    if op == "skim":
        key = evaluate_ir(tree["key"], environment, xp)
        return environment[tree["direction"]][key]
    if op in {"neg", "pos", "not"}:
        value = evaluate_ir(tree["arg"], environment, xp)
        return -value if op == "neg" else +value if op == "pos" else ~value
    if op in {"add", "sub", "mul", "div", "and", "or"}:
        if "args" in tree:
            values = [evaluate_ir(item, environment, xp) for item in tree["args"]]
            result = values[0]
            for value in values[1:]:
                result = result & value if op == "and" else result | value
            return result
        a = evaluate_ir(tree["left"], environment, xp)
        b = evaluate_ir(tree["right"], environment, xp)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div":
            return a / b
        if op == "and":
            return a & b
        return a | b
    if op == "clip":
        lower = tree["keywords"].get("lower")
        upper = tree["keywords"].get("upper")
        if tree["args"]:
            lower = tree["args"][0]
        return xp.clip(
            evaluate_ir(tree["value"], environment, xp),
            None if lower is None else evaluate_ir(lower, environment, xp),
            None if upper is None else evaluate_ir(upper, environment, xp),
        )
    if op == "maximum":
        return xp.maximum(*(evaluate_ir(x, environment, xp) for x in tree["args"]))
    if op == "compare":
        left = evaluate_ir(tree["left"], environment, xp)
        result = True
        for operator_name, right_tree in zip(tree["operators"], tree["rights"]):
            right = evaluate_ir(right_tree, environment, xp)
            comparison = {
                "eq": left == right, "ne": left != right, "lt": left < right,
                "le": left <= right, "gt": left > right, "ge": left >= right,
            }[operator_name]
            result = result & comparison
            left = right
        return result
    raise ExpressionUnsupported(f"IR execution for {op!r} is not implemented")


def evaluate_strict_cpu(
    document: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    rows: int | None = None,
    coefficient_environment: Mapping[str, Any] | None = None,
    expression_dtype: str = "float64",
) -> StrictCpuResult:
    """Evaluate a strict IR document using its explicit numeric contract.

    Numerical leaves and intermediate arithmetic use float64. A completed
    expression is rounded once into float32 feature storage. Coefficients are
    rounded into float32, then every utility term is evaluated as a separate
    float32 multiply followed by a separate float32 add in source order. No
    matrix multiplication, parallel reduction, reassociation, or FMA is used.
    """
    _validate_document(document)
    rows = _infer_rows(environment) if rows is None else int(rows)
    if rows < 0:
        raise ValueError("rows must be nonnegative")
    terms = document["terms"]
    alternatives = tuple(document["alternatives"])
    if expression_dtype not in {"float32", "float64"}:
        raise ValueError("expression_dtype must be 'float32' or 'float64'")
    expression_np_dtype = np.dtype(expression_dtype)
    features = np.empty((rows, len(terms)), dtype=np.float32)
    with np.errstate(all="ignore"):
        for column, term in enumerate(terms):
            value = _strict_node(
                term["tree"], environment, expression_np_dtype
            )
            features[:, column] = _row_array(
                value, rows, np.float32, expression_np_dtype
            )
    coefficients = _resolved_coefficients(
        document, coefficient_environment or {}, dtype=np.float32
    )
    utilities = ordered_float32_utilities(features, coefficients)
    return StrictCpuResult(
        term_labels=tuple(term["label"] for term in terms),
        expressions=tuple(term["expression"] for term in terms),
        alternative_names=alternatives,
        features=features,
        coefficients=coefficients,
        utilities=utilities,
        ir_sha256=document["sha256"],
        numeric_policy={
            **document["numeric_policy"],
            "expression_dtype": expression_dtype,
        },
    )


def ordered_float32_utilities(features, coefficients) -> np.ndarray:
    """Source-ordered float32 utility accumulation with no FMA contraction."""
    values = np.ascontiguousarray(features, dtype=np.float32)
    weights = np.ascontiguousarray(coefficients, dtype=np.float32)
    if values.ndim != 2 or weights.ndim != 2 or values.shape[1] != weights.shape[0]:
        raise ValueError("features and coefficients need compatible 2D shapes")
    result = np.zeros((values.shape[0], weights.shape[1]), dtype=np.float32)
    product = np.empty_like(result)
    for term in range(values.shape[1]):
        np.multiply(values[:, term, None], weights[term, None, :], out=product)
        _flush_subnormals_f32(product)
        np.add(result, product, out=result)
        _flush_subnormals_f32(result)
    return result


def compare_strict_to_sharrow(
    strict: StrictCpuResult,
    sharrow_features,
    sharrow_utilities,
    *,
    row_labels=None,
    trace_label: str | None = None,
) -> dict:
    """Build a machine-readable, term-first comparison with Sharrow.

    Exact equality is the gate. The report also recomputes utilities from the
    Sharrow feature matrix using the strict ordered accumulator. This splits
    expression-policy differences from reduction-policy differences and makes
    every mismatching cell belong to an explicit diagnostic category.
    """
    features = np.asarray(sharrow_features, dtype=np.float32)
    utilities = np.asarray(sharrow_utilities, dtype=np.float32)
    if features.shape != strict.features.shape:
        raise ValueError(
            f"Sharrow feature shape {features.shape} != strict shape {strict.features.shape}"
        )
    if utilities.shape != strict.utilities.shape:
        raise ValueError(
            f"Sharrow utility shape {utilities.shape} != strict shape {strict.utilities.shape}"
        )
    strict_from_sharrow = ordered_float32_utilities(features, strict.coefficients)
    feature_equal = _equal_mask(strict.features, features)
    utility_equal = _equal_mask(strict.utilities, utilities)
    accumulation_equal = _equal_mask(strict_from_sharrow, utilities)
    term_exact = feature_equal.all(axis=0)
    alternative_exact = utility_equal.all(axis=0)
    first_feature = _first_mismatch(feature_equal)
    first_utility = _first_mismatch(utility_equal)
    categories = {
        "expression_policy_cells": int((~feature_equal).sum()),
        "utility_cells_explained_by_expression_inputs": int(
            ((~utility_equal) & accumulation_equal).sum()
        ),
        "utility_cells_with_accumulation_policy_difference": int(
            ((~utility_equal) & (~accumulation_equal)).sum()
        ),
    }
    return {
        "schema_version": 1,
        "trace_label": trace_label,
        "ir_sha256": strict.ir_sha256,
        "numeric_policy": dict(strict.numeric_policy),
        "rows": int(features.shape[0]),
        "terms": int(features.shape[1]),
        "alternatives": int(utilities.shape[1]),
        "exact_gate_passed": bool(feature_equal.all() and utility_equal.all()),
        "feature_comparison": {
            "exact_cells": int(feature_equal.sum()),
            "total_cells": int(feature_equal.size),
            "exact_terms": int(term_exact.sum()),
            "divergent_terms": int((~term_exact).sum()),
            "max_abs": _max_abs(strict.features, features),
            "first_divergence": _feature_detail(
                first_feature, strict, features, row_labels
            ),
        },
        "utility_comparison": {
            "exact_cells": int(utility_equal.sum()),
            "total_cells": int(utility_equal.size),
            "exact_alternatives": int(alternative_exact.sum()),
            "divergent_alternatives": int((~alternative_exact).sum()),
            "max_abs": _max_abs(strict.utilities, utilities),
            "strict_accumulator_from_sharrow_features_max_abs": _max_abs(
                strict_from_sharrow, utilities
            ),
            "first_divergence": _utility_detail(
                first_utility, strict, utilities, strict_from_sharrow, row_labels
            ),
        },
        "classification": categories,
        "interpretation": (
            "exact strict-policy agreement"
            if feature_equal.all() and utility_equal.all()
            else "Sharrow is observational evidence, not the strict oracle; all mismatches are classified as expression-policy and/or ordered-accumulation-policy differences"
        ),
    }


def write_comparison_report(report: Mapping[str, Any], filename) -> None:
    """Write one deterministic strict/Sharrow comparison report."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_document(document: Mapping[str, Any]) -> None:
    if document.get("ir_version") != 3:
        raise ValueError("strict CPU evaluator requires IR version 3")
    if document.get("numeric_policy") != _NUMERIC_POLICY:
        raise ValueError("IR numeric policy is unsupported or incomplete")
    if document.get("sha256") != ir_sha256(dict(document)):
        raise ValueError("IR SHA-256 does not match its contents")
    alternatives = document.get("alternatives")
    terms = document.get("terms")
    if not alternatives or len(set(alternatives)) != len(alternatives):
        raise ValueError("IR alternatives must be unique and nonempty")
    if not isinstance(terms, list):
        raise ValueError("IR terms must be a list")


def _strict_node(
    tree: Mapping[str, Any], environment: Mapping[str, Any],
    expression_dtype=np.dtype(np.float64),
):
    op = tree["op"]
    if op == "const":
        return _strict_leaf(tree["value"], expression_dtype)
    if op == "name":
        return _strict_leaf(environment[tree["name"]], expression_dtype)
    if op == "column":
        return _strict_leaf(environment["df"][tree["name"]], expression_dtype)
    if op == "skim":
        key = tree["key"]["value"]
        return _strict_leaf(environment[tree["direction"]][key], expression_dtype)
    if op in {"neg", "pos", "not"}:
        value = _strict_node(tree["arg"], environment, expression_dtype)
        if op == "not":
            return ~value
        value = _numeric(value, expression_dtype)
        return -value if op == "neg" else +value
    if op in {"add", "sub", "mul", "div", "and", "or"}:
        if "args" in tree:
            values = [
                _strict_node(item, environment, expression_dtype)
                for item in tree["args"]
            ]
            result = values[0]
            for value in values[1:]:
                result = result & value if op == "and" else result | value
            return result
        left = _strict_node(tree["left"], environment, expression_dtype)
        right = _strict_node(tree["right"], environment, expression_dtype)
        if op == "add":
            return np.add(_numeric(left, expression_dtype), _numeric(right, expression_dtype))
        if op == "sub":
            return np.subtract(_numeric(left, expression_dtype), _numeric(right, expression_dtype))
        if op == "mul":
            return np.multiply(_numeric(left, expression_dtype), _numeric(right, expression_dtype))
        if op == "div":
            return np.divide(_numeric(left, expression_dtype), _numeric(right, expression_dtype))
        if op == "and":
            return np.bitwise_and(left, right)
        return np.bitwise_or(left, right)
    if op == "clip":
        lower = tree["keywords"].get("lower")
        upper = tree["keywords"].get("upper")
        if tree["args"]:
            lower = tree["args"][0]
        return np.clip(
            _numeric(_strict_node(tree["value"], environment, expression_dtype), expression_dtype),
            None if lower is None else _numeric(_strict_node(lower, environment, expression_dtype), expression_dtype),
            None if upper is None else _numeric(_strict_node(upper, environment, expression_dtype), expression_dtype),
        )
    if op == "maximum":
        values = [
            _strict_node(item, environment, expression_dtype)
            for item in tree["args"]
        ]
        result = _numeric(values[0], expression_dtype)
        for value in values[1:]:
            result = np.maximum(result, _numeric(value, expression_dtype))
        return result
    if op == "compare":
        left = _strict_node(tree["left"], environment, expression_dtype)
        result = True
        for operator_name, right_tree in zip(tree["operators"], tree["rights"]):
            right = _strict_node(right_tree, environment, expression_dtype)
            comparison = {
                "eq": np.equal, "ne": np.not_equal, "lt": np.less,
                "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
            }[operator_name](
                _numeric(left, expression_dtype),
                _numeric(right, expression_dtype),
            )
            result = np.bitwise_and(result, comparison)
            left = right
        return result
    raise ExpressionUnsupported(f"strict IR execution for {op!r} is not implemented")


def _strict_leaf(value, expression_dtype=np.dtype(np.float64)):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy(copy=False)
    array = np.asarray(value)
    if array.dtype.kind in "ufc":
        return array.astype(expression_dtype, copy=False)
    if array.ndim == 0:
        return array.item()
    return array


def _numeric64(value):
    """Convert a value at an arithmetic operation boundary, not at lookup.

    Keeping integer and Boolean leaves intact is required for bitwise masks;
    the explicit conversion here still makes every arithmetic operation use
    the policy's float64 expression dtype.
    """
    return np.asarray(value, dtype=np.float64)


def _numeric(value, expression_dtype):
    return np.asarray(value, dtype=expression_dtype)


def _infer_rows(environment: Mapping[str, Any]) -> int:
    def candidates(value):
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from candidates(nested)
        else:
            shape = getattr(value, "shape", ())
            if shape:
                yield int(shape[0])

    for count in candidates(environment):
        return count
    raise ValueError("environment needs at least one row-array value")


def _row_array(
    value, rows: int, dtype, expression_dtype=np.dtype(np.float64)
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(rows, array.item())
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"expression produced shape {array.shape}, expected ({rows},)")
    # The strict contract routes every completed numeric expression through
    # float64 before feature storage, including bare integer/Boolean terms.
    result = array.astype(expression_dtype, copy=False).astype(dtype, copy=False)
    if np.dtype(dtype) == np.dtype(np.float32):
        _flush_subnormals_f32(result)
    return result


def _resolved_coefficients(document, symbols, dtype) -> np.ndarray:
    alternatives = document["alternatives"]
    result = np.empty((len(document["terms"]), len(alternatives)), dtype=dtype)
    for term_index, term in enumerate(document["terms"]):
        for alternative_index, alternative in enumerate(alternatives):
            value = term["coefficients"][alternative]
            if isinstance(value, Mapping) and "symbol" in value:
                symbol = value["symbol"]
                if symbol not in symbols:
                    raise ValueError(f"unresolved coefficient symbol {symbol!r}")
                value = symbols[symbol]
            try:
                result[term_index, alternative_index] = value
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"coefficient for term {term['label']!r}, alternative {alternative!r} is not numeric"
                ) from err
    if not np.isfinite(result).all():
        raise ValueError("strict coefficients must be finite")
    if np.dtype(dtype) == np.dtype(np.float32):
        _flush_subnormals_f32(result)
    return result


def _flush_subnormals_f32(values) -> None:
    """Apply the cross-device FTZ rule while preserving signed zero."""
    mask = (values != 0) & (np.abs(values) < np.finfo(np.float32).tiny)
    if np.any(mask):
        values[mask] = np.copysign(np.float32(0), values[mask])


def _equal_mask(left, right):
    return np.equal(left, right) | (np.isnan(left) & np.isnan(right))


def _first_mismatch(equal_mask):
    positions = np.argwhere(~equal_mask)
    return None if not len(positions) else tuple(int(x) for x in positions[0])


def _max_abs(left, right) -> float:
    with np.errstate(invalid="ignore"):
        delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    finite = delta[np.isfinite(delta)]
    return 0.0 if not len(finite) else float(finite.max())


def _row_label(row, row_labels):
    if row_labels is None:
        return row
    value = np.asarray(row_labels)[row]
    return value.item() if hasattr(value, "item") else value


def _feature_detail(position, strict, sharrow, row_labels):
    if position is None:
        return None
    row, term = position
    return {
        "stage": "completed_expression_cast_to_float32",
        "row_position": row,
        "row_label": _row_label(row, row_labels),
        "term_position": term,
        "term_label": strict.term_labels[term],
        "expression": strict.expressions[term],
        "strict": float(strict.features[row, term]),
        "sharrow": float(sharrow[row, term]),
        "abs_difference": float(abs(strict.features[row, term] - sharrow[row, term])),
        "explanation": "Sharrow's compiled expression does not implement the declared strict float64-expression then float32-feature policy for this cell",
    }


def _utility_detail(position, strict, sharrow, strict_from_sharrow, row_labels):
    if position is None:
        return None
    row, alternative = position
    via_sharrow_features = strict_from_sharrow[row, alternative]
    accumulation_differs = not _equal_mask(
        np.asarray(via_sharrow_features), np.asarray(sharrow[row, alternative])
    ).item()
    return {
        "stage": "ordered_float32_utility_accumulation" if accumulation_differs else "expression_inputs",
        "row_position": row,
        "row_label": _row_label(row, row_labels),
        "alternative_position": alternative,
        "alternative": strict.alternative_names[alternative],
        "strict": float(strict.utilities[row, alternative]),
        "strict_from_sharrow_features": float(via_sharrow_features),
        "sharrow": float(sharrow[row, alternative]),
        "abs_difference": float(abs(strict.utilities[row, alternative] - sharrow[row, alternative])),
        "explanation": (
            "Sharrow's dot reduction does not implement separate source-ordered float32 multiply/add operations"
            if accumulation_differs
            else "The utility difference is fully attributable to earlier expression-policy inputs"
        ),
    }


def _coefficient(value):
    if value is None or str(value).strip() in {"", "nan", "NaN"}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return {"symbol": str(value)}


def _node(node):
    if isinstance(node, ast.Constant):
        return {"op": "const", "value": node.value}
    if isinstance(node, ast.Name):
        return {"op": "name", "name": node.id}
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            return {"op": "column", "name": node.attr}
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            return {"op": "function", "name": f"np.{node.attr}"}
        raise ExpressionUnsupported(f"unsupported attribute {ast.unparse(node)!r}")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {
            "od_skims", "odt_skims", "dot_skims", "odr_skims", "dor_skims",
            "od_skims_reverse",
        }:
            return {"op": "skim", "direction": node.value.id, "key": _node(node.slice)}
        raise ExpressionUnsupported(f"unsupported subscript {ast.unparse(node)!r}")
    if isinstance(node, ast.UnaryOp):
        operation = {ast.USub: "neg", ast.UAdd: "pos", ast.Invert: "not"}.get(type(node.op))
        if operation:
            return {"op": operation, "arg": _node(node.operand)}
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return {
            "op": _BINARY[type(node.op)], "left": _node(node.left),
            "right": _node(node.right),
        }
    if isinstance(node, ast.BoolOp):
        operation = "and" if isinstance(node.op, ast.And) else "or" if isinstance(node.op, ast.Or) else None
        if operation:
            return {"op": operation, "args": [_node(v) for v in node.values]}
    if isinstance(node, ast.Compare):
        return {
            "op": "compare", "left": _node(node.left),
            "operators": [_COMPARE[type(op)] for op in node.ops],
            "rights": [_node(x) for x in node.comparators],
        }
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "od_skims"
            and node.func.attr in {"reverse", "max"}
            and len(node.args) == 1
            and not node.keywords
        ):
            forward = {"op": "skim", "direction": "od_skims", "key": _node(node.args[0])}
            reverse = {
                "op": "skim",
                "direction": "od_skims_reverse",
                "key": _node(node.args[0]),
            }
            if node.func.attr == "reverse":
                return reverse
            return {"op": "maximum", "args": [forward, reverse]}
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clip":
            return {
                "op": "clip", "value": _node(node.func.value),
                "args": [_node(a) for a in node.args],
                "keywords": {k.arg: _node(k.value) for k in node.keywords},
            }
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "maximum"
        ):
            return {"op": "maximum", "args": [_node(a) for a in node.args]}
    raise ExpressionUnsupported(f"unsupported IR AST node {ast.dump(node)}")
