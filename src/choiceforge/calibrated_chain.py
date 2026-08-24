"""Calibrated ActivitySim MNL building blocks for the Phase 19 replay chain.

This module intentionally implements only the reviewed semantics used by the
public Prototype MTC Extended auto-ownership and mandatory-tour-frequency
specifications.  Configuration parsing remains CPU control-plane work.  Once
input tables and numeric coefficients are uploaded, joins, expression features,
utilities, probabilities, random draws, and choices can remain device resident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .activitysim_expression import evaluate_activitysim_expression
from .cuda_backend import CudaChoiceBackend, _cupy
from .gpu_native import GpuOnlyViolation, _is_cuda_array


@dataclass(frozen=True)
class ResolvedMnlSpec:
    """A coefficient-resolved, immutable ActivitySim fixed-alternative MNL."""

    name: str
    expressions: tuple[str, ...]
    labels: tuple[str, ...]
    alternatives: tuple[str, ...]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        expected = (len(self.expressions), len(self.alternatives))
        if not self.expressions or coefficients.shape != expected:
            raise ValueError(f"coefficient matrix must have shape {expected}")
        if len(self.labels) != len(self.expressions):
            raise ValueError("each expression requires one audit label")
        if len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("alternative names must be unique")
        if not np.isfinite(coefficients).all():
            raise ValueError("resolved coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)


def resolve_activitysim_mnl_spec(
    name: str,
    spec_path: str | Path,
    coefficients_path: str | Path,
) -> ResolvedMnlSpec:
    """Resolve an ActivitySim coefficient-symbol CSV with float32 semantics.

    ActivitySim casts each resolved specification column to float32 before its
    float64 matrix product.  Quantizing here and then storing those exact values
    in float64 is required for faithful replay.
    """

    import pandas as pd

    frame = pd.read_csv(spec_path, comment="#")
    if "Expression" not in frame:
        raise ValueError("ActivitySim spec needs an Expression column")
    alternatives = tuple(
        column
        for column in frame.columns
        if column not in {"Label", "Description", "Expression"}
    )
    coefficient_frame = pd.read_csv(coefficients_path)
    required = {"coefficient_name", "value"}
    if not required.issubset(coefficient_frame.columns):
        raise ValueError("coefficient table needs coefficient_name and value")
    symbols = coefficient_frame.set_index("coefficient_name")["value"].to_dict()

    active = frame[frame["Expression"].notna()].copy()
    active = active[active["Expression"].astype(str).str.strip().ne("")]
    resolved_columns = []
    for alternative in alternatives:
        values = []
        for raw in active[alternative]:
            if pd.isna(raw) or str(raw).strip() == "":
                value = 0.0
            elif isinstance(raw, str) and raw.strip() in symbols:
                value = symbols[raw.strip()]
            else:
                try:
                    value = float(raw)
                except (TypeError, ValueError) as err:
                    raise ValueError(f"unresolved coefficient {raw!r}") from err
            values.append(value)
        resolved_columns.append(np.asarray(values, dtype=np.float32))
    coefficients = np.column_stack(resolved_columns).astype(np.float64)
    keep = np.any(coefficients != 0.0, axis=1)
    labels = (
        tuple(active.loc[keep, "Label"].astype(str))
        if "Label" in active
        else tuple(f"expression_{i}" for i in np.flatnonzero(keep))
    )
    return ResolvedMnlSpec(
        name=name,
        expressions=tuple(active.loc[keep, "Expression"].astype(str)),
        labels=labels,
        alternatives=alternatives,
        coefficients=coefficients[keep],
    )


def expression_environment(
    columns: Mapping[str, Any], constants: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Expose explicit chooser columns as both bare names and ``df`` fields."""

    environment = dict(columns)
    environment["df"] = columns
    if constants:
        environment.update(constants)
    return environment


def evaluate_mnl_features(
    spec: ResolvedMnlSpec,
    columns: Mapping[str, Any],
    xp: Any = np,
    constants: Mapping[str, Any] | None = None,
) -> Any:
    """Evaluate every reviewed spec expression in published row order."""

    environment = expression_environment(columns, constants)
    row_count = _row_count(columns)
    values = []
    for expression in spec.expressions:
        value = evaluate_activitysim_expression(expression, environment, xp)
        if getattr(value, "ndim", 0) == 0:
            value = xp.full(row_count, value, dtype=xp.float64)
        else:
            value = xp.asarray(value, dtype=xp.float64)
        if value.shape != (row_count,):
            raise ValueError(f"expression {expression!r} did not produce one value per chooser")
        values.append(value)
    return xp.ascontiguousarray(xp.column_stack(values), dtype=xp.float64)


def mnl_utilities(features: Any, coefficients: Any, xp: Any = np) -> Any:
    """Apply ActivitySim's float64 dense MNL utility boundary."""

    x = xp.asarray(features, dtype=xp.float64)
    beta = xp.asarray(coefficients, dtype=xp.float64)
    if x.ndim != 2 or beta.ndim != 2 or x.shape[1] != beta.shape[0]:
        raise ValueError("features and coefficients have incompatible shapes")
    return x @ beta


def mnl_probabilities(utilities: Any, xp: Any = np) -> tuple[Any, Any]:
    """Match ActivitySim's overflow-protected utility-to-probability policy."""

    values = xp.asarray(utilities, dtype=xp.float64)
    shifted = values - xp.max(values, axis=1, keepdims=True)
    weights = xp.exp(shifted)
    weights = xp.where(weights <= 1.0e-300, 0.0, weights)
    totals = xp.sum(weights, axis=1)
    probabilities = weights / totals[:, None]
    logsums = xp.log(totals) + xp.max(values, axis=1)
    return probabilities, logsums


def choice_from_probabilities_cpu(probabilities: Any, uniforms: Any) -> np.ndarray:
    """Independent ordered-subtraction oracle for ActivitySim ``choice_maker``."""

    probs = np.ascontiguousarray(probabilities, dtype=np.float64)
    draws = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    if probs.ndim != 2 or draws.shape != (probs.shape[0],):
        raise ValueError("probabilities must be (N,A) and uniforms must contain N values")
    choices = np.empty(probs.shape[0], dtype=np.int32)
    for row in range(probs.shape[0]):
        remainder = draws[row]
        selected = -1
        for alternative in range(probs.shape[1]):
            remainder -= probs[row, alternative]
            if remainder <= 0.0:
                selected = alternative
                break
        if selected < 0:
            selected = int(np.argmax(probs[row]))
        choices[row] = selected
    return choices


def choice_from_probabilities_gpu(probabilities: Any, uniforms: Any) -> Any:
    """Select alternatives on-device with ActivitySim's float64 traversal."""

    if not _is_cuda_array(probabilities) or not _is_cuda_array(uniforms):
        raise GpuOnlyViolation("probabilities and uniforms must reside on the GPU")
    return CudaChoiceBackend().choose_from_probabilities(
        probabilities, uniforms, return_device=True
    )


def gather_by_key_gpu(
    source_keys: Any,
    target_keys: Any,
    source_columns: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform a validated many-to-one key lookup without host modeled data."""

    cp = _cupy()
    if not _is_cuda_array(source_keys) or not _is_cuda_array(target_keys):
        raise GpuOnlyViolation("join keys must reside on the GPU")
    if any(not _is_cuda_array(value) for value in source_columns.values()):
        raise GpuOnlyViolation("join value columns must reside on the GPU")
    source = cp.ascontiguousarray(source_keys, dtype=cp.int64)
    target = cp.ascontiguousarray(target_keys, dtype=cp.int64)
    order = cp.argsort(source, kind="stable")
    sorted_keys = source[order]
    positions = cp.searchsorted(sorted_keys, target)
    in_bounds = positions < sorted_keys.size
    safe_positions = cp.minimum(positions, max(int(sorted_keys.size) - 1, 0))
    if sorted_keys.size == 0 or bool(cp.any(~in_bounds).item()) or bool(
        cp.any(sorted_keys[safe_positions] != target).item()
    ):
        raise KeyError("target key is missing from GPU lookup table")
    rows = order[safe_positions]
    return {name: value[rows] for name, value in source_columns.items()}


def key_rows_gpu(source_keys: Any, target_keys: Any) -> Any:
    """Compile a validated device row map for a repeatedly used key join.

    The returned CUDA vector can be retained by a resident runtime and reused
    as ``source_column[row_map]`` without sorting the same static keys in every
    scenario. Source keys must be unique so the mapping is unambiguous.
    """

    cp = _cupy()
    if not _is_cuda_array(source_keys) or not _is_cuda_array(target_keys):
        raise GpuOnlyViolation("join keys must reside on the GPU")
    source = cp.ascontiguousarray(source_keys, dtype=cp.int64)
    target = cp.ascontiguousarray(target_keys, dtype=cp.int64)
    if source.size == 0:
        if target.size:
            raise KeyError("target key is missing from GPU lookup table")
        return cp.empty(0, dtype=cp.int64)
    order = cp.argsort(source, kind="stable")
    sorted_keys = source[order]
    if sorted_keys.size > 1 and bool(cp.any(sorted_keys[1:] == sorted_keys[:-1]).item()):
        raise ValueError("source keys must be unique when compiling a GPU row map")
    positions = cp.searchsorted(sorted_keys, target)
    in_bounds = positions < sorted_keys.size
    safe_positions = cp.minimum(positions, sorted_keys.size - 1)
    if bool(cp.any(~in_bounds).item()) or bool(
        cp.any(sorted_keys[safe_positions] != target).item()
    ):
        raise KeyError("target key is missing from GPU lookup table")
    return cp.ascontiguousarray(order[safe_positions], dtype=cp.int64)


def _row_count(columns: Mapping[str, Any]) -> int:
    lengths = {int(value.shape[0]) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("chooser columns must have equal lengths")
    return next(iter(lengths))
