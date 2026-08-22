"""NumPy correctness oracle for ChoiceForge kernels.

The reference is intentionally straightforward. It materializes utilities and
weights, so it is not intended to be the fastest CPU implementation. Its roles
are defining edge-case behavior and checking an optimized backend against a
small, independently readable implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .api import ChoiceResult


def _validated_inputs(
    utilities: ArrayLike,
    uniforms: ArrayLike,
    availability: ArrayLike | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    u = np.ascontiguousarray(utilities, dtype=np.float32)
    draws = np.ascontiguousarray(uniforms, dtype=np.float32)
    if u.ndim != 2:
        raise ValueError(f"utilities must be two-dimensional, got shape {u.shape}")
    if draws.shape != (u.shape[0],):
        raise ValueError(
            f"uniforms must have one value per chooser; expected {(u.shape[0],)}, "
            f"got {draws.shape}"
        )
    if np.any((draws < 0.0) | (draws >= 1.0) | ~np.isfinite(draws)):
        raise ValueError("uniforms must be finite values in the half-open interval [0, 1)")

    if availability is None:
        available = np.ones(u.shape, dtype=np.bool_)
    else:
        available = np.ascontiguousarray(availability, dtype=np.bool_)
        if available.shape != u.shape:
            raise ValueError(
                f"availability must match utilities shape {u.shape}, got {available.shape}"
            )

    # Non-finite utilities cannot define a valid MNL probability. Treating them
    # as unavailable gives the same well-defined behavior on CPU and GPU.
    available &= np.isfinite(u)
    return u, draws, available


def choose_from_utilities(
    utilities: ArrayLike,
    uniforms: ArrayLike,
    availability: ArrayLike | None = None,
) -> ChoiceResult[NDArray]:
    """Simulate one MNL choice and logsum per utility row.

    Alternatives are traversed in column order. Selection uses the first
    cumulative weight greater than or equal to ``uniform * total_weight``. This
    matches ActivitySim's subtract-until-nonpositive ``choice_maker`` boundary
    rule, including its behavior for an exact zero draw.
    """

    u, draws, available = _validated_inputs(utilities, uniforms, availability)
    n_rows, n_alts = u.shape
    choices = np.full(n_rows, -1, dtype=np.int32)
    logsums = np.full(n_rows, -np.inf, dtype=np.float32)
    valid_rows = available.any(axis=1)
    if not valid_rows.any():
        return ChoiceResult(choices, logsums)

    masked = np.where(available, u, -np.inf)
    row_max = masked.max(axis=1)
    shifted = np.full_like(u, -np.inf)
    np.subtract(u, row_max[:, None], out=shifted, where=available & valid_rows[:, None])
    weights = np.zeros_like(u)
    np.exp(shifted, out=weights, where=available & valid_rows[:, None])
    totals = weights.sum(axis=1, dtype=np.float32)
    logsums[valid_rows] = row_max[valid_rows] + np.log(totals[valid_rows])

    thresholds = draws * totals
    cumulative = np.cumsum(weights, axis=1, dtype=np.float32)
    hits = cumulative >= thresholds[:, None]
    selected = hits.argmax(axis=1).astype(np.int32)
    has_hit = hits.any(axis=1) & valid_rows
    choices[has_hit] = selected[has_hit]

    # Match ActivitySim's rare bad-probability fallback: choose the first
    # maximum-probability alternative when no CDF bin captures the draw.
    missed = valid_rows & ~has_hit
    if missed.any():
        choices[missed] = np.argmax(weights[missed], axis=1).astype(np.int32)
    return ChoiceResult(choices, logsums)


def choose_from_probabilities(probabilities: ArrayLike, uniforms: ArrayLike) -> NDArray[np.int32]:
    """ActivitySim-compatible inverse-CDF selection from probability rows."""

    probs = np.ascontiguousarray(probabilities)
    draws = np.ascontiguousarray(uniforms).reshape(-1)
    if probs.ndim != 2 or draws.shape != (probs.shape[0],):
        raise ValueError("probabilities must be (N,A) and uniforms must contain N values")
    if np.any((draws < 0.0) | (draws >= 1.0) | ~np.isfinite(draws)):
        raise ValueError("uniforms must be finite values in [0, 1)")
    cumulative = np.cumsum(probs, axis=1, dtype=probs.dtype)
    hits = cumulative >= draws[:, None]
    choices = hits.argmax(axis=1).astype(np.int32)
    missed = ~hits.any(axis=1)
    choices[missed] = np.argmax(probs[missed], axis=1).astype(np.int32)
    return choices


def linear_choice(
    chooser_features: ArrayLike,
    coefficients: ArrayLike,
    constants: ArrayLike,
    uniforms: ArrayLike,
    availability: ArrayLike | None = None,
) -> ChoiceResult[NDArray]:
    """Reference fused-linear model, materializing ``X @ beta.T + constant``.

    Shapes are ``chooser_features=(N,F)``, ``coefficients=(A,F)`` and
    ``constants=(A,)``. The CUDA implementation exposes the same contract but
    does not write the intermediate ``(N,A)`` utility matrix to global memory.
    """

    x = np.ascontiguousarray(chooser_features, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float32)
    asc = np.ascontiguousarray(constants, dtype=np.float32)
    if x.ndim != 2 or beta.ndim != 2:
        raise ValueError("chooser_features and coefficients must be two-dimensional")
    if x.shape[1] != beta.shape[1]:
        raise ValueError("chooser_features and coefficients must share a feature dimension")
    if asc.shape != (beta.shape[0],):
        raise ValueError(f"constants must have shape {(beta.shape[0],)}, got {asc.shape}")
    utilities = x @ beta.T + asc
    return choose_from_utilities(utilities, uniforms, availability)
