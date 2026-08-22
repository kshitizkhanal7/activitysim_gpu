"""Fused CPU baselines for the fixed-alternative linear-choice operation.

These implementations deliberately use the same public contract and edge-case
semantics as the NumPy reference and CUDA backend. They do not materialize an
``N x A`` utility matrix. Each chooser is processed independently and its
utilities are recomputed across three short passes: maximum, exponential sum,
and inverse-CDF selection. The parallel version distributes choosers across
Numba CPU threads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .api import ChoiceResult

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover - depends on optional installation
    numba = None
    njit = None
    prange = range


def numba_available() -> bool:
    """Return whether the optional Numba dependency is installed."""

    return numba is not None


if numba is not None:

    @njit(cache=True)
    def _linear_choice_serial(x, beta, constants, draws, available, use_availability):
        n_rows, n_features = x.shape
        n_alts = beta.shape[0]
        choices = np.full(n_rows, -1, dtype=np.int32)
        logsums = np.full(n_rows, -np.inf, dtype=np.float32)
        for row in range(n_rows):
            row_max = np.float32(-np.inf)
            for alt in range(n_alts):
                if use_availability and not available[row, alt]:
                    continue
                utility = constants[alt]
                for feature in range(n_features):
                    utility += x[row, feature] * beta[alt, feature]
                if np.isfinite(utility) and utility > row_max:
                    row_max = utility
            if not np.isfinite(row_max):
                continue

            total = np.float32(0.0)
            for alt in range(n_alts):
                if use_availability and not available[row, alt]:
                    continue
                utility = constants[alt]
                for feature in range(n_features):
                    utility += x[row, feature] * beta[alt, feature]
                if np.isfinite(utility):
                    total += np.exp(utility - row_max)

            threshold = draws[row] * total
            cumulative = np.float32(0.0)
            selected = -1
            max_weight = np.float32(-1.0)
            max_alt = -1
            for alt in range(n_alts):
                weight = np.float32(0.0)
                if not use_availability or available[row, alt]:
                    utility = constants[alt]
                    for feature in range(n_features):
                        utility += x[row, feature] * beta[alt, feature]
                    if np.isfinite(utility):
                        weight = np.exp(utility - row_max)
                if weight > max_weight:
                    max_weight = weight
                    max_alt = alt
                cumulative += weight
                if selected < 0 and cumulative >= threshold:
                    selected = alt
            choices[row] = selected if selected >= 0 else max_alt
            logsums[row] = row_max + np.log(total)
        return choices, logsums


    @njit(cache=True, parallel=True)
    def _linear_choice_parallel(x, beta, constants, draws, available, use_availability):
        n_rows, n_features = x.shape
        n_alts = beta.shape[0]
        choices = np.full(n_rows, -1, dtype=np.int32)
        logsums = np.full(n_rows, -np.inf, dtype=np.float32)
        for row in prange(n_rows):
            row_max = np.float32(-np.inf)
            for alt in range(n_alts):
                if use_availability and not available[row, alt]:
                    continue
                utility = constants[alt]
                for feature in range(n_features):
                    utility += x[row, feature] * beta[alt, feature]
                if np.isfinite(utility) and utility > row_max:
                    row_max = utility
            if not np.isfinite(row_max):
                continue

            total = np.float32(0.0)
            for alt in range(n_alts):
                if use_availability and not available[row, alt]:
                    continue
                utility = constants[alt]
                for feature in range(n_features):
                    utility += x[row, feature] * beta[alt, feature]
                if np.isfinite(utility):
                    total += np.exp(utility - row_max)

            threshold = draws[row] * total
            cumulative = np.float32(0.0)
            selected = -1
            max_weight = np.float32(-1.0)
            max_alt = -1
            for alt in range(n_alts):
                weight = np.float32(0.0)
                if not use_availability or available[row, alt]:
                    utility = constants[alt]
                    for feature in range(n_features):
                        utility += x[row, feature] * beta[alt, feature]
                    if np.isfinite(utility):
                        weight = np.exp(utility - row_max)
                if weight > max_weight:
                    max_weight = weight
                    max_alt = alt
                cumulative += weight
                if selected < 0 and cumulative >= threshold:
                    selected = alt
            choices[row] = selected if selected >= 0 else max_alt
            logsums[row] = row_max + np.log(total)
        return choices, logsums


def _validated_linear_inputs(
    chooser_features: Any,
    coefficients: Any,
    constants: Any,
    uniforms: Any,
    availability: Any | None,
):
    x = np.ascontiguousarray(chooser_features, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float32)
    asc = np.ascontiguousarray(constants, dtype=np.float32)
    draws = np.ascontiguousarray(uniforms, dtype=np.float32)
    if x.ndim != 2 or beta.ndim != 2 or x.shape[1] != beta.shape[1]:
        raise ValueError("chooser_features and coefficients require matching feature dimensions")
    n_rows = x.shape[0]
    n_alts = beta.shape[0]
    if asc.shape != (n_alts,) or draws.shape != (n_rows,):
        raise ValueError("constants or uniforms have incompatible shapes")
    if np.any((draws < 0.0) | (draws >= 1.0) | ~np.isfinite(draws)):
        raise ValueError("uniforms must be finite values in [0, 1)")
    if availability is None:
        available = np.empty((1, 1), dtype=np.bool_)
        use_availability = False
    else:
        available = np.ascontiguousarray(availability, dtype=np.bool_)
        if available.shape != (n_rows, n_alts):
            raise ValueError(f"availability must have shape {(n_rows, n_alts)}, got {available.shape}")
        use_availability = True
    return x, beta, asc, draws, available, use_availability


def linear_choice_numba(
    chooser_features: Any,
    coefficients: Any,
    constants: Any,
    uniforms: Any,
    availability: Any | None = None,
    *,
    parallel: bool = True,
    threads: int | None = None,
) -> ChoiceResult[np.ndarray]:
    """Run a fused Numba CPU implementation without a utility matrix.

    ``threads`` only applies to the parallel implementation. Numba's thread
    setting is process-global, so benchmark callers should avoid changing it
    concurrently with other Numba work.
    """

    if numba is None:
        raise RuntimeError("The fused CPU backend requires Numba; install the 'cpu' extra.")
    x, beta, asc, draws, available, use_availability = _validated_linear_inputs(
        chooser_features, coefficients, constants, uniforms, availability
    )
    if parallel and threads is not None:
        if not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError(
                f"threads must be between 1 and {numba.config.NUMBA_NUM_THREADS}, got {threads}"
            )
        numba.set_num_threads(threads)
    implementation = _linear_choice_parallel if parallel else _linear_choice_serial
    choices, logsums = implementation(x, beta, asc, draws, available, use_availability)
    return ChoiceResult(choices, logsums)
