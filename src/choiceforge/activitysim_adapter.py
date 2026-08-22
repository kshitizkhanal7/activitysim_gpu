"""A narrow integration seam for ActivitySim-style utility matrices.

This is intentionally not monkey-patching ActivitySim internals. It establishes
the reproducibility and labeling contract that a future registered ActivitySim
backend can call after expression evaluation. The next milestone will compile a
supported subset of specification expressions directly into the fused kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .api import ChoiceResult
from .cuda_backend import CudaChoiceBackend, cuda_available
from .reference import choose_from_probabilities, choose_from_utilities


def simulate_utility_matrix(
    utilities: Any,
    random_draws: Any,
    availability: Any | None = None,
    *,
    alternative_ids: Any | None = None,
    backend: str = "auto",
) -> ChoiceResult[np.ndarray]:
    """Run MNL simulation while preserving ActivitySim-owned random draws.

    ``alternative_ids`` maps zero-based kernel positions back to model labels.
    With no mapping, choices remain zero-based positions. Invalid rows remain
    ``-1`` under either representation.
    """

    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    use_cuda = backend == "cuda" or (backend == "auto" and cuda_available())
    if use_cuda:
        result = CudaChoiceBackend().choose_from_utilities(
            utilities, random_draws, availability
        )
    else:
        result = choose_from_utilities(utilities, random_draws, availability)

    choices = np.asarray(result.choices)
    if alternative_ids is not None:
        labels = np.asarray(alternative_ids)
        if labels.ndim != 1 or labels.shape[0] != np.shape(utilities)[1]:
            raise ValueError("alternative_ids must contain one label per utility column")
        mapped = np.empty(choices.shape, dtype=labels.dtype)
        valid = choices >= 0
        if not valid.all() and not np.issubdtype(labels.dtype, np.signedinteger):
            raise ValueError("non-integer labels cannot represent invalid choice -1")
        mapped[valid] = labels[choices[valid]]
        mapped[~valid] = -1
        choices = mapped
    return ChoiceResult(choices, np.asarray(result.logsums))


def make_choices(
    state: Any,
    probs: Any,
    trace_label: str | None = None,
    trace_choosers: Any | None = None,
    allow_bad_probs: bool = False,
    *,
    backend: str = "auto",
) -> tuple[Any, Any]:
    """ActivitySim-compatible ``logit.make_choices`` execution seam.

    The signature intentionally mirrors ActivitySim 1.4. The state remains the
    owner of random-number generation. Tracing arguments are accepted for call
    compatibility but detailed tracing remains a future integration milestone.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("ActivitySim integration requires pandas") from exc

    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    row_sums = probs.sum(axis=1)
    bad = np.abs(np.asarray(row_sums) - 1.0) > 0.001
    if bad.any() and not allow_bad_probs:
        raise ValueError(f"{int(bad.sum())} probability rows do not add up to 1")

    draws = np.asarray(state.get_rn_generator().random_for_df(probs)).reshape(-1)
    use_cuda = backend == "cuda" or (backend == "auto" and cuda_available())
    if use_cuda:
        choices = CudaChoiceBackend().choose_from_probabilities(probs.values, draws)
    else:
        choices = choose_from_probabilities(probs.values, draws)
    return (
        pd.Series(choices, index=probs.index),
        pd.Series(draws, index=probs.index),
    )
