"""Public data structures shared by CPU and GPU backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class ChoiceResult(Generic[ArrayT]):
    """Outputs from a row-wise multinomial-logit simulation.

    ``choices`` contains zero-based alternative positions. A value of ``-1``
    means that the chooser had no finite, available alternative. ``logsums``
    contains stable log-sum-exp values and is ``-inf`` for those invalid rows.

    Random draws are inputs rather than being generated internally. This is a
    deliberate reproducibility boundary: ActivitySim can remain the owner of
    its random-number stream while CPU and GPU implementations consume exactly
    the same draws.
    """

    choices: ArrayT
    logsums: ArrayT
    boundary_distances: ArrayT | None = None
