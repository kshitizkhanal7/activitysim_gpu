from dataclasses import dataclass

import numpy as np
import pytest


cp = pytest.importorskip("cupy")

from choiceforge.device_input_expansion import ResidentInputExpansionPlan


@dataclass(frozen=True)
class _Invocation:
    rows: int
    float_inputs: object
    int_inputs: object
    skim_arguments: tuple
    logical_skim_bindings: int
    dense_input_bytes: int
    skim_coordinate_bytes: int


def _fixture(nonfactorable=False):
    rows = 50
    owner = np.repeat(np.arange(2), 25)
    slot = np.tile(np.arange(25), 2)
    varying = np.arange(rows) if nonfactorable else slot
    floats = cp.asarray(
        np.column_stack((np.ones(rows), owner + 0.5, varying * 0.25)),
        dtype=cp.float64,
    )
    integers = cp.asarray(np.column_stack((owner, slot)), dtype=cp.int64)
    origin = cp.asarray(owner, dtype=cp.int64)
    destination = cp.asarray(owner + 3, dtype=cp.int64)
    period = cp.asarray(slot, dtype=cp.int64)
    cube = cp.zeros((4, 4, 25), dtype=cp.float32)
    invocation = _Invocation(
        rows=rows,
        float_inputs=floats,
        int_inputs=integers,
        skim_arguments=(cube, origin, destination, period, np.int64(4), np.int64(25)),
        logical_skim_bindings=1,
        dense_input_bytes=floats.nbytes + integers.nbytes,
        skim_coordinate_bytes=origin.nbytes + destination.nbytes + period.nbytes,
    )
    metadata = {
        "chooser_ids": np.repeat((101, 202), 25),
        "start": np.tile(np.repeat((1, 6, 10, 15, 19), 5), 2),
        "end": np.tile(np.tile((1, 6, 10, 15, 19), 5), 2),
    }
    return invocation, metadata


def test_compact_plan_reconstructs_every_array_bit_exactly():
    invocation, metadata = _fixture()
    plan = ResidentInputExpansionPlan.compile(invocation, metadata)
    plan.execute()
    cp.cuda.Stream.null.synchronize()

    assert cp.array_equal(plan.invocation.float_inputs, invocation.float_inputs)
    assert cp.array_equal(plan.invocation.int_inputs, invocation.int_inputs)
    for position in (1, 2, 3):
        assert cp.array_equal(
            plan.invocation.skim_arguments[position],
            invocation.skim_arguments[position],
        )
    assert plan.compact_bytes < (
        invocation.dense_input_bytes + invocation.skim_coordinate_bytes
    )
    assert plan.classification()["float_inputs"] == {
        "constant_columns": 1,
        "chooser_columns": 1,
        "slot_columns": 1,
        "chooser_slot_pattern_columns": 0,
        "target_bytes": 1200,
        "compact_bytes": 239,
    }


def test_compact_plan_fails_closed_on_row_specific_dense_column():
    invocation, metadata = _fixture(nonfactorable=True)
    with pytest.raises(ValueError, match="neither constant, chooser-factored, nor slot-factored"):
        ResidentInputExpansionPlan.compile(invocation, metadata)
