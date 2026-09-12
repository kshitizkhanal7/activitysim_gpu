from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from choiceforge.activitysim_scheduling import LoweredSchedulingSpec
from choiceforge.phase59_live_mandatory import (LiveMandatoryScheduler, STATEFUL,
                                               ASSIGNMENTS, validate_lowering)


def test_constructor_does_not_read_artifacts(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact read forbidden")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    scheduler = LiveMandatoryScheduler()
    assert not scheduler.complete
    assert scheduler.boundary_map_entries == 0
    with pytest.raises(ValueError, match="incomplete"):
        scheduler.finish()


def test_live_timetable_vocabulary_is_checked_not_assumed():
    lowered = LoweredSchedulingSpec((), (), (),
        ("mode_choice_logsum", *(f"stateful_{i}" for i in range(7))),
        ("start", "end", "duration"), STATEFUL, ASSIGNMENTS, ())
    validate_lowering(lowered)
    with pytest.raises(ValueError, match="expressions"):
        validate_lowering(replace(lowered, stateful_expressions=STATEFUL[::-1]))
    with pytest.raises(ValueError, match="assignments"):
        validate_lowering(replace(lowered, assignments=ASSIGNMENTS[:1]))
    with pytest.raises(ValueError, match="layout"):
        validate_lowering(replace(lowered, alternative_columns=("end", "start", "duration")))


def test_every_feasible_slot_must_have_a_finite_live_logsum():
    from types import SimpleNamespace
    from choiceforge.cuda_backend import _cupy
    from choiceforge.phase59_live_mandatory import validate_feasible_cache
    cp = _cupy()
    pending = SimpleNamespace(present=cp.ones((2, 25), dtype=cp.bool_), cache=cp.zeros((2, 25)))
    prepared = SimpleNamespace(row_owners=cp.asarray([0, 1]), alternative_ids=cp.asarray([0, 1]))
    slots = cp.asarray([0, 6])
    validate_feasible_cache(pending, prepared, slots)
    pending.present[1, 6] = False
    with pytest.raises(ValueError, match="missing"):
        validate_feasible_cache(pending, prepared, slots)
    pending.present[1, 6] = True
    pending.cache[1, 6] = cp.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_feasible_cache(pending, prepared, slots)
