from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from activitysim.core.random import Random
from activitysim.core import simulate
from choiceforge.phase59_boundary_logsums import BorrowedNormals, recompute, period_slots


def test_representative_hour_must_not_replace_its_skim_period_key():
    rows = pd.DataFrame({"start":[18, 15], "end":[22, 17],
                         "out_period":["EV", "PM"], "in_period":["EV", "PM"]})
    np.testing.assert_array_equal(period_slots(rows), [24, 18])
    rows.loc[0, "out_period"] = "unknown"
    with pytest.raises(ValueError, match="skim-period"):
        period_slots(rows)


def rng_for(frame, seed, offset):
    rng = Random()
    rng.set_base_seed(seed)
    rng.add_channel("tours", frame)
    rng.begin_step("mandatory_tour_scheduling")
    rng.channels["tours"].row_states["offset"] += offset
    return rng


@pytest.mark.parametrize("seed", [0, 17, 991])
@pytest.mark.parametrize("offset", [0, 3, 11])
def test_borrowed_six_normals_match_authoritative_calls(seed, offset):
    unique = pd.DataFrame({"person_id":[4, 5, 6]}, index=pd.Index([13, 7, 99], name="tour_id"))
    frame = unique.loc[[99, 13, 13, 7, 99]]
    first, second = rng_for(unique, seed, offset), rng_for(unique, seed, offset)
    values = first.normal_for_df(unique, broadcast=True, size=6)
    borrowed = BorrowedNormals(values)
    np.testing.assert_array_equal(borrowed(frame, broadcast=True, size=6),
                                  second.normal_for_df(frame, broadcast=True, size=6))
    assert borrowed.cursor == 6
    pd.testing.assert_frame_equal(first.channels["tours"].row_states, second.channels["tours"].row_states)
    scalar_rng = rng_for(unique, seed, offset)
    scalars = np.column_stack([scalar_rng.normal_for_df(unique, broadcast=True) for _ in range(6)])
    # The actual assign_variables preprocessor deliberately batches these.
    # Six scalar calls would change the stream despite the same final offset.
    assert np.any(values.to_numpy()[:, 1:] != scalars[:, 1:])
    with pytest.raises(ValueError, match="unsupported"):
        borrowed(frame, broadcast=True)


@pytest.mark.parametrize("failure", [None, "raise", "ledger", "few"])
def test_recompute_restores_hooks_and_requires_unchanged_ledger(failure):
    frame = pd.DataFrame({"person_id":[4, 5]}, index=pd.Index([13, 7], name="tour_id"))
    rng = rng_for(frame, 17, 0)
    normals = rng.normal_for_df(frame, broadcast=True, size=6)
    state = SimpleNamespace(get_rn_generator=lambda:rng)
    context = dict(state=state, rows=frame, tours=frame, purpose="work", settings=None,
                   network_los=None, skims={}, trace_label="test", normals=normals)
    previous_simple = simulate.simple_simulate_logsums
    prior_dict = dict(rng.__dict__)
    def cpu(*args): pass
    def compute(state, rows, *args):
        assert simulate.simple_simulate_logsums is cpu
        if failure == "raise": raise RuntimeError("intentional")
        rng.normal_for_df(rows, broadcast=True, size=5 if failure == "few" else 6)
        if failure == "ledger": rng.channels["tours"].row_states["offset"] += 1
        return pd.Series(np.zeros(len(rows)), index=rows.index)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            recompute(context, frame.index, compute, cpu)
    else:
        rows, values = recompute(context, frame.index, compute, cpu)
        assert rows.index.equals(frame.index)
        np.testing.assert_array_equal(values, [0., 0.])
    assert simulate.simple_simulate_logsums is previous_simple
    assert rng.__dict__.get("normal_for_df") is prior_dict.get("normal_for_df")
