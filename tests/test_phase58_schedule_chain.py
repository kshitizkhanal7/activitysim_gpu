from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from activitysim.abm.models import trip_scheduling as upstream
from activitysim.abm.models.util import probabilistic_scheduling as ps
from choiceforge.phase58_trip_runtime import TripRuntime
from choiceforge.phase58_schedule_chain import ChainSchedulingService
from test_phase58_trip_runtime import state_for


def fixture(seed):
    random = np.random.default_rng(seed)
    trips, tours = [], []
    for tour_id in range(1, 37):
        start = int(random.integers(5, 16))
        end = int(random.integers(start, 24))
        tours.append(dict(tour_id=tour_id, start=start, end=end, primary_purpose="work",
                          tour_num=1, tour_count=1, parent_tour_id=-1))
        for outbound in (True, False):
            count = int(random.integers(1, 5))
            for num in range(1, count+1):
                trips.append(dict(trip_id=tour_id*100+(0 if outbound else 10)+num,
                                  tour_id=tour_id, person_id=tour_id, outbound=outbound,
                                  trip_num=num, trip_count=count, primary_purpose="work"))
    # A live at-work subtour tests fixed departures and parent-tour constraints.
    tours.append(dict(tour_id=37, start=12, end=14, primary_purpose="atwork",
                      tour_num=1, tour_count=1, parent_tour_id=1))
    tours[0].update(start=7, end=19)
    for outbound in (True, False):
        for num in range(1, 3):
            trips.append(dict(trip_id=3700+(0 if outbound else 10)+num, tour_id=37,
                              person_id=1, outbound=outbound, trip_num=num, trip_count=2,
                              primary_purpose="atwork"))
    spec = pd.DataFrame(random.random((1, 19)), columns=[str(i) for i in range(19)])
    spec.iloc[0, 3:9] = 0  # deliberate zero-probability windows and retry failures
    spec.insert(0, "primary_purpose", "work")
    return pd.DataFrame(trips).set_index("trip_id"), pd.DataFrame(tours).set_index("tour_id"), spec


@pytest.mark.parametrize("seed", [17, 582, 994])
@pytest.mark.parametrize("last", [False, True])
def test_complete_live_chains_match_upstream_choices_and_rng(seed, last, monkeypatch):
    trips, tours, spec = fixture(seed)
    # Group order does not need to match table order.
    trips = trips.sample(frac=1, random_state=seed)
    expected, actual = state_for(trips, seed), state_for(trips, seed)
    for state in (expected, actual):
        state.settings = SimpleNamespace(trace_hh_id=None, skip_failed_choices=False)
        state.get_injectable = lambda _:None
        state.get_rn_generator().begin_step("trip_scheduling")
    settings = upstream.TripSchedulingSettings(probs_join_cols=["primary_purpose"], logic_version=2)
    chunk = SimpleNamespace(log_df=lambda *a, **k:None)
    # Disable diagnostics only; probability arithmetic/choice/retry logic is real upstream.
    monkeypatch.setattr(ps, "_report_bad_choices", lambda *a, **k:None)
    reference = upstream.run_trip_scheduling(expected, trips.copy(), tours, spec, settings,
        estimator=None, is_last_iteration=last, trace_label="test", chunk_sizer=chunk)
    runtime = TripRuntime()
    runtime.chain_events = []
    service = ChainSchedulingService(spec, ["primary_purpose"], runtime)
    with runtime.for_step(actual, "trip_scheduling"):
        choices = service.run(actual, trips, tours, settings, last)
    pd.testing.assert_series_equal(choices.sort_index(), reference.sort_index(), check_dtype=False, check_names=False)
    pd.testing.assert_frame_equal(expected.get_rn_generator().channels["trips"].row_states,
                                  actual.get_rn_generator().channels["trips"].row_states)
    assert runtime.chain_events[0]["intermediate_choice_download_bytes"] == 0


def test_incomplete_legs_fail_closed_before_rng():
    trips, tours, spec = fixture(9)
    trips = trips.drop(trips.index[0])
    state = state_for(trips, 9)
    state.get_rn_generator().begin_step("trip_scheduling")
    runtime = TripRuntime()
    service = ChainSchedulingService(spec, ["primary_purpose"], runtime)
    settings = upstream.TripSchedulingSettings(probs_join_cols=["primary_purpose"], logic_version=2)
    with runtime.for_step(state, "trip_scheduling"), pytest.raises(ValueError, match="complete contiguous"):
        service.run(state, trips, tours, settings, False)
    assert runtime.events == []


def test_negative_probability_specification_rejected():
    _, _, spec = fixture(19)
    spec.iloc[0, 1] = -.1
    with pytest.raises(ValueError, match="nonnegative"):
        ChainSchedulingService(spec, ["primary_purpose"], TripRuntime())
