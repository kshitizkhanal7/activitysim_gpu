"""Compare the complete retry controller against real upstream ActivitySim."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from activitysim.abm.models import trip_scheduling as upstream
from activitysim.abm.models.util import probabilistic_scheduling as ps
from activitysim.abm.models.util.trip import failed_trip_cohorts
from test_phase58_schedule_chain import fixture
from test_phase58_trip_runtime import state_for


@pytest.mark.parametrize("seed", [17, 582, 994])
@pytest.mark.parametrize("iterations", [1, 3, 100])
def test_complete_retries_match_upstream_and_exact_ledger(seed, iterations, monkeypatch):
    from choiceforge.phase58_trip_runtime import TripRuntime
    from choiceforge.phase59_retry import RetrySchedulingService
    trips, tours, spec = fixture(seed)
    trips = trips.sample(frac=1, random_state=seed)
    expected, actual = state_for(trips, seed), state_for(trips, seed)
    for state in (expected, actual):
        state.settings = SimpleNamespace(trace_hh_id=None, skip_failed_choices=False)
        state.get_injectable = lambda _: None
        state.get_rn_generator().begin_step("trip_scheduling")
        # Nonzero live offsets must not be mistaken for a fresh stream.
        state.get_rn_generator().random_for_df(trips, n=3)
    settings = upstream.TripSchedulingSettings(probs_join_cols=["primary_purpose"],
        logic_version=2, MAX_ITERATIONS=iterations)
    chunk = SimpleNamespace(log_df=lambda *a, **k: None)
    monkeypatch.setattr(ps, "_report_bad_choices", lambda *a, **k: None)
    remaining = trips.copy()
    committed = []
    initial_ledger = actual.get_rn_generator().channels["trips"].row_states.copy()
    for i in range(iterations):
        if remaining.empty:
            break
        choices = upstream.run_trip_scheduling(expected, remaining, tours, spec, settings,
            estimator=None, is_last_iteration=i == iterations-1, trace_label="test", chunk_sizer=chunk)
        failed = choices.reindex(remaining.index).isna()
        cohorts = failed_trip_cohorts(remaining, failed) if i < iterations-1 else failed
        committed.append(choices[~cohorts])
        remaining = remaining[cohorts]
    reference = pd.concat(committed).sort_index()
    runtime = TripRuntime()
    service = RetrySchedulingService(spec, ["primary_purpose"], runtime)
    with runtime.for_step(actual, "trip_scheduling"):
        choices = service.run(actual, trips, tours, settings, iterations == 1)
    pd.testing.assert_series_equal(choices.sort_index(), reference, check_dtype=False, check_names=False)
    pd.testing.assert_frame_equal(expected.get_rn_generator().channels["trips"].row_states,
                                  actual.get_rn_generator().channels["trips"].row_states)
    assert runtime.chain_events[-1]["intermediate_choice_download_bytes"] == 0
    from choiceforge.phase59_cpu_algorithms import retries_cpu
    frame, active, groups, packed, firstout, firstin = service.pack(trips, tours, settings)
    ledger = initial_ledger.loc[active.index]
    random = np.array([np.random.RandomState(int(row.row_seed)).random_sample(int(row.offset)+iterations)[int(row.offset):]
                       for row in ledger.itertuples()])
    cpu_choices, used, failures = retries_cpu(*packed, random, service.cp.asnumpy(service.spec_probabilities),
        settings.DEPART_ALT_BASE, firstout, firstin, iterations)
    pd.testing.assert_series_equal(pd.Series(cpu_choices, index=frame.index).sort_index(), reference,
                                  check_dtype=False, check_names=False)
    np.testing.assert_array_equal(used, (actual.get_rn_generator().channels["trips"].row_states.offset
                                  - initial_ledger.offset).loc[frame.index].to_numpy())
    assert int(failures.sum()) == service.telemetry.failed_choices
