from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from activitysim.core.random import Random
from choiceforge.phase58_trip_runtime import TripRuntime


def state_for(frame, seed):
    rng = Random()
    rng.set_base_seed(seed)
    rng.add_channel("trips", frame)
    return SimpleNamespace(get_rn_generator=lambda:rng)


@pytest.mark.parametrize("seed", [0, 17, 991])
def test_live_rng_matches_upstream_across_changed_seeds_steps_and_permutations(seed):
    frame = pd.DataFrame(index=pd.Index([311, 22, 530, 719, 813, 44], name="trip_id"))
    expected_state, actual_state = state_for(frame, seed), state_for(frame, seed)
    runtime = TripRuntime()
    for step in ("trip_destination", "trip_scheduling", "trip_mode_choice"):
        ref, gpu = expected_state.get_rn_generator(), actual_state.get_rn_generator()
        ref.begin_step(step)
        gpu.begin_step(step)
        with runtime.for_step(actual_state, step):
            for active in (frame, frame.iloc[[4, 0, 5, 2]], frame.iloc[::-1]):
                np.testing.assert_array_equal(ref.random_for_df(active, 3), gpu.random_for_df(active, 3))
                mu = pd.Series(np.linspace(0.3, 1.1, len(active)), index=active.index)
                np.testing.assert_array_equal(ref.lognormal_for_df(active, mu, 0, broadcast=True, scale=True),
                                              gpu.lognormal_for_df(active, mu, 0, broadcast=True, scale=True))
                np.testing.assert_array_equal(ref.normal_for_df(active, mu=1, sigma=.2, size=3),
                                              gpu.normal_for_df(active, mu=1, sigma=.2, size=3))
                for size in (None, 3, 7):
                    for broadcast in (False, True):
                        np.testing.assert_array_equal(ref.normal_for_df(active, size=size, broadcast=broadcast),
                                                      gpu.normal_for_df(active, size=size, broadcast=broadcast))
                np.testing.assert_array_equal(ref.random_for_df(active), gpu.random_for_df(active))
            pd.testing.assert_frame_equal(ref.channels["trips"].row_states, gpu.channels["trips"].row_states)
        assert "random_for_df" not in gpu.__dict__
        assert "_choiceforge_phase58_runtime" not in gpu.__dict__
        ref.end_step(step)
        gpu.end_step(step)
    assert runtime.summary()["epochs"] == 3


def test_runtime_restores_methods_on_failure_and_rejects_duplicate_uniform_ids():
    frame = pd.DataFrame(index=pd.Index([1, 2], name="trip_id"))
    state = state_for(frame, 5)
    rng = state.get_rn_generator()
    rng.begin_step("trip_scheduling")
    with pytest.raises(ValueError, match="unique"):
        with TripRuntime().for_step(state, "trip_scheduling"):
            rng.random_for_df(frame.iloc[[0, 0]])
    assert "random_for_df" not in rng.__dict__


@pytest.mark.parametrize("first", [False, True])
def test_scheduling_consumes_resident_draws_with_exact_choices_and_ledger(first):
    from choiceforge.activitysim_trip_scheduling import TripSchedulingDeviceService
    from choiceforge.phase58_trip_runtime import ResidentSchedulingService
    spec = pd.DataFrame({"purpose": ["work", "shop"],
                         "0": [.1, .2], "1": [.3, .1], "2": [.1, .4],
                         "3": [.2, .1], "4": [.3, .2]})
    frame = pd.DataFrame({"purpose": ["work", "shop"]*19,
                          "earliest": [5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 5]*2,
                          "latest": [9]*38}, index=pd.Index(np.arange(38)*37+9, name="trip_id"))
    reference, candidate = state_for(frame, 319), state_for(frame, 319)
    for state in (reference, candidate):
        state.get_rn_generator().begin_step("trip_scheduling")
    old = TripSchedulingDeviceService(spec, ["purpose"])
    runtime = TripRuntime()
    new = ResidentSchedulingService(spec, ["purpose"], runtime)
    expected, _, expected_failed = old.choose(reference, frame, depart_alt_base=5, first_trip_in_leg=first)
    with runtime.for_step(candidate, "trip_scheduling"):
        actual, random_series, failed = new.choose(candidate, frame, depart_alt_base=5, first_trip_in_leg=first)
    pd.testing.assert_series_equal(actual, expected)
    np.testing.assert_array_equal(failed, expected_failed)
    assert random_series is None
    assert runtime.summary()["device_only_draws"] == len(frame)
    pd.testing.assert_frame_equal(reference.get_rn_generator().channels["trips"].row_states,
                                  candidate.get_rn_generator().channels["trips"].row_states)
