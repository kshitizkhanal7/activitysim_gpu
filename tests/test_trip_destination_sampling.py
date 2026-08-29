import numpy as np
import pandas as pd

from choiceforge.trip_destination_sampling import (
    Phase39Unsupported,
    _EXPECTED_EXPRESSIONS,
    _activitysim_inverse_cdf,
    _specification_contract,
)


class _Settings:
    skip_failed_choices = False


class _Rng:
    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=np.float64)
        self.calls = []

    def random_for_df(self, frame, n=1):
        self.calls.append((frame.index.copy(), n))
        return self.draws[: len(frame), :n].copy()


class _State:
    settings = _Settings()

    def __init__(self, draws):
        self.rng = _Rng(draws)

    def get_rn_generator(self):
        return self.rng


def test_phase39_specification_contract_is_hashed_and_fail_closed():
    spec = pd.DataFrame(
        {"work": np.arange(15, dtype=np.float32)}, index=_EXPECTED_EXPRESSIONS
    )
    coefficients, fingerprint = _specification_contract(spec)
    assert coefficients.dtype == np.float32
    assert len(fingerprint) == 64
    changed = spec.rename(index={_EXPECTED_EXPRESSIONS[0]: "changed"})
    with np.testing.assert_raises(Phase39Unsupported):
        _specification_contract(changed)


def test_phase39_retains_activitysim_inverse_cdf_duplicate_contract():
    trips = pd.DataFrame(index=pd.Index([10, 20], name="trip_id"))
    alternatives = pd.DataFrame(index=pd.Index([0, 1, 2], name="dest_taz"))
    utilities = np.asarray([[0.0, -1.0, -2.0], [-1000.0, -1000.0, -1000.0]])
    state = _State([[0.05, 0.10, 0.95], [0.1, 0.2, 0.3]])
    sample, random_draws, guard_rows, guard_seconds = _activitysim_inverse_cdf(
        state, utilities, trips, alternatives, 3, "dest_taz"
    )
    assert random_draws == 3
    assert guard_rows == 0
    assert guard_seconds >= 0
    assert len(state.rng.calls) == 1
    assert state.rng.calls[0][0].tolist() == [10]
    assert sample.index.tolist() == [10, 10]
    assert sample["dest_taz"].tolist() == [0, 2]
    assert sample["pick_count"].tolist() == [2, 1]
    assert sample["prob"].dtype == np.float32
    assert sample["pick_count"].dtype == np.uint32


def test_phase39_precision_guard_reuses_draws_and_exactly_rechecks_ambiguous_rows():
    trips = pd.DataFrame(index=pd.Index([10], name="trip_id"))
    alternatives = pd.DataFrame(index=pd.Index([0, 1], name="dest_taz"))
    state = _State([[0.49]])
    resolved = []

    def resolver(index):
        resolved.extend(index.tolist())
        return np.asarray([[-10.0, 10.0]], dtype=np.float32)

    sample, random_draws, guard_rows, _ = _activitysim_inverse_cdf(
        state,
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        trips,
        alternatives,
        1,
        "dest_taz",
        utility_error_bounds=np.asarray([[1.0, 1.0]], dtype=np.float32),
        exact_utility_resolver=resolver,
    )
    assert random_draws == 1
    assert len(state.rng.calls) == 1
    assert guard_rows == 1
    assert resolved == [10]
    assert sample["dest_taz"].tolist() == [1]
