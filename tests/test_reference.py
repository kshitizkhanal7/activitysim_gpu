import numpy as np
import pytest

from choiceforge.activitysim_adapter import simulate_utility_matrix
from choiceforge.reference import choose_from_utilities, linear_choice


def test_known_choices_and_logsums():
    utilities = np.array([[0.0, 0.0], [0.0, np.log(3.0)]], dtype=np.float32)
    result = choose_from_utilities(utilities, [0.25, 0.80])
    np.testing.assert_array_equal(result.choices, [0, 1])
    np.testing.assert_allclose(result.logsums, [np.log(2.0), np.log(4.0)], rtol=1e-6)


def test_availability_nonfinite_and_invalid_rows():
    utilities = np.array([[3.0, 10.0], [np.nan, np.inf]], dtype=np.float32)
    availability = np.array([[True, False], [True, True]])
    result = choose_from_utilities(utilities, [0.999, 0.5], availability)
    np.testing.assert_array_equal(result.choices, [0, -1])
    assert result.logsums[0] == pytest.approx(3.0)
    assert np.isneginf(result.logsums[1])


def test_linear_reference_matches_materialized_utilities():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 5)).astype(np.float32)
    beta = rng.normal(size=(6, 5)).astype(np.float32)
    constants = rng.normal(size=6).astype(np.float32)
    draws = rng.random(20, dtype=np.float32)
    expected = choose_from_utilities(x @ beta.T + constants, draws)
    actual = linear_choice(x, beta, constants, draws)
    np.testing.assert_array_equal(actual.choices, expected.choices)
    np.testing.assert_allclose(actual.logsums, expected.logsums, rtol=0, atol=0)


def test_draw_validation():
    with pytest.raises(ValueError, match="half-open"):
        choose_from_utilities([[0.0, 1.0]], [1.0])


def test_adapter_maps_alternative_ids():
    result = simulate_utility_matrix(
        [[10.0, 0.0], [0.0, 10.0]],
        [0.5, 0.5],
        alternative_ids=np.array([11, 42]),
        backend="cpu",
    )
    np.testing.assert_array_equal(result.choices, [11, 42])

