import numpy as np
import pytest

from choiceforge.numba_backend import linear_choice_numba, numba_available
from choiceforge.reference import linear_choice


pytestmark = pytest.mark.skipif(not numba_available(), reason="Numba is not installed")


@pytest.mark.parametrize("parallel", [False, True])
def test_numba_matches_reference(parallel):
    rng = np.random.default_rng(20260810)
    x = rng.normal(size=(257, 9)).astype(np.float32)
    beta = rng.normal(scale=0.2, size=(17, 9)).astype(np.float32)
    constants = rng.normal(scale=0.2, size=17).astype(np.float32)
    availability = rng.random((257, 17)) > 0.1
    availability[:, 0] = True
    draws = rng.random(257, dtype=np.float32)

    expected = linear_choice(x, beta, constants, draws, availability)
    actual = linear_choice_numba(
        x, beta, constants, draws, availability, parallel=parallel, threads=2
    )

    np.testing.assert_array_equal(actual.choices, expected.choices)
    np.testing.assert_allclose(actual.logsums, expected.logsums, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("parallel", [False, True])
def test_numba_invalid_row_and_availability(parallel):
    x = np.array([[1.0], [np.nan]], dtype=np.float32)
    beta = np.array([[1.0], [2.0]], dtype=np.float32)
    constants = np.zeros(2, dtype=np.float32)
    draws = np.array([0.9, 0.5], dtype=np.float32)
    availability = np.array([[True, False], [True, True]])

    result = linear_choice_numba(
        x, beta, constants, draws, availability, parallel=parallel, threads=2
    )

    np.testing.assert_array_equal(result.choices, [0, -1])
    assert result.logsums[0] == pytest.approx(1.0)
    assert np.isneginf(result.logsums[1])
