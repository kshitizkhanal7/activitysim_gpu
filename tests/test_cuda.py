import numpy as np
import pytest

from choiceforge.cuda_backend import CudaChoiceBackend, cuda_available
from choiceforge.reference import choose_from_utilities, linear_choice

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")


def test_cuda_utility_kernel_matches_reference():
    rng = np.random.default_rng(11)
    utilities = rng.normal(size=(257, 17)).astype(np.float32)
    availability = rng.random((257, 17)) > 0.15
    availability[:, 0] = True
    draws = rng.random(257, dtype=np.float32) * np.float32(0.98) + np.float32(0.01)
    expected = choose_from_utilities(utilities, draws, availability)
    actual = CudaChoiceBackend().choose_from_utilities(utilities, draws, availability)
    np.testing.assert_array_equal(actual.choices, expected.choices)
    np.testing.assert_allclose(actual.logsums, expected.logsums, rtol=2e-6, atol=2e-6)


def test_cuda_fused_linear_kernel_matches_reference():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(513, 12)).astype(np.float32)
    beta = rng.normal(scale=0.25, size=(24, 12)).astype(np.float32)
    constants = rng.normal(scale=0.25, size=24).astype(np.float32)
    availability = rng.random((513, 24)) > 0.1
    availability[:, 0] = True
    draws = rng.random(513, dtype=np.float32) * np.float32(0.98) + np.float32(0.01)
    expected = linear_choice(x, beta, constants, draws, availability)
    actual = CudaChoiceBackend().linear_choice(x, beta, constants, draws, availability)
    np.testing.assert_array_equal(actual.choices, expected.choices)
    np.testing.assert_allclose(actual.logsums, expected.logsums, rtol=3e-6, atol=3e-6)

