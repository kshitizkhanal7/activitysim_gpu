import numpy as np
import pytest

from choiceforge.cuda_backend import CudaChoiceBackend, cuda_available
from choiceforge.reference import linear_choice


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA is not available")


@pytest.mark.parametrize("alternatives,features", [(33, 17), (190, 69)])
def test_linear_choice_multiple_warps_matches_reference(alternatives, features):
    rng = np.random.default_rng(20260810 + alternatives)
    rows = 4097
    x = rng.normal(size=(rows, features)).astype(np.float32)
    beta = rng.normal(scale=0.2, size=(alternatives, features)).astype(np.float32)
    constants = rng.normal(scale=0.2, size=alternatives).astype(np.float32)
    availability = rng.random((rows, alternatives)) > 0.05
    availability[:, 0] = True
    draws = rng.random(rows, dtype=np.float32)

    expected = linear_choice(x, beta, constants, draws, availability)
    actual = CudaChoiceBackend().linear_choice(x, beta, constants, draws, availability)

    np.testing.assert_array_equal(actual.choices, expected.choices)
    np.testing.assert_allclose(actual.logsums, expected.logsums, rtol=2e-6, atol=2e-6)
