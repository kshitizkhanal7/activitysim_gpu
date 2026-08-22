import numpy as np
import pytest

pd = pytest.importorskip("pandas")
logit = pytest.importorskip("activitysim.core.logit")

from choiceforge.activitysim_adapter import make_choices
from choiceforge.cuda_backend import cuda_available


class FixedRandomGenerator:
    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=np.float64)

    def random_for_df(self, frame):
        assert len(frame) == len(self.draws)
        return self.draws.copy()


class FakeState:
    def __init__(self, draws):
        self.generator = FixedRandomGenerator(draws)

    def get_rn_generator(self):
        return self.generator


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable"))])
def test_make_choices_matches_activitysim_choice_maker(backend):
    probs = pd.DataFrame(
        [[0.0, 0.25, 0.75], [0.4, 0.3, 0.3], [0.1, 0.2, 0.7]],
        index=pd.Index([101, 205, 999], name="chooser_id"),
    )
    draws = np.array([0.0, 0.4, 0.999], dtype=np.float64)
    expected = logit.choice_maker(probs.values, draws)
    choices, returned_draws = make_choices(FakeState(draws), probs, backend=backend)
    np.testing.assert_array_equal(choices.values, expected)
    np.testing.assert_array_equal(returned_draws.values, draws)
    assert choices.index.equals(probs.index)
