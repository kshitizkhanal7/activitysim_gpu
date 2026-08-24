from __future__ import annotations

import numpy as np
import pytest

from choiceforge.calibrated_chain import (
    choice_from_probabilities_cpu,
    mnl_probabilities,
    mnl_utilities,
)
from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.fused_mnl import FusedFixedMnlCudaModel
from choiceforge.gpu_native import GpuOnlyViolation


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")


def test_fused_fixed_mnl_matches_independent_dense_oracle():
    cp = _cupy()
    expressions = (
        "age < 30",
        "income * (ptype == 1)",
        "(ptype == 2) & female",
        "auto_ownership == 0",
    )
    coefficients = np.asarray(
        [[0.2, -0.1, 0.4], [0.01, 0.02, -0.03], [0.7, -0.2, 0.1], [-1, 0, 1]],
        dtype=np.float64,
    )
    columns = ("age", "income", "ptype", "female")
    values = np.asarray(
        [[20, 10, 1, 0], [40, 50, 2, 1], [25, 80, 2, 1]], dtype=np.float64
    )
    auto = np.asarray([0, 2, 1], dtype=np.float64)
    draws = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    features = np.column_stack(
        [
            values[:, 0] < 30,
            values[:, 1] * (values[:, 2] == 1),
            (values[:, 2] == 2) & values[:, 3].astype(bool),
            auto == 0,
        ]
    ).astype(np.float64)
    probabilities, expected_logsums = mnl_probabilities(
        mnl_utilities(features, coefficients), np
    )
    expected_choices = choice_from_probabilities_cpu(probabilities, draws)

    model = FusedFixedMnlCudaModel(expressions, coefficients, columns)
    result = model.choose(cp.asarray(values), cp.asarray(auto), cp.asarray(draws))
    np.testing.assert_array_equal(cp.asnumpy(result.choices), expected_choices)
    np.testing.assert_allclose(cp.asnumpy(result.logsums), expected_logsums, atol=1e-12)
    with pytest.raises(GpuOnlyViolation):
        model.choose(values, cp.asarray(auto), cp.asarray(draws))
