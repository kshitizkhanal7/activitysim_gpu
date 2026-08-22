import numpy as np
import pytest

from choiceforge.cuda_backend import cuda_available
from choiceforge.interaction_backend import (
    CudaInteractionBackend,
    choose_batched_terms_numpy,
    choose_terms_numpy,
    offsets_from_ids,
)


def test_offsets_from_contiguous_ids():
    assert np.array_equal(offsets_from_ids([4, 4, 9, 9, 9]), [0, 2, 5])


def test_offsets_reject_empty():
    with pytest.raises(ValueError):
        offsets_from_ids([])


def _segmented_fixture():
    terms = np.asarray(
        [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, -1]],
        dtype=np.float32,
    )
    coefficients = np.asarray([[0.5, 1.0], [-0.25, 0.75]], dtype=np.float32)
    offsets = np.asarray([0, 2, 3, 6], dtype=np.int64)
    segments = np.asarray([0, 0, 1], dtype=np.int32)
    draws = np.asarray([0.2, 0.8, 0.4], dtype=np.float64)
    return terms, coefficients, offsets, segments, draws


def test_batched_cpu_matches_individual_segments():
    terms, coefficients, offsets, segments, draws = _segmented_fixture()
    batched = choose_batched_terms_numpy(
        terms, coefficients, offsets, segments, draws
    )
    first = choose_terms_numpy(terms[:3], coefficients[0], offsets[:3], draws[:2])
    second = choose_terms_numpy(
        terms[3:], coefficients[1], np.asarray([0, 3]), draws[2:]
    )
    assert np.array_equal(batched.choices, np.r_[first.choices, second.choices])
    assert np.allclose(batched.logsums, np.r_[first.logsums, second.logsums])


@pytest.mark.skipif(not cuda_available(), reason="CUDA is unavailable")
def test_batched_cuda_matches_batched_cpu():
    args = _segmented_fixture()
    expected = choose_batched_terms_numpy(*args)
    actual = CudaInteractionBackend().choose_from_batched_terms(*args)
    assert np.array_equal(actual.choices, expected.choices)
    assert np.allclose(actual.logsums, expected.logsums, atol=1e-5)
