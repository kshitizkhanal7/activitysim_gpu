import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.modelwide_sampling import (
    _compile_phase46_choice,
    _pack_sample_phase46,
    _preserved_order_choices,
)
from choiceforge.modelwide_service import Phase46DestinationService
from choiceforge.trip_destination_resident import (
    _host_duplicate_contract,
    _pack_sample,
)


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")


def test_phase46_gpu_mt19937_matches_scalar_numpy_randomstate():
    cp = _cupy()
    service = Phase46DestinationService(cp)
    seeds = np.asarray([0, 1, 17, 2**31 + 91, 2**32 - 1], dtype=np.uint32)
    offsets = np.asarray([0, 1, 30, 311, 650], dtype=np.int32)
    draws = 31
    actual = cp.asnumpy(service.generate_from_seeds(seeds, offsets, draws))
    expected = []
    for seed, offset in zip(seeds, offsets):
        generator = np.random.RandomState(int(seed))
        generator.rand(int(offset))
        expected.append(generator.rand(draws))
    expected = np.asarray(expected, dtype=np.float64)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_phase46_precomputed_weights_and_selected_probabilities_are_exact():
    cp = _cupy()
    rng = np.random.default_rng(460046)
    rows, alternatives, draws = 37, 1454, 30
    utilities = rng.normal(-3.0, 4.0, size=(rows, alternatives)).astype(np.float32)
    row_maxima = utilities.max(axis=1)
    shifted_utilities = utilities - row_maxima[:, None]
    random_draws = rng.random((rows, draws), dtype=np.float64)
    device_utilities = cp.asarray(utilities)
    device_weights = cp.empty_like(device_utilities)
    device_choices = cp.empty((rows, draws), dtype=cp.int32)
    device_probabilities = cp.empty((rows, draws), dtype=cp.float32)
    guard = cp.zeros(rows, dtype=cp.uint8)
    bad = cp.zeros(rows, dtype=cp.uint8)
    weight_kernel, choice_kernel, _ = _compile_phase46_choice(cp, alternatives)
    weight_kernel(
        ((utilities.size + 255) // 256,),
        (256,),
        (
            device_utilities,
            cp.asarray(row_maxima),
            device_weights,
            np.int64(utilities.size),
            np.int32(alternatives),
        ),
    )
    choice_kernel(
        ((rows + 127) // 128,),
        (128,),
        (
            device_weights,
            cp.asarray(random_draws),
            device_choices,
            device_probabilities,
            guard,
            bad,
            np.int32(rows),
            np.int32(alternatives),
            np.int32(draws),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    expected_weights = np.exp(shifted_utilities)
    assert np.array_equal(
        cp.asnumpy(device_weights).view(np.uint32),
        expected_weights.view(np.uint32),
    )
    expected_probs = expected_weights / expected_weights.sum(axis=1, keepdims=True)
    expected_choices, expected_selected = _preserved_order_choices(
        expected_probs, random_draws, np.arange(alternatives, dtype=np.int32)
    )
    assert not int(cp.count_nonzero(bad).get())
    assert np.array_equal(cp.asnumpy(device_choices), expected_choices)
    assert np.array_equal(
        cp.asnumpy(device_probabilities).view(np.uint32),
        expected_selected.view(np.uint32),
    )


def test_phase46_workspace_grows_then_reuses_its_allocations():
    cp = _cupy()
    service = Phase46DestinationService(cp)
    first = service.sample_workspace(11, 1454, 30)
    pointers = {name: value.data.ptr for name, value in first.items()}
    second = service.sample_workspace(7, 1454, 30)
    assert all(second[name].data.ptr == pointers[name] for name in pointers)
    summary = service.summary()
    assert summary["cell_capacity"] == 11 * 1454
    assert summary["row_capacity"] == 11
    assert 0 < summary["workspace_bytes"] < 1024**3


def test_activitysim_numba_exact_guard_preserves_reference_order_and_bits():
    from activitysim.core.choosing import sample_choices_maker_preserve_ordering

    rng = np.random.default_rng(460047)
    utilities = rng.normal(size=(23, 1454)).astype(np.float32)
    weights = np.exp(utilities - utilities.max(axis=1, keepdims=True))
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    draws = rng.random((23, 30), dtype=np.float64)
    alternatives = np.arange(1454, dtype=np.int32)
    expected_choices, expected_probabilities = _preserved_order_choices(
        probabilities, draws, alternatives
    )
    choices, selected_probabilities = sample_choices_maker_preserve_ordering(
        probabilities, draws, alternatives
    )
    assert np.array_equal(choices.T, expected_choices)
    assert np.array_equal(
        selected_probabilities.T.view(np.uint32),
        expected_probabilities.view(np.uint32),
    )


def test_phase46_compact_packer_matches_activitysim_sorted_contract():
    import pandas as pd

    rng = np.random.default_rng(460048)
    rows, draws = 101, 30
    choosers = pd.DataFrame(index=pd.Index(np.arange(rows) * 17 + 3, name="tour_id"))
    choices = rng.integers(1, 1455, size=(rows, draws), dtype=np.int32)
    probabilities = rng.random((rows, draws), dtype=np.float32)
    random_draws = rng.random((rows, draws), dtype=np.float64)
    first, counts = _host_duplicate_contract(choices)
    expected = _pack_sample(
        choosers, choices, probabilities, random_draws, first, counts, "dest_MAZ"
    )
    actual = _pack_sample_phase46(
        choosers, choices, probabilities, random_draws, first, counts, "dest_MAZ"
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
