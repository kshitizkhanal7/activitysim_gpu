import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.modelwide_sampling import (
    _compile_phase46_choice,
    _pack_sample_phase46,
    _preserved_order_choices,
    numpy_preserved_order_choices,
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


def test_phase47_numpy_guard_matches_activitysim_numba_contract():
    from activitysim.core.choosing import sample_choices_maker_preserve_ordering

    rng = np.random.default_rng(470047)
    probabilities = rng.random((47, 30), dtype=np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random((47, 7), dtype=np.float64)
    alternatives = np.arange(30, dtype=np.int32)
    expected_choices, expected_probabilities = sample_choices_maker_preserve_ordering(
        probabilities, draws, alternatives
    )
    choices, selected_probabilities = numpy_preserved_order_choices(
        probabilities, draws, alternatives
    )
    assert np.array_equal(choices, expected_choices.T)
    assert np.array_equal(
        selected_probabilities.view(np.uint32),
        expected_probabilities.T.view(np.uint32),
    )


def test_phase47_final_workspace_reuses_phase46_storage():
    cp = _cupy()
    service = Phase46DestinationService(cp)
    first = service.final_workspace(101, 2_929, 30)
    pointers = {name: value.data.ptr for name, value in first.items()}
    second = service.final_workspace(47, 1_099, 21)
    assert all(second[name].data.ptr == pointers[name] for name in pointers)
    assert service.summary()["workspace_bytes"] < 1024**3


def test_phase48_resumed_mt19937_state_matches_reseeded_reference_bits():
    cp = _cupy()
    service = Phase46DestinationService(cp)
    seeds = np.asarray([0, 19, 2**31 + 7, 2**32 - 1], dtype=np.uint32)
    offsets = np.asarray([0, 3, 117, 650], dtype=np.int32)
    first = service.generate_from_seeds(seeds, offsets, 30)
    cp.cuda.Stream.null.synchronize()
    first_host = cp.asnumpy(first)
    resumed = service._resume_from_device_state(
        len(seeds), 1, np.arange(len(seeds), dtype=np.int32)
    )
    cp.cuda.Stream.null.synchronize()
    resumed_host = cp.asnumpy(resumed)
    expected = []
    for seed, offset in zip(seeds, offsets):
        generator = np.random.RandomState(int(seed))
        generator.rand(int(offset))
        expected.append(generator.rand(31))
    expected = np.asarray(expected)
    assert np.array_equal(first_host.view(np.uint64), expected[:, :30].view(np.uint64))
    assert np.array_equal(resumed_host.view(np.uint64), expected[:, 30:].view(np.uint64))


def test_phase48_resumed_mt19937_state_follows_final_chooser_permutation():
    cp = _cupy()
    service = Phase46DestinationService(cp)
    seeds = np.asarray([11, 22, 33, 44, 55], dtype=np.uint32)
    offsets = np.asarray([1, 5, 9, 13, 17], dtype=np.int32)
    service.generate_from_seeds(seeds, offsets, 30)
    cp.cuda.Stream.null.synchronize()
    permutation = np.asarray([3, 0, 4, 1, 2], dtype=np.int32)
    resumed = service._resume_from_device_state(
        len(seeds), 1, permutation, skip_draws=6
    )
    cp.cuda.Stream.null.synchronize()
    expected = []
    for source in permutation:
        generator = np.random.RandomState(int(seeds[source]))
        generator.rand(int(offsets[source]) + 36)
        expected.append(generator.rand())
    assert np.array_equal(
        cp.asnumpy(resumed)[:, 0].view(np.uint64),
        np.asarray(expected).view(np.uint64),
    )


@pytest.mark.parametrize("width", [21, 25, 29, 30])
def test_phase48_resident_probability_sum_and_choice_match_numpy(width):
    from choiceforge.modelwide_graph import _compile_resident_choice

    cp = _cupy()
    service = Phase46DestinationService(cp)
    rng = np.random.default_rng(480000 + width)
    rows = 59
    utilities = rng.normal(-2, 3, size=(rows, width)).astype(np.float32)
    draws = rng.random((rows, 1), dtype=np.float64)
    workspace = service.final_workspace(rows, rows * width, width)
    workspace["utilities"].set(utilities)
    cp.max(workspace["utilities"], axis=1, out=workspace["row_maxima"])
    weight_kernel, _, _ = _compile_phase46_choice(cp, width)
    weight_kernel(
        ((utilities.size + 255) // 256,),
        (256,),
        (
            workspace["utilities"],
            workspace["row_maxima"],
            workspace["weights"],
            np.int64(utilities.size),
            np.int32(width),
        ),
    )
    choice_kernel, _ = _compile_resident_choice(cp, width)
    choice_kernel(
        ((rows + 127) // 128,),
        (128,),
        (
            workspace["weights"],
            cp.asarray(draws),
            workspace["positions"],
            workspace["selected_probabilities"],
            workspace["row_totals"],
            workspace["guard"],
            workspace["bad"],
            np.int32(rows),
            np.int32(width),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    shifted = utilities - utilities.max(axis=1, keepdims=True)
    expected_weights = np.exp(shifted)
    expected_totals = expected_weights.sum(axis=1)
    expected_probs = expected_weights / expected_totals[:, None]
    expected_positions, _ = numpy_preserved_order_choices(
        expected_probs, draws, np.arange(width, dtype=np.int32)
    )
    assert not int(cp.count_nonzero(workspace["bad"]).get())
    assert np.array_equal(
        cp.asnumpy(workspace["weights"]).view(np.uint32),
        expected_weights.view(np.uint32),
    )
    assert np.array_equal(
        cp.asnumpy(workspace["row_totals"]).view(np.uint32),
        expected_totals.view(np.uint32),
    )
    actual_positions = cp.asnumpy(workspace["positions"])
    guards = cp.asnumpy(workspace["guard"]).astype(bool)
    assert np.array_equal(actual_positions[~guards], expected_positions[:, 0][~guards])


def test_phase48_exp_correction_matches_numpy_bits_and_enforces_domain():
    from choiceforge.modelwide_graph import _compile_exp_correction

    cp = _cupy()
    # This exact negative float32 bit pattern is one of the 73 values found by
    # the exhaustive 2**32-pattern qualification scan.
    corrected_input = np.asarray([3213170066], dtype=np.uint32).view(np.float32)[0]
    utilities = np.asarray(
        [[corrected_input, 0.0], [-999.0, 0.0], [-81.0, 0.0]], dtype=np.float32
    )
    row_maxima = np.zeros(3, dtype=np.float32)
    device_utilities = cp.asarray(utilities)
    device_weights = cp.exp(device_utilities)
    bad = cp.zeros(3, dtype=cp.uint8)
    correction, (input_bits, output_bits), _ = _compile_exp_correction(cp)
    correction(
        ((utilities.size + 255) // 256,),
        (256,),
        (
            device_utilities,
            cp.asarray(row_maxima),
            device_weights,
            bad,
            input_bits,
            output_bits,
            np.int64(utilities.size),
            np.int32(utilities.shape[1]),
            np.int32(input_bits.size),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    expected = np.exp(np.asarray([corrected_input], dtype=np.float32))
    assert np.array_equal(
        cp.asnumpy(device_weights[:1, :1]).reshape(-1).view(np.uint32),
        expected.view(np.uint32),
    )
    # -999 is the declared padding sentinel; an ordinary value outside
    # [-80, 80] must fail closed.
    assert np.array_equal(cp.asnumpy(bad), np.asarray([0, 0, 1], dtype=np.uint8))
