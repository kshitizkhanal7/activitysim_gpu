import numpy as np
import pytest

from choiceforge.cuda_backend import cuda_available, _cupy
from choiceforge.gpu_native import GpuOnlyViolation
from choiceforge.gpu_scheduling_pipeline import (
    CompiledCpuSchedulingPreparer,
    CpuSchedulingPreparer,
    GpuSchedulingPreparer,
    build_tdd_footprints,
    compress_mode_choice_logsums,
    mode_logsum_slots,
    skim_period_code,
)


def test_tdd_footprints_and_period_boundaries_match_activitysim_codes():
    alternatives = np.array(
        [[5, 5, 0], [5, 6, 1], [6, 9, 3], [10, 14, 4], [15, 18, 3], [19, 23, 4]],
        dtype=np.float32,
    )
    footprints, first = build_tdd_footprints(alternatives)
    assert first == 4
    np.testing.assert_array_equal(skim_period_code([5, 6, 9, 10, 14, 15, 18, 19]), [0, 1, 1, 2, 2, 3, 3, 4])
    assert footprints[0, 1] == 6
    assert footprints[1, 1] == 2
    assert footprints[1, 2] == 4
    assert footprints[2, 2] == 2
    assert footprints[2, 3:5].tolist() == [7, 7]
    assert footprints[2, 5] == 4


def test_logsum_factorization_is_bit_exact_and_fails_closed():
    alternatives = np.array(
        [[6, 6, 0], [6, 7, 1], [6, 9, 3], [10, 14, 4]], dtype=np.float32
    )
    ids = np.array([0, 1, 2, 3, 0, 1], dtype=np.int16)
    offsets = np.array([0, 4, 6], dtype=np.int64)
    rows = np.zeros((6, 8), dtype=np.float32)
    # The first three alternatives share the AM->AM slot and must share a logsum.
    rows[:, 0] = [1.25, 1.25, 1.25, 3.5, -4.0, -4.0]
    cache, present = compress_mode_choice_logsums(offsets, ids, alternatives, rows)
    slots = mode_logsum_slots(alternatives, ids)
    assert cache[0, slots[0]].view(np.uint32) == np.float32(1.25).view(np.uint32)
    assert int(present.sum()) == 3

    rows[1, 0] = np.nextafter(np.float32(1.25), np.float32(2.0))
    with pytest.raises(ValueError, match="not constant"):
        compress_mode_choice_logsums(offsets, ids, alternatives, rows)


def _small_inputs():
    alternatives = np.array(
        [[5, 5, 0], [5, 6, 1], [6, 7, 1], [7, 8, 1]], dtype=np.float32
    )
    people = np.array([0, 1], dtype=np.int32)
    chooser = np.array(
        [
            [999, 2, 1],
            [999, 1, 1],
        ],
        dtype=np.float32,
    )
    cache = np.arange(50, dtype=np.float32).reshape(2, 25)
    return alternatives, people, chooser, cache


def test_cpu_preparer_builds_feasible_rows_and_updates_previous_end():
    alternatives, people, chooser, cache = _small_inputs()
    preparer = CpuSchedulingPreparer(2, alternatives)
    first = preparer.prepare(
        people,
        chooser,
        cache,
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    np.testing.assert_array_equal(first.offsets, [0, 4, 8])
    np.testing.assert_array_equal(first.chooser_values[:, 0], [5, 5])
    # Assign 5--6 to person zero. Starting at its end is allowed. ActivitySim
    # also allows a zero-duration start/end marker at the occupied start.
    preparer.assign(np.array([0]), np.array([1]))
    second = preparer.prepare(
        np.array([0], dtype=np.int32),
        np.array([[999, 2, 2]], dtype=np.float32),
        cache[:1],
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    np.testing.assert_array_equal(second.alternative_ids, [0, 2, 3])
    assert second.chooser_values[0, 0] == 6
    assert second.row_values[1, 1] == 1  # previous tour ends at this departure


def test_compiled_cpu_preparer_matches_readable_reference():
    alternatives, people, chooser, cache = _small_inputs()
    readable = CpuSchedulingPreparer(2, alternatives)
    compiled = CompiledCpuSchedulingPreparer(2, alternatives)
    kwargs = dict(end_previous_column=0, tour_count_column=1, tour_num_column=2)
    expected = readable.prepare(people, chooser, cache, **kwargs)
    actual = compiled.prepare(people, chooser, cache, **kwargs)
    for name in ("chooser_values", "row_values", "alternative_ids", "offsets", "row_owners"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    selected = np.array([1, 3], dtype=np.int16)
    readable.assign(people, selected)
    compiled.assign(people, selected)
    expected = readable.prepare(people[:1], chooser[:1], cache[:1], **kwargs)
    actual = compiled.prepare(people[:1], chooser[:1], cache[:1], **kwargs)
    for name in ("chooser_values", "row_values", "alternative_ids", "offsets", "row_owners"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_gpu_preparer_matches_cpu_and_rejects_host_inputs():
    cp = _cupy()
    alternatives, people, chooser, cache = _small_inputs()
    cpu = CpuSchedulingPreparer(2, alternatives)
    gpu = GpuSchedulingPreparer(2, cp.asarray(alternatives))

    expected = cpu.prepare(
        people,
        chooser,
        cache,
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    actual = gpu.prepare(
        cp.asarray(people),
        cp.asarray(chooser),
        cp.asarray(cache),
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    for name in ("chooser_values", "row_values", "alternative_ids", "offsets", "row_owners"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(actual, name)), getattr(expected, name))

    cpu.assign(people, np.array([1, 3], dtype=np.int16))
    gpu.assign(cp.asarray(people), cp.asarray([1, 3], dtype=cp.int16))
    expected = cpu.prepare(
        people[:1],
        np.array([[999, 2, 2]], dtype=np.float32),
        cache[:1],
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    actual = gpu.prepare(
        cp.asarray(people[:1]),
        cp.asarray([[999, 2, 2]], dtype=cp.float32),
        cp.asarray(cache[:1]),
        end_previous_column=0,
        tour_count_column=1,
        tour_num_column=2,
    )
    np.testing.assert_array_equal(cp.asnumpy(actual.alternative_ids), expected.alternative_ids)
    np.testing.assert_array_equal(cp.asnumpy(actual.row_values), expected.row_values)

    with pytest.raises(GpuOnlyViolation):
        gpu.prepare(
            people,
            cp.asarray(chooser),
            cp.asarray(cache),
            end_previous_column=0,
            tour_count_column=1,
            tour_num_column=2,
        )
