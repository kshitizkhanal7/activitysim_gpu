import numpy as np
import pandas as pd

from choiceforge.raw_table_input_generation import RAW_FLOAT_SOURCES, RAW_INT_SOURCES
from choiceforge.trip_logsum_native import (
    _RAW_FLOAT_COLUMNS,
    _RAW_INT_COLUMNS,
    _ROW_COORD_COLUMNS,
    _STABLE_INT_COLUMNS,
    TripLogsumNativePlan,
    _normalized_row_layout,
    _period,
    _wait,
)


def test_phase35_native_source_contract_is_declared():
    assert "column:total_terminal_time" in RAW_FLOAT_SOURCES
    assert "column:origTaxiWaitTime" in RAW_FLOAT_SOURCES
    assert "column:trip_topology" in RAW_INT_SOURCES
    assert "name:drive_lrf_available_inbound" in RAW_INT_SOURCES


def test_phase35_period_contract_accepts_labels_and_ordinals():
    expected = np.arange(5, dtype=np.int64)
    np.testing.assert_array_equal(_period(["EA", "AM", "MD", "PM", "EV"]), expected)
    np.testing.assert_array_equal(_period(expected), expected)


def test_phase35_scaled_lognormal_zero_sigma_is_mean():
    draws = np.array([-2.0, 0.0, 3.0])
    means = np.array([3.0, 6.3, 8.4])
    np.testing.assert_allclose(
        _wait(draws, means, np.zeros(3), 0.0, 50.0), means, rtol=0, atol=1e-14
    )


def test_phase36_compact_packet_removes_more_than_one_gigabyte_at_full_scale():
    rows = 4_188_312
    compact = rows * (
        len(_RAW_INT_COLUMNS) * 4 + len(_RAW_FLOAT_COLUMNS) * 8 + 3 * 4
    )
    former = rows * (11 * 4 + 45 * 8 + 3 * 8)
    assert compact == 351_818_208
    assert former == 1_792_597_536
    assert former - compact == 1_440_779_328


def test_phase36_mode_masks_and_int32_guard_fail_closed():
    assert int(TripLogsumNativePlan._mode_mask([0, 2, 5])) == 0b100101
    np.testing.assert_array_equal(
        TripLogsumNativePlan._checked_int32([0, 2**31 - 1], "ok"),
        np.array([0, 2**31 - 1], dtype=np.int32),
    )
    with np.testing.assert_raises(ValueError):
        TripLogsumNativePlan._checked_int32([2**31], "overflow")
    with np.testing.assert_raises(ValueError):
        TripLogsumNativePlan._mode_mask([63])


def test_phase38_normalizes_repeated_candidates_without_merging_directional_draws():
    frame = pd.DataFrame(
        {"age": [31, 31, 52, 31, 31, 52]},
        index=pd.Index([10, 10, 20, 10, 10, 20], name="trip_id"),
    )
    draws = np.asarray(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [10, 11, 12]],
        dtype=np.float64,
    )
    first, selectors, unique_trips = _normalized_row_layout(
        frame, draws, ("age",)
    )
    np.testing.assert_array_equal(first, [0, 2, 3, 5])
    np.testing.assert_array_equal(selectors, [0, 0, 1, 2, 2, 3])
    assert unique_trips == 2
    assert len(_ROW_COORD_COLUMNS) * 4 + 4 == 12
    assert len(_STABLE_INT_COLUMNS) * 4 + len(_RAW_FLOAT_COLUMNS) * 8 + 3 * 4 == 76


def test_phase38_normalization_fails_closed_on_unstable_facts_or_draws():
    frame = pd.DataFrame(
        {"age": [31, 32, 31, 31]},
        index=pd.Index([10, 10, 10, 10], name="trip_id"),
    )
    draws = np.ones((4, 3), dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "stable column"):
        _normalized_row_layout(frame, draws, ("age",))

    frame["age"] = 31
    draws[1, 0] = 2
    with np.testing.assert_raises_regex(ValueError, "controlled wait draws"):
        _normalized_row_layout(frame, draws, ("age",))
