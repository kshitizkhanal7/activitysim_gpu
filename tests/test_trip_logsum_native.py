import numpy as np

from choiceforge.raw_table_input_generation import RAW_FLOAT_SOURCES, RAW_INT_SOURCES
from choiceforge.trip_logsum_native import _period, _wait


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
