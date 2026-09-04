import numpy as np
import pandas as pd
import pytest

from choiceforge.destination_input_supergraph import (
    DestinationInputSupergraph,
    _owner_topology,
    _period_positions,
    _stable_owner,
    _time_state,
    _wait_table,
)


class _Settings:
    IN_PERIOD = {"work": 18}
    OUT_PERIOD = {"work": 8}


class _Los:
    @staticmethod
    def skim_time_period_label(values, as_cat=True, broadcast_to=None):
        del as_cat
        labels = np.asarray(["EA", "AM", "MD", "PM", "EV"])
        if broadcast_to is not None:
            return pd.Series(labels[int(values) // 5], index=broadcast_to)
        values = np.asarray(values)
        return labels[np.minimum(values // 5, 4)]


def _constants():
    return {
        "min_waitTime": 0,
        "max_waitTime": 50,
        "Taxi_waitTime_mean": {1: 5.5, 2: 9.5, 3: 13.3, 4: 17.3, 5: 26.5},
        "Taxi_waitTime_sd": {key: 0 for key in range(1, 6)},
        "TNC_single_waitTime_mean": {1: 3.0, 2: 6.3, 3: 8.4, 4: 8.5, 5: 10.3},
        "TNC_single_waitTime_sd": {key: 0 for key in range(1, 6)},
        "TNC_shared_waitTime_mean": {1: 5.0, 2: 8.0, 3: 11.0, 4: 15.0, 5: 15.0},
        "TNC_shared_waitTime_sd": {key: 0 for key in range(1, 6)},
    }


def test_phase50_owner_topology_is_compact_and_fail_closed():
    ids, starts, offsets = _owner_topology([10, 10, 20, 20, 20, 30])
    np.testing.assert_array_equal(ids, [10, 10, 20, 20, 20, 30])
    np.testing.assert_array_equal(starts, [0, 2, 5])
    np.testing.assert_array_equal(offsets, [0, 2, 5, 6])
    np.testing.assert_array_equal(
        _stable_owner([4, 4, 7, 7, 7, 9], starts, offsets, "x"), [4, 7, 9]
    )
    with pytest.raises(ValueError, match="not contiguous"):
        _owner_topology([10, 20, 10])
    with pytest.raises(ValueError, match="varies inside"):
        _stable_owner([4, 5, 7, 7, 7, 9], starts, offsets, "x")


def test_phase50_period_and_duration_contract_matches_public_defaults():
    choosers = pd.DataFrame(index=pd.Index([1, 1, 2], name="tour_id"))
    outgoing, incoming, duration = _time_state(
        choosers,
        _Settings(),
        _Los(),
        "work",
        in_period_col=None,
        out_period_col=None,
        duration_col=None,
    )
    np.testing.assert_array_equal(outgoing, [1, 1, 1])
    np.testing.assert_array_equal(incoming, [3, 3, 3])
    np.testing.assert_array_equal(duration, [10, 10, 10])
    np.testing.assert_array_equal(_period_positions(["EA", "AM", "MD", "PM", "EV"]), range(5))
    with pytest.raises(ValueError, match="no public skim-period"):
        _period_positions(["overnight"])


def test_phase50_wait_table_reconstructs_owner_by_destination_band():
    # Density measure 100 maps to band 5, while 20,000 maps to band 1.
    land_use = pd.DataFrame(
        {"TOTPOP": [100.0, 20_000.0], "TOTEMP": [0.0, 0.0], "TOTACRE": [640.0, 640.0]},
        index=[0, 1],
    )
    table = _wait_table(land_use, np.array([0, 1]), np.zeros((2, 6)), _constants())
    assert table.shape == (2, 5, 3)
    # With zero standard deviations, each total is exactly origin mean + destination mean.
    expected = np.asarray([32.0, 13.3, 20.0], dtype=np.float32)
    np.testing.assert_array_equal(table[0, 0], expected)
    np.testing.assert_array_equal(table[1, 4], expected)


def test_phase50_summary_preserves_accounting_and_exact_abi_gate():
    runtime = DestinationInputSupergraph(None, cbd_threshold=3, cp=object())
    runtime._events = [
        {
            "trace_label": "school_location.i1.logsums.university",
            "rows": 100,
            "owners": 5,
            "dense_preprocessor_rows_avoided": 100,
            "dense_preprocessor_values_avoided": 4_100,
            "dense_host_pack_bytes_avoided": 41_600,
            "compact_upload_bytes": 1_000,
            "net_upload_bytes_avoided": 40_600,
            "binding_resolution_calls": 0,
            "host_dense_pack_calls": 0,
            "fallback_used": False,
            "device_generate_seconds": 0.01,
            "utility_kernel_seconds": 0.02,
            "total_seconds": 0.04,
            "float_row_sources": 10,
            "int_row_sources": 31,
            "skim_coordinate_groups": 6,
        }
    ]
    summary = runtime.summary()
    assert summary["contract_version"] == 1
    assert summary["dense_preprocessor_values_avoided"] == 4_100
    assert summary["net_upload_bytes_avoided"] == 40_600
    assert summary["all_source_abis_exact"] is True
    assert summary["fallback_calls"] == 0
