import numpy as np
import pandas as pd
import pytest

from choiceforge.destination_supergraph import DestinationSupergraphBridge


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.ndim = self.values.ndim
        self.shape = self.values.shape
        self.nbytes = self.values.nbytes

    def astype(self, dtype, copy=True):
        return _FakeArray(self.values.astype(dtype, copy=copy))

    def __getitem__(self, key):
        return _FakeArray(self.values[key])


class _FakeCp:
    float32 = np.float32

    @staticmethod
    def asarray(values):
        return np.asarray(values)

    @staticmethod
    def asnumpy(values):
        return values.values if isinstance(values, _FakeArray) else np.asarray(values)


def _metadata(ids):
    return {"chooser_ids": np.asarray(ids), "trace_label": "producer"}


def test_phase49_exact_ordered_handoff_and_byte_accounting():
    bridge = DestinationSupergraphBridge(_FakeCp())
    bridge.publish(
        _FakeArray(np.array([1.25, -2.5, 3.75], dtype=np.float64)),
        _metadata([8, 8, 9]),
        host_materialized=False,
    )
    alternatives = pd.DataFrame(index=pd.Index([8, 8, 9]))
    result = bridge.consume(alternatives, consumer_trace="consumer")
    np.testing.assert_array_equal(
        result.values, np.array([1.25, -2.5, 3.75], dtype=np.float32)
    )
    bridge.capture_selected([8, 9], [0, 2], component="tour_destination")
    summary = bridge.summary()
    assert summary["calls"] == 1
    assert summary["rows"] == 3
    assert summary["device_to_host_bytes_avoided"] == 24
    assert summary["host_to_device_bytes_avoided"] == 12
    assert summary["round_trip_bytes_avoided"] == 36
    assert summary["pending_packets"] == 0
    bridge.assert_empty()


def test_phase49_materialized_output_counts_only_upload_avoidance():
    bridge = DestinationSupergraphBridge(_FakeCp())
    bridge.publish(_FakeArray(np.ones(2)), _metadata([1, 2]), host_materialized=True)
    bridge.consume(pd.DataFrame(index=[1, 2]), consumer_trace="consumer")
    bridge.capture_selected([1], [0], component="tour_destination")
    assert bridge.summary()["device_to_host_bytes_avoided"] == 0
    assert bridge.summary()["host_to_device_bytes_avoided"] == 8


def test_phase49_fails_closed_on_missing_or_reordered_rows():
    bridge = DestinationSupergraphBridge(_FakeCp())
    with pytest.raises(RuntimeError, match="no resident logsum"):
        bridge.consume(pd.DataFrame(index=[1]), consumer_trace="consumer")
    bridge.publish(_FakeArray(np.ones(2)), _metadata([1, 2]), host_materialized=False)
    with pytest.raises(ValueError, match="order differs"):
        bridge.consume(pd.DataFrame(index=[2, 1]), consumer_trace="consumer")


def test_phase49_fails_closed_on_invalid_shape_and_unconsumed_packet():
    bridge = DestinationSupergraphBridge(_FakeCp())
    with pytest.raises(ValueError, match="shape differs"):
        bridge.publish(_FakeArray(np.ones((2, 1))), _metadata([1, 2]), host_materialized=False)
    bridge.publish(_FakeArray(np.ones(1)), _metadata([1]), host_materialized=False)
    with pytest.raises(RuntimeError, match="unconsumed"):
        bridge.assert_empty()


def test_phase49_materializes_only_selected_location_logsums_and_restores_order():
    bridge = DestinationSupergraphBridge(_FakeCp())
    bridge.publish(
        _FakeArray(np.array([10.25, 20.5, 30.75], dtype=np.float64)),
        _metadata([7, 7, 9]),
        host_materialized=False,
    )
    bridge.consume(pd.DataFrame(index=[7, 7, 9]), consumer_trace="consumer")
    bridge.capture_selected([7, 9], [1, 2], component="school_location")
    np.testing.assert_array_equal(
        bridge.consume_selected(pd.Index([9, 7])),
        np.array([30.75, 20.5], dtype=np.float64),
    )
    summary = bridge.summary()
    assert summary["source_device_to_host_bytes_eliminated"] == 24
    assert summary["selected_output_device_to_host_bytes"] == 16
    assert summary["device_to_host_bytes_avoided"] == 8
    assert summary["pending_selected_outputs"] == 0


def test_phase49_fails_closed_on_unconsumed_selected_output():
    bridge = DestinationSupergraphBridge(_FakeCp())
    bridge.publish(
        _FakeArray(np.array([10.25], dtype=np.float64)),
        _metadata([7]),
        host_materialized=False,
    )
    bridge.consume(pd.DataFrame(index=[7]), consumer_trace="consumer")
    bridge.capture_selected([7], [0], component="school_location")
    with pytest.raises(RuntimeError, match="selected-output"):
        bridge.assert_empty()
