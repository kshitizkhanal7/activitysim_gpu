import pandas as pd
import numpy as np
import pytest

from choiceforge.activitysim_destination import (
    _candidate_sink_metadata,
    _phase43_compact_draws_for_bundles,
    _phase43_compact_directional_draws,
    phase43_runtime_telemetry,
    reset_trip_destination_stage_telemetry,
)


def test_destination_candidate_does_not_read_scheduling_columns_without_sink():
    choosers = pd.DataFrame({"trip_id": [1, 2]}, index=[10, 11])

    assert _candidate_sink_metadata(choosers, "trip_destination", required=False) == {}


def test_scheduling_candidate_captures_device_sink_identity():
    choosers = pd.DataFrame(
        {
            "start": [5, 10],
            "end": [9, 15],
            "out_period": ["EA", "AM"],
            "in_period": ["AM", "PM"],
        },
        index=[101, 102],
    )

    metadata = _candidate_sink_metadata(
        choosers, "mandatory_tour_scheduling", required=True
    )

    assert metadata["chooser_ids"].tolist() == [101, 102]
    assert metadata["start"].tolist() == [5, 10]
    assert metadata["end"].tolist() == [9, 15]
    assert metadata["out_period"].tolist() == ["EA", "AM"]
    assert metadata["in_period"].tolist() == ["AM", "PM"]


def test_destination_device_sink_captures_only_repeated_row_identity():
    choosers = pd.DataFrame({"zone_id": [4, 9, 3]}, index=[10, 10, 11])

    metadata = _candidate_sink_metadata(
        choosers, "school_location.logsums", required=True
    )

    assert set(metadata) == {"chooser_ids"}
    assert metadata["chooser_ids"].tolist() == [10, 10, 11]


def test_device_sink_rejects_partial_scheduling_coordinates():
    choosers = pd.DataFrame({"start": [5]}, index=[10])
    with pytest.raises(ValueError, match="partial scheduling-coordinate"):
        _candidate_sink_metadata(
            choosers, "mandatory_tour_scheduling.changed_abi", required=True
        )


def test_phase43_generates_unique_directional_draw_state_without_expansion():
    class FakeRng:
        def __init__(self):
            self.calls = 0

        def normal_for_df(self, frame, broadcast, size):
            assert broadcast is False
            assert size == 3
            values = np.arange(len(frame) * size, dtype=np.float64).reshape(-1, size)
            values += self.calls * 100
            self.calls += 1
            return values

    class FakeState:
        def __init__(self):
            self.rng = FakeRng()

        def get_rn_generator(self):
            return self.rng

    reset_trip_destination_stage_telemetry()
    index = pd.Index([10, 10, 10, 20, 20], name="trip_id")
    bundle = {
        "destination_sample": pd.DataFrame(index=index),
        "trips": pd.DataFrame(index=pd.Index([10, 20], name="trip_id")),
    }
    draws = _phase43_compact_directional_draws(FakeState(), bundle, 3)
    np.testing.assert_array_equal(
        draws,
        np.asarray(
            [[0, 1, 2], [3, 4, 5], [100, 101, 102], [103, 104, 105]],
            dtype=np.float64,
        ),
    )
    assert phase43_runtime_telemetry() == {
        "compact_draw_rows": 4,
        "expanded_draw_rows_avoided": 6,
        "rng_calls": 2,
        "normal_draws_per_row": 3,
        "choice_draw_rows": 0,
        "choice_draws_consumed": 0,
    }


def test_phase43_batches_final_choice_draws_on_unique_trip_rows():
    class FakeRng:
        def __init__(self):
            self.normal_calls = 0
            self.random_calls = 0

        def normal_for_df(self, frame, broadcast, size):
            self.normal_calls += 1
            return np.zeros((len(frame), size), dtype=np.float64)

        def random_for_df(self, frame):
            self.random_calls += 1
            return np.arange(len(frame), dtype=np.float64).reshape(-1, 1) / 10

    class FakeState:
        def __init__(self):
            self.rng = FakeRng()

        def get_rn_generator(self):
            return self.rng

    reset_trip_destination_stage_telemetry()
    bundles = [
        {
            "destination_sample": pd.DataFrame(
                index=pd.Index([10, 10, 20], name="trip_id")
            ),
            "trips": pd.DataFrame(index=pd.Index([10, 20], name="trip_id")),
        },
        {
            "destination_sample": pd.DataFrame(
                index=pd.Index([30, 30], name="trip_id")
            ),
            "trips": pd.DataFrame(index=pd.Index([30], name="trip_id")),
        },
    ]
    state = FakeState()
    _phase43_compact_draws_for_bundles(
        state, bundles, 3, include_choice_draws=True
    )
    np.testing.assert_array_equal(bundles[0]["compact_choice_draws"], [0.0, 0.1])
    np.testing.assert_array_equal(bundles[1]["compact_choice_draws"], [0.2])
    assert state.rng.normal_calls == 2
    assert state.rng.random_calls == 1
    assert phase43_runtime_telemetry() == {
        "compact_draw_rows": 6,
        "expanded_draw_rows_avoided": 4,
        "rng_calls": 3,
        "normal_draws_per_row": 3,
        "choice_draw_rows": 3,
        "choice_draws_consumed": 0,
    }
