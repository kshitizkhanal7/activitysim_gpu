import pandas as pd

from choiceforge.activitysim_destination import _candidate_sink_metadata


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
