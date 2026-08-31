import numpy as np
import pandas as pd
import pytest

from choiceforge.trip_destination_final import (
    SUPPORTED_EXPRESSIONS,
    _pad_ragged_f32,
    narrow_interaction_frame,
    ragged_offsets,
    validate_final_spec,
)


def test_phase44_validates_reviewed_expression_abi():
    spec = pd.DataFrame({"work": np.ones(16)}, index=SUPPORTED_EXPRESSIONS)
    validate_final_spec(spec)
    changed = spec.rename(index={SUPPORTED_EXPRESSIONS[-1]: "changed"})
    with pytest.raises(ValueError, match="expression ABI"):
        validate_final_spec(changed)


def test_phase44_ragged_offsets_require_contiguous_chooser_order():
    choosers = pd.DataFrame(index=pd.Index([10, 20], name="trip_id"))
    alternatives = pd.DataFrame(index=pd.Index([10, 10, 20], name="trip_id"))
    offsets, counts = ragged_offsets(alternatives, choosers)
    np.testing.assert_array_equal(offsets, [0, 2, 3])
    np.testing.assert_array_equal(counts, [2, 1])
    bad = alternatives.iloc[[0, 2, 1]]
    with pytest.raises(ValueError, match="contiguous"):
        ragged_offsets(bad, choosers)


def test_phase44_narrow_frame_preserves_categorical_purpose_and_row_order():
    index = pd.Index([10, 10, 20], name="trip_id")
    alternatives = pd.DataFrame(
        {
            "dest_taz": [1, 2, 3],
            "prob": [0.2, 0.8, 1.0],
            "pick_count": [1, 2, 1],
            "od_logsum": [1.0, 2.0, 3.0],
            "dp_logsum": [4.0, 5.0, 6.0],
            "unused": [99, 99, 99],
        },
        index=index,
    )
    choosers = pd.DataFrame(
        {
            "origin": [7, 8],
            "tour_leg_dest": [9, 10],
            "trip_period": pd.Categorical(["AM", "PM"]),
            "is_joint": [False, True],
            "outbound": [True, False],
            "purpose": pd.Categorical(["work", "school"]),
            "purpose_index_num": [0, 1],
            "unused": [1, 2],
        },
        index=pd.Index([10, 20], name="trip_id"),
    )
    frame = narrow_interaction_frame(alternatives, choosers, np.array([2, 1]))
    assert list(frame.columns) == [
        "dest_taz",
        "prob",
        "pick_count",
        "od_logsum",
        "dp_logsum",
        "origin",
        "tour_leg_dest",
        "trip_period",
        "is_joint",
        "outbound",
        "purpose",
        "purpose_index_num",
    ]
    assert list(frame["origin"]) == [7, 7, 8]
    assert list(frame["purpose"].astype(str)) == ["work", "work", "school"]
    assert isinstance(frame["purpose"].dtype, pd.CategoricalDtype)


def test_phase44_numba_padding_matches_activitysim_dummy_contract():
    padded = _pad_ragged_f32(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([0, 2, 3], dtype=np.int64),
        2,
    )
    np.testing.assert_array_equal(
        padded, np.asarray([[1.0, 2.0], [3.0, -999.0]], dtype=np.float32)
    )
