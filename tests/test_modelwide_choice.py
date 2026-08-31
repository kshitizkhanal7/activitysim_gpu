import numpy as np
import pandas as pd

from choiceforge.modelwide_choice import (
    compact_interaction_frame,
    summarize_telemetry,
)
from choiceforge.modelwide_sampling import _feature_source


def test_compact_interaction_frame_matches_pandas_join_contract():
    index = pd.Index([10, 10, 20], name="chooser_id")
    alternatives = pd.DataFrame(
        {
            "choice": [1, 2, 3],
            "shared": [100, 200, 300],
            "category": pd.Categorical(["a", "b", "a"]),
        },
        index=index,
    )
    choosers = pd.DataFrame(
        {
            "income": [7, 9],
            "shared": [11, 22],
            "purpose": pd.Categorical(["work", "school"]),
        },
        index=pd.Index([10, 20], name="chooser_id"),
    )
    expected = alternatives.join(choosers, how="left", rsuffix="_chooser")
    actual = compact_interaction_frame(
        alternatives, choosers, np.asarray([2, 1], dtype=np.int64)
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_phase45_telemetry_groups_components():
    events = [
        {
            "component": "school_location",
            "chooser_rows": 2,
            "alternative_rows": 40,
            "total_seconds": 0.2,
        },
        {
            "component": "school_location",
            "chooser_rows": 3,
            "alternative_rows": 60,
            "total_seconds": 0.3,
        },
        {
            "component": "workplace_location",
            "chooser_rows": 4,
            "alternative_rows": 80,
            "total_seconds": 0.4,
        },
    ]
    summary = summarize_telemetry(events)
    assert summary["calls"] == 3
    assert summary["chooser_rows"] == 9
    assert summary["alternative_rows"] == 180
    assert summary["groups"]["school_location"] == {
        "calls": 2,
        "chooser_rows": 5,
        "alternative_rows": 100,
        "seconds": 0.5,
    }


def test_phase45_public_sampling_expression_contract():
    expressions = [
        "_DIST@skims['DIST']",
        "@_DIST.clip(0,1)",
        "@(_DIST-1).clip(0,1)",
        "@(_DIST-15.0).clip(0)",
        "@(df['size_term'] * df['shadow_price_size_term_adjustment']).apply(np.log1p)",
        "@df['shadow_price_utility_adjustment']",
        "@df['size_term']==0",
        "@(df['income_segment']>=WORK_HIGH_SEGMENT_ID) * (_DIST-5).clip(0)",
    ]
    generated = [_feature_source(item) for item in expressions]
    assert generated[0] == "0.0f"
    assert generated[4] == "size_log[alternative]"
    assert "shadow_utility" in generated[5]
    assert "income[row]" in generated[7]
