import numpy as np
import pandas as pd
import pytest

from choiceforge.raw_table_input_generation import (
    _SEMANTIC_INT,
    _input_declarations,
    _owner_sources,
)
from choiceforge.semantic_input_generation import _availability_expression


def _raw_fixture():
    land = pd.DataFrame(
        {
            "TOTPOP": [1000, 5000],
            "TOTEMP": [2000, 9000],
            "TOTACRE": [640, 1280],
            "PRKCST": [2.5, 7.0],
            "area_type": [1, 4],
            "TOPOLOGY": [3, 2],
            "TERMINAL": [4.0, 9.0],
            "density_index": [1.25, 8.5],
        },
        index=pd.Index([1, 2], name="zone_id"),
    )
    tours = pd.DataFrame(
        {
            "home_zone_id": [1, 2],
            "workplace_zone_id": [2, 1],
            "value_of_time": [2.0, 4.0],
            "tour_type": ["work", "work"],
            "tour_category": ["mandatory", "mandatory"],
            "number_of_participants": [1, 1],
            "free_parking_at_work": [True, False],
            "auto_ownership": [0, 2],
            "age": [19, 51],
            "hhsize": [1, 4],
            "num_workers": [1, 2],
            "density_index": [1.25, 8.5],
        },
        index=pd.Index([101, 202], name="tour_id"),
    )
    means = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0}
    zeros = {key: 0.0 for key in means}
    constants = {
        "shortWalk": 0.333,
        "walkSpeed": 3.0,
        "min_waitTime": 0.0,
        "max_waitTime": 50.0,
        "Taxi_waitTime_mean": means,
        "Taxi_waitTime_sd": zeros,
        "TNC_single_waitTime_mean": means,
        "TNC_single_waitTime_sd": zeros,
        "TNC_shared_waitTime_mean": means,
        "TNC_shared_waitTime_sd": zeros,
    }
    return {
        "tours": tours,
        "land_use": land,
        "tour_purpose": "work",
        "constants": constants,
        "cbd_threshold": 2,
        "standard_normal_draws": np.zeros((2, 6)),
    }


def test_raw_owner_sources_use_land_use_and_free_parking_directly():
    raw = _raw_fixture()
    values, origin, destination, parking = _owner_sources(
        raw, np.array([101, 202]), raw["constants"]
    )
    assert np.array_equal(origin, [1, 2])
    assert np.array_equal(destination, [2, 1])
    assert np.array_equal(parking, [0.0, 2.5])
    assert np.array_equal(values["column:terminal_time"], [9.0, 4.0])
    assert np.array_equal(values["column:ivot"], [0.5, 0.25])
    assert np.array_equal(values["column:destination_in_cbd"], [False, True])


def test_raw_source_registry_fails_closed():
    with pytest.raises(ValueError, match="no raw-table formula"):
        _input_declarations(("column:unknown",), {}, set(), 3)


def test_all_eighteen_availability_rules_have_cuda_expressions():
    for label in _SEMANTIC_INT:
        required = []
        expression = _availability_expression(
            label, lambda source: required.append(source) or "skim", 0
        )
        assert expression
        assert required
    assert len(_SEMANTIC_INT) == 18
