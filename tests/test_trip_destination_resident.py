import numpy as np
import pandas as pd

from choiceforge.trip_destination_resident import (
    _host_duplicate_contract,
    _pack_sample,
    _preserved_order_choices,
)


def test_phase40_duplicate_contract_keeps_first_draw_and_counts_all_picks():
    choices = np.asarray([[2, 1, 2, 2], [0, 0, 1, 1]], dtype=np.int32)
    first, counts = _host_duplicate_contract(choices)
    np.testing.assert_array_equal(first, [[1, 1, 0, 0], [1, 0, 1, 0]])
    np.testing.assert_array_equal(counts, [[3, 1, 3, 3], [2, 2, 2, 2]])
    assert first.dtype == np.uint8
    assert counts.dtype == np.uint32


def test_phase40_pack_sample_matches_activitysim_sort_and_narrow_contract():
    trips = pd.DataFrame(index=pd.Index([20, 10], name="trip_id"))
    choices = np.asarray([[2, 1, 2], [1, 0, 0]], dtype=np.int32)
    probabilities = np.asarray(
        [[0.2, 0.5, 0.2], [0.4, 0.3, 0.3]], dtype=np.float32
    )
    random_draws = np.asarray([[0.8, 0.1, 0.9], [0.6, 0.1, 0.2]])
    first, counts = _host_duplicate_contract(choices)
    result = _pack_sample(
        trips,
        choices,
        probabilities,
        random_draws,
        first,
        counts,
        "dest_taz",
    )
    assert result.index.tolist() == [10, 10, 20, 20]
    assert result["dest_taz"].tolist() == [0, 1, 1, 2]
    assert result["pick_count"].tolist() == [2, 1, 1, 2]
    assert result["prob"].dtype == np.float32
    assert result["pick_count"].dtype == np.uint32


def test_phase40_preserved_order_chooser_matches_activitysim():
    from activitysim.core.choosing import sample_choices_maker_preserve_ordering

    probabilities = np.asarray(
        [[0.05, 0.15, 0.8], [0.4, 0.3, 0.3]], dtype=np.float32
    )
    random_draws = np.asarray([[0.99, 0.01, 0.2], [0.4, 0.0, 0.999999]])
    alternatives = np.asarray([0, 1, 2], dtype=np.int32)
    expected_choices, expected_probs = sample_choices_maker_preserve_ordering(
        probabilities, random_draws, alternatives
    )
    choices, choice_probs = _preserved_order_choices(
        probabilities, random_draws, alternatives
    )
    np.testing.assert_array_equal(choices, expected_choices.T)
    np.testing.assert_array_equal(choice_probs, expected_probs.T)
