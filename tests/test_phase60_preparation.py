from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from choiceforge import phase60_preparation as fast


@pytest.mark.parametrize("seed", [0,17,991])
def test_pair_keys_preserve_exact_sorted_int16_topology(seed):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(-32768,32768,(10000,2),dtype=np.int16)
    pairs = np.concatenate([pairs, pairs[:300], np.array([[-32768,32767],[32767,-32768]],np.int16)])
    expected, inverse = np.unique(pairs, axis=0, return_inverse=True)
    actual, actual_inverse = fast.pair_slots(*pairs.T)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_inverse, inverse)


@pytest.mark.parametrize("bad", [None,"missing","inconsistent","negative"])
def test_slot_values_validates_complete_mapping(bad):
    values, slots = np.array([8,4,8,6,4]), np.array([2,0,2,1,0])
    if bad == "missing": slots[:] = 0
    if bad == "inconsistent": values[-1] = 5
    if bad == "negative": slots[-1] = -1
    if bad:
        with pytest.raises(ValueError): fast.slot_values(values, slots, 3, "example")
    else:
        np.testing.assert_array_equal(fast.slot_values(values, slots, 3, "example"), [4,6,8])


def test_period_labels_and_unknown_rejection():
    np.testing.assert_array_equal(fast.period_positions(["EV","EA","MD","PM","AM"]), [4,0,2,3,1])
    with pytest.raises(ValueError): fast.period_positions(["night"])


@pytest.mark.parametrize("size", [0,1,1000])
@pytest.mark.parametrize("fmt", ["{left:,.2f} - {right:,.2f}","{rank}","{mid}","{rank:02d}"])
def test_report_bin_labels_exact_upstream(size, fmt):
    from activitysim.abm.models.summarize import construct_bin_labels
    bins = pd.Series(pd.cut(np.arange(size)%15, bins=[-1,2,6,20]))
    try:
        expected = construct_bin_labels(bins, fmt)
    except (TypeError,ValueError) as error:
        with pytest.raises(type(error)): fast.bin_labels(construct_bin_labels, bins, fmt)
    else:
        pd.testing.assert_series_equal(fast.bin_labels(construct_bin_labels, bins, fmt), expected)


@pytest.mark.parametrize("size", [1,2,3,4,5,7])
@pytest.mark.parametrize("joint", [False, True])
def test_cdap_live_rule_construction_exact(size, joint):
    from activitysim.abm.models.util import cdap
    raw = pd.DataFrame({"activity":["H","M","N","M","N","H","H"],
                        "interaction_ptypes":["11","11","12","123","***","*****","11"],
                        "coefficient":[.3,-.4,1.2,4.,2.,-3.,.7]})
    coeff = cdap.preprocess_interaction_coefficients(raw)
    state = SimpleNamespace(get_injectable=lambda *a:None)
    expected = cdap.build_cdap_spec(state, coeff, size, cache=False, joint_tour_alt=joint)
    actual = fast.build_cdap_spec(state, coeff, size, cache=False, joint_tour_alt=joint)
    pd.testing.assert_frame_equal(actual, expected)


def test_context_restores_hooks_and_thread_mask_after_failure():
    import numba
    from choiceforge import raw_table_input_generation as raw
    initial = numba.get_num_threads()
    events = []
    with pytest.raises(RuntimeError):
        with fast.for_step("non_mandatory_tour_frequency", events, threads=1):
            assert raw._PREPARATION_FAST and numba.get_num_threads() == 1
            raise RuntimeError("test")
    assert not raw._PREPARATION_FAST and numba.get_num_threads() == initial
    assert len(events) == 1


def test_fast_timetable_still_matches_generic(monkeypatch):
    from choiceforge import activitysim_scheduling
    from test_activitysim_scheduling import test_vectorized_timetable_primitives_match_generic_activitysim_semantics, test_optimized_timetable_infers_joint_tour_row_owner
    monkeypatch.setattr(activitysim_scheduling, "_PREPARATION_FAST", True)
    test_vectorized_timetable_primitives_match_generic_activitysim_semantics()
    test_optimized_timetable_infers_joint_tour_row_owner()


@pytest.mark.parametrize("fail",[False,True])
def test_frequency_control_checks_bytes_and_restores_threads(fail):
    import numba
    from choiceforge.phase60_frequency_control import MeasuredDispatcher
    previous = numba.get_num_threads()
    records = []
    def calculation(values):
        return values + (numba.get_num_threads() if fail else 1.)
    wrapped = MeasuredDispatcher(calculation, records)
    if fail and numba.config.NUMBA_NUM_THREADS > 1:
        with pytest.raises(AssertionError): wrapped(np.array([1.,2.]))
        assert not records
    else:
        np.testing.assert_array_equal(wrapped(np.array([1.,2.])),[2.,3.])
        assert records[0]["utilities_byte_exact_all_masks"]
    assert numba.get_num_threads() == previous
