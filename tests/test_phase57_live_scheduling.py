import numpy as np
import pandas as pd
import pytest

from choiceforge.phase57_live_scheduling import period_pairs_cuda, representative_times


def test_cuda_pairs_equal_exhaustive_collision_oracle_with_changed_windows():
    pytest.importorskip("cupy")
    rng = np.random.default_rng(571)
    codes = np.array([0, 2, 4, 6, 7], dtype=np.int8)
    windows = rng.choice(codes, size=(47, 11))
    footprints = rng.choice(codes, size=(53, 11))
    footprints[:4] = 0
    slots = rng.integers(0, 25, len(footprints), dtype=np.int32)
    collisions = {(2, 2), (2, 7), (4, 4), (4, 7),
                  (7, 2), (7, 4), (7, 6), (7, 7), (6, 7)}
    for current in (windows, np.zeros_like(windows), np.full_like(windows, 7)):
        expected = np.full((len(current), 25), len(footprints), dtype=np.int32)
        counts = np.zeros(len(current), dtype=np.int32)
        for owner, window in enumerate(current):
            for alt, footprint in enumerate(footprints):
                if not any((int(c), int(w)) in collisions for c, w in zip(footprint, window)):
                    expected[owner, slots[alt]] = min(expected[owner, slots[alt]], alt)
                    counts[owner] += 1
        actual, actual_counts = period_pairs_cuda(current, footprints, slots)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_counts, counts)


def test_representative_times_respond_to_live_configuration_and_preserve_order():
    segments = pd.DataFrame({"time_period": ["EA", "AM", "MD", "PM", "EV"],
                             "start": [3, 6, 10, 15, 19], "end": [5, 8, 13, 17, 23]})
    class State:
        def get_injectable(self, name, default=None):
            return segments
    rows = pd.DataFrame({"out_period": ["AM", "EA", "AM"],
                         "in_period": ["EV", "AM", "PM"]}, index=[11, 11, 19])
    result = representative_times(State(), rows, "work")
    np.testing.assert_array_equal(result.start, [6, 3, 6])
    np.testing.assert_array_equal(result.end, [23, 8, 17])
    np.testing.assert_array_equal(result.index, rows.index)
    segments.loc[segments.time_period == "AM", "start"] = 7
    np.testing.assert_array_equal(representative_times(State(), rows, "work").start, [7, 3, 7])


def test_pair_shape_guard():
    with pytest.raises(ValueError, match="shapes differ"):
        period_pairs_cuda(np.zeros((2, 5)), np.zeros((4, 6)), np.zeros(4))


def test_purpose_specific_times_and_null_fallback():
    segments = pd.DataFrame({"time_period": list("ABCDE")*2,
                             "start": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             "end": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                             "tour_purpose": [None]*5+["work"]*5})
    class State:
        def get_injectable(self, name, default=None):
            return segments
    rows = pd.DataFrame({"out_period": ["B"], "in_period": ["D"]})
    assert representative_times(State(), rows, "work").start.iloc[0] == 7
    assert representative_times(State(), rows, "school").start.iloc[0] == 2
    segments.loc[0, "time_period"] = "B"
    with pytest.raises(ValueError, match="exactly one representative"):
        representative_times(State(), rows, "school")


def test_live_compaction_matches_actual_activitysim_table_and_dedupe(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from activitysim.core.timetable import TimeTable
    from activitysim.abm.models.util import vectorize_tour_scheduling as vts
    from choiceforge.phase57_live_scheduling import representative_rows

    categories = ["EA", "AM", "MD", "PM", "EV"]
    alts = pd.DataFrame([(s, e) for s in range(5, 24) for e in range(s, 24)], columns=["start", "end"])
    alts["duration"] = alts.end-alts.start
    tours = pd.DataFrame({"person_id": [100, 101, 102]}, index=pd.Index([7, 3, 19], name="tour_id"))
    timetable = TimeTable(pd.DataFrame(0, index=[100, 101, 102], columns=[str(p) for p in range(4, 25)], dtype=np.int8), alts)
    segments = pd.DataFrame({"time_period": categories, "start": [3, 6, 10, 15, 19], "end": [5, 8, 13, 17, 23]})
    class Los:
        def skim_time_period_label(self, values, as_cat=True):
            return pd.Series(pd.Categorical.from_codes(np.searchsorted([5, 9, 14, 18], values, side="left"), categories=categories, ordered=True), index=values.index)
    class State:
        def get_injectable(self, name, default=None):
            return {"network_los": Los(), "tdd_alt_segments": segments}.get(name, default)
    state = State()
    monkeypatch.setattr(vts.chunk, "chunk_log", lambda *a, **k: nullcontext(SimpleNamespace(log_df=lambda *a, **k: None)))
    for assignment in (None, [0, 30, 150]):
        if assignment is not None:
            timetable.windows[:] = timetable.tdd_footprints[assignment]
        dense = vts.tdd_interaction_dataset(state, tours, alts, timetable, "tdd", "person_id", "test")
        dense["out_period"] = Los().skim_time_period_label(dense.start)
        dense["in_period"] = Los().skim_time_period_label(dense.end)
        expected, _ = vts.dedupe_alt_tdd(state, dense, "work", "test")
        compact, telemetry = representative_rows(state, tours, alts, timetable, "person_id")
        actual = representative_times(state, compact, "work")
        pd.testing.assert_frame_equal(actual, expected)
        assert telemetry["full_interaction_rows_avoided"] == len(dense)
