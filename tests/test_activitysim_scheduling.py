import numpy as np
import pandas as pd

import choiceforge.activitysim_scheduling as scheduling
from choiceforge.api import ChoiceResult
from choiceforge.activitysim_scheduling import (
    evaluate_stateful_primitives,
    interaction_sample_simulate_choiceforge,
    lower_activitysim_spec,
    pack_compact_scheduling_inputs,
)


def test_lower_and_pack_activitysim_frames_without_wide_join():
    choosers = pd.DataFrame(
        {"x": [2, 3], "category": ["work", "school"]}, index=[10, 20]
    )
    alternatives = pd.DataFrame(
        {
            "tdd": [0, 1, 0, 1],
            "start": [1, 2, 1, 2],
            "duration": [2, 3, 2, 3],
            "mode_choice_logsum": [0.1, 0.2, 0.3, 0.4],
        },
        index=[10, 10, 20, 20],
    )
    spec = pd.DataFrame(
        {"coef": [0.5, -1.0, 0.25]},
        index=[
            "x * start",
            "(category == 'work') & (duration < 3)",
            "mode_choice_logsum",
        ],
    )
    lowered = lower_activitysim_spec(spec, choosers, alternatives)
    packed = pack_compact_scheduling_inputs(lowered, choosers, alternatives)

    assert lowered.expressions[1] == "(category_work) & (duration < 3)"
    assert lowered.chooser_columns == ("category_work", "x")
    np.testing.assert_array_equal(packed.offsets, [0, 2, 4])
    np.testing.assert_array_equal(packed.chooser_values[:, 0], [1, 0])
    np.testing.assert_array_equal(packed.alternative_ids, [0, 1, 0, 1])
    assert packed.row_values.shape == (4, 1)


class _GenericTimetable:
    def __init__(self, windows, person_rows, period_cols):
        self._windows = windows
        self._person_rows = person_rows
        self._period_cols = period_cols

    def _state(self, people, periods):
        return np.array(
            [self._windows[self._person_rows[p], self._period_cols[t]] for p, t in zip(people, periods)]
        )

    def previous_tour_ends(self, people, periods):
        return np.isin(self._state(people, periods), (4, 6))

    def previous_tour_begins(self, people, periods):
        return np.isin(self._state(people, periods), (2, 6))

    def _adjacent(self, people, periods, before):
        result = []
        for person, period in zip(people, periods):
            row = self._windows[self._person_rows[person]]
            column = self._period_cols[period]
            step = -1 if before else 1
            count = 0
            cursor = column + step
            while 0 < cursor < len(row) - 1 and row[cursor] != 7:
                count += 1
                cursor += step
            result.append(count)
        return np.asarray(result)

    def adjacent_window_before(self, people, periods):
        return self._adjacent(people, periods, True)

    def adjacent_window_after(self, people, periods):
        return self._adjacent(people, periods, False)

    def remaining_periods_available(self, people, starts, ends):
        result = []
        for person, start, end in zip(people, starts, ends):
            available = np.count_nonzero(self._windows[self._person_rows[person]] != 7) - 2
            result.append(available - max(end - start - 1, 0))
        return np.asarray(result)


class _OptimizedTimetable(_GenericTimetable):
    def __init__(self, windows, person_rows, period_cols):
        super().__init__(windows, person_rows, period_cols)
        self.windows = windows
        self.windows_df = pd.DataFrame(windows, index=list(person_rows))
        self.window_row_ix = person_rows
        self.time_ix = period_cols


def test_vectorized_timetable_primitives_match_generic_activitysim_semantics():
    # ActivitySim uses non-middle padding states and subtracts the two padding
    # periods when reporting remaining availability.
    windows = np.array([[0, 0, 0, 0, 0], [0, 0, 4, 2, 0]], dtype=np.int8)
    person_rows = {100: 0, 200: 1}
    period_cols = {1: 1, 2: 2, 3: 3}
    choosers = pd.DataFrame(
        {"person_id": [100, 200], "tour_count": [2, 2], "tour_num": [1, 2]},
        index=[10, 20],
    )
    alternatives = pd.DataFrame(
        {"tdd": [0, 1, 0, 1], "start": [1, 2, 1, 2], "end": [2, 3, 2, 3], "duration": [2, 2, 2, 2]},
        index=[10, 10, 20, 20],
    )
    expressions = [
        "_adjacent_window_before@tt.adjacent_window_before(df.person_id, df.start)>0",
        "_adjacent_window_after@tt.adjacent_window_after(df.person_id, df.end)>0",
        "@tt.previous_tour_ends(df.person_id, df.start)",
        "@tt.previous_tour_begins(df.person_id, df.end)",
        "@(df.tour_count>1) * (df.tour_num == 1) * _adjacent_window_before",
        "@(df.tour_count>1) * (df.tour_num == 1) * _adjacent_window_after",
        "@(df.tour_num > 1) * _adjacent_window_before",
        "@(df.tour_num > 1) * _adjacent_window_after",
        "@((df.tour_count>1) & (df.tour_num == 1)) * 1.0 / tt.remaining_periods_available(df.person_id, df.start, df.end)",
    ]
    spec = pd.DataFrame({"coef": np.ones(len(expressions))}, index=expressions)
    lowered = lower_activitysim_spec(spec, choosers, alternatives)
    generic = evaluate_stateful_primitives(
        lowered, choosers, alternatives, {"tt": _GenericTimetable(windows, person_rows, period_cols)}
    )
    optimized = evaluate_stateful_primitives(
        lowered, choosers, alternatives, {"tt": _OptimizedTimetable(windows, person_rows, period_cols)}
    )
    for expected, actual in zip(generic, optimized):
        np.testing.assert_allclose(actual, expected)


def test_lowerer_handles_assignment_categoricals_and_inequality():
    choosers = pd.DataFrame(
        {"tour_type": ["escort", "shopping"], "x": [2, 3]}, index=[10, 20]
    )
    alternatives = pd.DataFrame(
        {"tdd": [0, 1, 0, 1], "start": [1, 2, 1, 2]},
        index=[10, 10, 20, 20],
    )
    spec = pd.DataFrame(
        {"coef": [1.0, 2.0, 3.0]},
        index=[
            "_is_escort@df.tour_type == 'escort'",
            "@(_is_escort) * df.start",
            "(tour_type != 'escort') * x",
        ],
    )

    lowered = lower_activitysim_spec(spec, choosers, alternatives)
    packed = pack_compact_scheduling_inputs(lowered, choosers, alternatives)

    assert lowered.stateful_expressions == ()
    assert lowered.expressions == (
        "(_is_escort) * start",
        "((not tour_type_escort)) * x",
    )
    assert {x[0] for x in lowered.categorical_columns} == {
        "_is_escort",
        "tour_type_escort",
    }
    columns = dict(zip(lowered.chooser_columns, packed.chooser_values.T))
    np.testing.assert_array_equal(columns["_is_escort"], [1, 0])
    np.testing.assert_array_equal(columns["tour_type_escort"], [1, 0])


def test_optimized_timetable_infers_joint_tour_row_owner():
    windows = np.array([[0, 0, 4, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    tour_rows = {1000: 0, 2000: 1}
    period_cols = {1: 1, 2: 2, 3: 3}
    choosers = pd.DataFrame(
        {"tour_id": [1000, 2000], "tour_category": ["joint", "joint"]},
        index=[10, 20],
    )
    alternatives = pd.DataFrame(
        {"tdd": [0, 1, 0, 1], "start": [1, 2, 1, 2], "end": [2, 3, 2, 3]},
        index=[10, 10, 20, 20],
    )
    spec = pd.DataFrame(
        {"coef": [1.0, 1.0]},
        index=[
            "@tt.previous_tour_ends(df.tour_id, df.start)",
            "@tt.previous_tour_begins(df.tour_id, df.end)",
        ],
    )
    lowered = lower_activitysim_spec(spec, choosers, alternatives)
    generic = evaluate_stateful_primitives(
        lowered, choosers, alternatives, {"tt": _GenericTimetable(windows, tour_rows, period_cols)}
    )
    optimized = evaluate_stateful_primitives(
        lowered, choosers, alternatives, {"tt": _OptimizedTimetable(windows, tour_rows, period_cols)}
    )
    for expected, actual in zip(generic, optimized):
        np.testing.assert_array_equal(actual, expected)


def test_unsupported_activitysim_path_uses_explicit_fallback():
    marker = object()

    def fallback(*args, **kwargs):
        return marker

    result = interaction_sample_simulate_choiceforge(
        object(),
        pd.DataFrame(index=[1]),
        pd.DataFrame({"tdd": [0]}, index=[1]),
        pd.DataFrame({"coef": [1.0]}, index=["1"]),
        "tdd",
        estimator=object(),
        fallback=fallback,
    )
    assert result is marker


def test_current_activitysim_alts_context_is_forwarded_to_fallback():
    marker = object()
    context = object()
    received = {}

    def fallback(*args, **kwargs):
        received.update(kwargs)
        return marker

    result = interaction_sample_simulate_choiceforge(
        object(),
        pd.DataFrame(index=[1]),
        pd.DataFrame({"tdd": [0]}, index=[1]),
        pd.DataFrame({"coef": [1.0]}, index=["1"]),
        "tdd",
        alts_context=context,
        fallback=fallback,
    )
    assert result is marker
    assert received["alts_context"] is context


def test_precision_guard_restores_rng_and_returns_activitysim_on_mismatch(monkeypatch):
    """The guard must not advance ActivitySim's controlled draw twice."""

    class Channel:
        def __init__(self):
            self.row_states = pd.DataFrame({"offset": [7]}, index=[10])

    class Rng:
        def __init__(self):
            self.channel = Channel()

        def get_channel_for_df(self, _df):
            return self.channel

        def random_for_df(self, df):
            self.channel.row_states.loc[df.index, "offset"] += 1
            return np.full(len(df), 0.5)

    class State:
        def __init__(self):
            self.rng = Rng()

        def get_rn_generator(self):
            return self.rng

    class MismatchingModel:
        def choose(self, *_args):
            return ChoiceResult(np.array([1]), np.array([0.0], dtype=np.float32))

    state = State()
    choosers = pd.DataFrame(index=[10])
    alternatives = pd.DataFrame({"tdd": [0, 1], "start": [1, 2]}, index=[10, 10])
    spec = pd.DataFrame({"coef": [1.0]}, index=["start"])
    fallback_offsets = []

    def fallback(fallback_state, fallback_choosers, *_args, **_kwargs):
        fallback_offsets.append(int(fallback_state.rng.channel.row_states.loc[10, "offset"]))
        fallback_state.get_rn_generator().random_for_df(fallback_choosers)
        return pd.Series([0], index=fallback_choosers.index)

    monkeypatch.setattr(scheduling, "_model_for", lambda _lowered: MismatchingModel())
    actual = interaction_sample_simulate_choiceforge(
        state,
        choosers,
        alternatives,
        spec,
        "tdd",
        fallback=fallback,
        precision_guard="shadow_fallback",
    )

    pd.testing.assert_series_equal(actual, pd.Series([0], index=choosers.index))
    assert fallback_offsets == [7]
    assert int(state.rng.channel.row_states.loc[10, "offset"]) == 8


def test_precision_guard_fails_closed_when_rng_offsets_are_unavailable(monkeypatch):
    class Rng:
        def random_for_df(self, _df):
            raise AssertionError("GPU path must not consume an unverifiable draw")

    class State:
        def get_rn_generator(self):
            return Rng()

    marker = pd.Series([0], index=[10])
    monkeypatch.setattr(scheduling, "_rng_offset_snapshot", lambda *_args: None)
    actual = interaction_sample_simulate_choiceforge(
        State(),
        pd.DataFrame(index=[10]),
        pd.DataFrame({"tdd": [0], "start": [1]}, index=[10]),
        pd.DataFrame({"coef": [1.0]}, index=["start"]),
        "tdd",
        fallback=lambda *_args, **_kwargs: marker,
        precision_guard="shadow_fallback",
    )
    pd.testing.assert_series_equal(actual, marker)


def test_precision_guard_rewinds_rng_after_gpu_failure(monkeypatch):
    class Channel:
        def __init__(self):
            self.row_states = pd.DataFrame({"offset": [4]}, index=[10])

    class Rng:
        def __init__(self):
            self.channel = Channel()

        def get_channel_for_df(self, _df):
            return self.channel

        def random_for_df(self, df):
            self.channel.row_states.loc[df.index, "offset"] += 1
            return np.full(len(df), 0.5)

    class State:
        def __init__(self):
            self.rng = Rng()

        def get_rn_generator(self):
            return self.rng

    class FailingModel:
        def choose(self, *_args):
            raise RuntimeError("simulated CUDA failure")

    state = State()
    fallback_offsets = []

    def fallback(fallback_state, fallback_choosers, *_args, **_kwargs):
        fallback_offsets.append(int(fallback_state.rng.channel.row_states.loc[10, "offset"]))
        fallback_state.rng.random_for_df(fallback_choosers)
        return pd.Series([0], index=fallback_choosers.index)

    monkeypatch.setattr(scheduling, "_model_for", lambda _lowered: FailingModel())
    actual = interaction_sample_simulate_choiceforge(
        state,
        pd.DataFrame(index=[10]),
        pd.DataFrame({"tdd": [0], "start": [1]}, index=[10]),
        pd.DataFrame({"coef": [1.0]}, index=["start"]),
        "tdd",
        fallback=fallback,
        precision_guard="shadow_fallback",
    )
    pd.testing.assert_series_equal(actual, pd.Series([0], index=[10]))
    assert fallback_offsets == [4]
    assert int(state.rng.channel.row_states.loc[10, "offset"]) == 5
