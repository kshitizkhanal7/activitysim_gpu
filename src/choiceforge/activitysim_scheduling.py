"""Explicit ActivitySim tour-scheduling backend for the compact GPU kernel.

The public function in this module mirrors ``interaction_sample_simulate`` at
the narrow call boundary used by vectorized tour scheduling.  ActivitySim keeps
ownership of feasible-alternative construction, mode-choice logsums, random
draws, tracing fallbacks, and timetable mutation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import logging
import re
import time
from typing import Any, Callable
from types import SimpleNamespace

import numpy as np

from .scheduling_compiler import CompiledCudaSchedulingModel, SchedulingSchema

logger = logging.getLogger(__name__)


class UnsupportedActivitySimPath(RuntimeError):
    """Raised when a call needs behavior deliberately left to ActivitySim."""


@dataclass(frozen=True)
class LoweredSchedulingSpec:
    expressions: tuple[str, ...]
    coefficients: tuple[float, ...]
    chooser_columns: tuple[str, ...]
    row_columns: tuple[str, ...]
    alternative_columns: tuple[str, ...]
    stateful_expressions: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...]
    categorical_columns: tuple[tuple[str, str, str, bool], ...]

    @property
    def schema(self) -> SchedulingSchema:
        return SchedulingSchema(
            self.chooser_columns, self.row_columns, self.alternative_columns
        )


@dataclass(frozen=True)
class CompactSchedulingInputs:
    chooser_values: np.ndarray
    row_values: np.ndarray
    alternative_values: np.ndarray
    alternative_ids: np.ndarray
    offsets: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            x.nbytes
            for x in (
                self.chooser_values,
                self.row_values,
                self.alternative_values,
                self.alternative_ids,
                self.offsets,
            )
        )


@dataclass(frozen=True)
class SchedulingTelemetry:
    lower_ms: float
    stateful_ms: float
    pack_ms: float
    random_ms: float
    gpu_ms: float
    map_ms: float
    compact_bytes: int

    @property
    def total_ms(self) -> float:
        return (
            self.lower_ms
            + self.stateful_ms
            + self.pack_ms
            + self.random_ms
            + self.gpu_ms
            + self.map_ms
        )


_STRING_EQUALITY = re.compile(
    r"\b([A-Za-z_]\w*)\s*(==|!=)\s*(['\"])([^'\"]+)\3"
)
_CATEGORICAL_ASSIGNMENT = re.compile(
    r"df\.([A-Za-z_]\w*)\s*(==|!=)\s*(['\"])([^'\"]+)\3"
)


def _spec_expressions(spec):
    try:
        from activitysim.core import simulate

        expression_level = simulate.SPEC_EXPRESSION_NAME
    except ImportError:  # pragma: no cover - ActivitySim is optional
        expression_level = "Expression"
    if getattr(spec.index, "nlevels", 1) > 1:
        return [str(x) for x in spec.index.get_level_values(expression_level)]
    return [str(x) for x in spec.index]


def _normalize_categoricals(
    expression: str, transforms: list[tuple[str, str, str, bool]]
) -> str:
    def replace(match):
        source, operator, _quote, value = match.groups()
        safe_value = re.sub(r"\W+", "_", value).strip("_")
        generated = f"{source}_{safe_value}"
        item = (generated, source, value, False)
        if item not in transforms:
            transforms.append(item)
        return generated if operator == "==" else f"(not {generated})"

    return _STRING_EQUALITY.sub(replace, expression)


def _names(expression: str) -> set[str]:
    return {
        node.id
        for node in ast.walk(ast.parse(expression, mode="eval"))
        if isinstance(node, ast.Name)
    }


def lower_activitysim_spec(spec, choosers, alternatives) -> LoweredSchedulingSpec:
    """Lower the supported mandatory-scheduling spec into compact scopes."""
    expressions = _spec_expressions(spec)
    coefficients = np.asarray(spec.iloc[:, 0], dtype=np.float32)
    assignments: list[tuple[str, str]] = []
    stateful: list[str] = []
    lowered: list[str] = []
    kept_coefficients: list[float] = []
    categoricals: list[tuple[str, str, str, bool]] = []

    # Classify assignment rows first because later @ expressions may refer to
    # them. Simple categorical flags belong at chooser scope; timetable and
    # other general assignments remain in the stateful evaluator.
    simple_assignments: set[str] = set()
    stateful_assignments: set[str] = set()
    for expression in expressions:
        if not (expression.startswith("_") and "@" in expression):
            continue
        name, rhs = expression.split("@", 1)
        match = _CATEGORICAL_ASSIGNMENT.fullmatch(rhs.strip())
        if match:
            source, operator, _quote, value = match.groups()
            categoricals.append((name, source, value, operator == "!="))
            simple_assignments.add(name)
        else:
            stateful_assignments.add(name)

    for expression, coefficient in zip(expressions, coefficients):
        if expression.startswith("_") and "@" in expression:
            name, rhs = expression.split("@", 1)
            if name in stateful_assignments:
                assignments.append((name, rhs))
            continue
        if expression.startswith("@"):
            candidate = expression[1:].replace("df.", "")
            candidate_names = _names(candidate)
            if (
                "tt." in expression
                or "np." in expression
                or bool(candidate_names & stateful_assignments)
            ):
                name = f"stateful_{len(stateful)}"
                stateful.append(expression[1:])
                lowered.append(name)
            else:
                lowered.append(_normalize_categoricals(candidate, categoricals))
        else:
            lowered.append(_normalize_categoricals(expression, categoricals))
        kept_coefficients.append(float(coefficient))

    all_names: set[str] = set()
    for expression in lowered:
        all_names.update(_names(expression))

    alternative_columns = tuple(
        name for name in ("start", "end", "duration") if name in all_names
    )
    stateful_names = tuple(f"stateful_{i}" for i in range(len(stateful)))
    row_columns = tuple(
        name for name in alternatives.columns if name in all_names and name not in alternative_columns
    ) + stateful_names
    categorical_names = {name for name, _source, _value, _negate in categoricals}
    chooser_columns = tuple(
        sorted(
            name
            for name in all_names
            if name not in alternative_columns
            and name not in row_columns
            and (name in choosers.columns or name in categorical_names)
        )
    )
    unresolved = all_names - set(alternative_columns) - set(row_columns) - set(chooser_columns)
    if unresolved:
        raise UnsupportedActivitySimPath(f"unresolved scheduling names: {sorted(unresolved)}")

    return LoweredSchedulingSpec(
        tuple(lowered),
        tuple(kept_coefficients),
        chooser_columns,
        row_columns,
        alternative_columns,
        tuple(stateful),
        tuple(assignments),
        tuple(categoricals),
    )


def _row_offsets(index_values: np.ndarray, n_choosers: int) -> np.ndarray:
    if index_values.size == 0:
        raise ValueError("scheduling alternatives cannot be empty")
    starts = np.flatnonzero(np.r_[True, index_values[1:] != index_values[:-1]])
    if starts.size != n_choosers:
        raise ValueError("alternatives must contain one contiguous tranche per chooser")
    return np.r_[starts, index_values.size].astype(np.int64, copy=False)


def _optimized_timetable_primitives(lowered, choosers, alternatives, locals_d):
    """Evaluate the MTC timetable vocabulary once per 21-period day.

    ActivitySim's generic functions revisit the same person's timetable for
    every feasible TDD row. Mandatory scheduling has at most 21 time periods,
    so computing per-chooser lookup tables and gathering is much cheaper.
    """
    timetable = (locals_d or {}).get("tt")
    vocabulary = (
        "previous_tour_ends",
        "previous_tour_begins",
        "adjacent_window_before",
        "adjacent_window_after",
        "remaining_periods_available",
    )
    source = " ".join(rhs for _name, rhs in lowered.assignments) + " " + " ".join(
        lowered.stateful_expressions
    )
    timetable_calls = set(re.findall(r"tt\.([A-Za-z_]\w*)", source))
    if not timetable_calls:
        return None
    if timetable is None or not all(
        hasattr(timetable, name)
        for name in ("windows", "window_row_ix", "time_ix")
    ) or not timetable_calls.issubset(vocabulary):
        return None
    required = set(re.findall(r"df\.([A-Za-z_]\w*)", source))
    id_matches = re.findall(
        r"tt\.(?:previous_tour_ends|previous_tour_begins|adjacent_window_before|adjacent_window_after|remaining_periods_available)\(df\.([A-Za-z_]\w*)",
        source,
    )
    if not id_matches or len(set(id_matches)) != 1 or not {"start", "end"}.issubset(required):
        return None
    window_id_column = id_matches[0]

    index_values = np.asarray(alternatives.index)
    offsets = _row_offsets(index_values, len(choosers))
    counts = np.diff(offsets)
    row_chooser = np.repeat(np.arange(len(choosers), dtype=np.int32), counts)
    first_rows = offsets[:-1]
    window_ids = np.asarray(choosers[window_id_column])

    if hasattr(timetable, "windows_df"):
        window_rows = timetable.windows_df.index.get_indexer(window_ids)
        if np.any(window_rows < 0):
            raise UnsupportedActivitySimPath("timetable is missing chooser window rows")
    else:  # pragma: no cover - compatibility with older ActivitySim
        window_rows = np.fromiter(
            (timetable.window_row_ix[x] for x in window_ids),
            dtype=np.int64,
            count=len(person_ids),
        )
    windows = np.asarray(timetable.windows)[window_rows]
    n_periods = windows.shape[1]

    starts = np.asarray(alternatives["start"])
    ends = np.asarray(alternatives["end"])
    period_values = np.unique(np.r_[starts, ends])
    period_map = {value: int(timetable.time_ix[value]) for value in period_values}
    start_cols = np.fromiter((period_map[x] for x in starts), dtype=np.int32, count=len(starts))
    end_cols = np.fromiter((period_map[x] for x in ends), dtype=np.int32, count=len(ends))

    window_for_row = windows[row_chooser]
    previous_ends = np.isin(window_for_row[np.arange(len(starts)), start_cols], (4, 6))
    previous_begins = np.isin(window_for_row[np.arange(len(ends)), end_cols], (2, 6))

    available = windows != 7
    available[:, 0] = False
    available[:, -1] = False
    before = np.zeros_like(windows, dtype=np.int16)
    last_unavailable = np.zeros(len(windows), dtype=np.int16)
    for column in range(n_periods):
        before[:, column] = column - last_unavailable - 1
        last_unavailable = np.where(available[:, column], last_unavailable, column)
    after = np.zeros_like(windows, dtype=np.int16)
    next_unavailable = np.full(len(windows), n_periods - 1, dtype=np.int16)
    for column in range(n_periods - 1, -1, -1):
        after[:, column] = next_unavailable - column - 1
        next_unavailable = np.where(available[:, column], next_unavailable, column)
    adjacent_before = before[row_chooser, start_cols]
    adjacent_after = after[row_chooser, end_cols]
    # Padding was forced unavailable above, so its two periods are already
    # excluded (ActivitySim subtracts them after counting non-middle states).
    available_counts = available.sum(axis=1)
    remaining = available_counts[row_chooser] - np.maximum(ends - starts - 1, 0)

    values = {}
    for name in required:
        if name in alternatives.columns:
            values[name] = np.asarray(alternatives[name])
        elif name in choosers.columns:
            values[name] = np.asarray(choosers[name])[row_chooser]
        else:
            raise UnsupportedActivitySimPath(f"stateful input unavailable: {name}")
    df = SimpleNamespace(**values)

    class TimetableProxy:
        def previous_tour_ends(self, *_args):
            return previous_ends

        def previous_tour_begins(self, *_args):
            return previous_begins

        def adjacent_window_before(self, *_args):
            return adjacent_before

        def adjacent_window_after(self, *_args):
            return adjacent_after

        def remaining_periods_available(self, *_args):
            return remaining

    local = dict(locals_d or {})
    local.update({"df": df, "tt": TimetableProxy()})
    namespace = {"np": np}
    for name, rhs in lowered.assignments:
        local[name] = eval(rhs, namespace, local)
    output = []
    for expression in lowered.stateful_expressions:
        value = eval(expression, namespace, local)
        if np.isscalar(value):
            value = np.full(len(alternatives), value)
        output.append(np.asarray(value, dtype=np.float32))
    return output


def evaluate_stateful_primitives(lowered, choosers, alternatives, locals_d) -> list[np.ndarray]:
    """Evaluate only the timetable-dependent expressions on a narrow frame."""
    if not lowered.stateful_expressions:
        return []
    optimized = _optimized_timetable_primitives(lowered, choosers, alternatives, locals_d)
    if optimized is not None:
        return optimized
    local = dict(locals_d or {})
    required = set()
    for _name, rhs in lowered.assignments:
        required.update(re.findall(r"df\.([A-Za-z_]\w*)", rhs))
    for expression in lowered.stateful_expressions:
        required.update(re.findall(r"df\.([A-Za-z_]\w*)", expression))

    frame = alternatives[[c for c in required if c in alternatives.columns]].copy()
    chooser_required = [c for c in required if c not in frame.columns and c in choosers.columns]
    if chooser_required:
        frame = frame.join(choosers[chooser_required], how="left")
    missing = required - set(frame.columns)
    if missing:
        raise UnsupportedActivitySimPath(f"stateful inputs unavailable: {sorted(missing)}")
    local["df"] = frame
    namespace = {"np": np}
    for name, rhs in lowered.assignments:
        local[name] = eval(rhs, namespace, local)
    output = []
    for expression in lowered.stateful_expressions:
        value = eval(expression, namespace, local)
        if np.isscalar(value):
            value = np.full(len(frame), value)
        output.append(np.asarray(value, dtype=np.float32))
    return output


def pack_compact_scheduling_inputs(
    lowered: LoweredSchedulingSpec,
    choosers,
    alternatives,
    *,
    choice_column: str = "tdd",
    stateful_values: list[np.ndarray] | None = None,
) -> CompactSchedulingInputs:
    """Pack pandas inputs without building ActivitySim's wide interaction matrix."""
    index_values = np.asarray(alternatives.index)
    offsets = _row_offsets(index_values, len(choosers))

    chooser_arrays = []
    categorical = {
        name: (source, value, negate)
        for name, source, value, negate in lowered.categorical_columns
    }
    for name in lowered.chooser_columns:
        if name in categorical:
            source, value, negate = categorical[name]
            array = np.asarray(choosers[source] == value)
            if negate:
                array = ~array
        else:
            array = np.asarray(choosers[name])
        chooser_arrays.append(array.astype(np.float32, copy=False))
    chooser_values = (
        np.ascontiguousarray(np.column_stack(chooser_arrays), dtype=np.float32)
        if chooser_arrays
        else np.empty((len(choosers), 0), dtype=np.float32)
    )

    alternative_ids = np.ascontiguousarray(np.asarray(alternatives[choice_column]), dtype=np.int16)
    n_alternatives = int(alternative_ids.max()) + 1
    alternative_values = np.empty(
        (n_alternatives, len(lowered.alternative_columns)), dtype=np.float32
    )
    for column, name in enumerate(lowered.alternative_columns):
        alternative_values[alternative_ids, column] = np.asarray(
            alternatives[name], dtype=np.float32
        )

    row_arrays = []
    stateful_values = stateful_values or []
    stateful_by_name = {
        f"stateful_{i}": values for i, values in enumerate(stateful_values)
    }
    for name in lowered.row_columns:
        if name in stateful_by_name:
            array = stateful_by_name[name]
        else:
            array = np.asarray(alternatives[name])
        row_arrays.append(np.asarray(array, dtype=np.float32))
    row_values = (
        np.ascontiguousarray(np.column_stack(row_arrays), dtype=np.float32)
        if row_arrays
        else np.empty((len(alternatives), 0), dtype=np.float32)
    )
    return CompactSchedulingInputs(
        chooser_values, row_values, alternative_values, alternative_ids, offsets
    )


@lru_cache(maxsize=8)
def _model_for(lowered: LoweredSchedulingSpec) -> CompiledCudaSchedulingModel:
    return CompiledCudaSchedulingModel(
        lowered.expressions, lowered.coefficients, lowered.schema
    )


def _fallback_or_raise(fallback: Callable | None, reason: str, args, kwargs):
    if fallback is None:
        raise UnsupportedActivitySimPath(reason)
    return fallback(*args, **kwargs)


def _rng_offset_snapshot(state, choosers):
    """Capture the offsets advanced by ``random_for_df`` when possible.

    The precision guard runs ChoiceForge first, then may need ActivitySim to
    make the authoritative choice.  ActivitySim's random streams are stateful,
    so the fallback must see the *same* draw rather than the next one.  The
    public random manager stores those offsets in the chooser channel.
    """
    try:
        rng = state.get_rn_generator()
        channel = rng.get_channel_for_df(choosers)
        row_states = channel.row_states
        return channel, row_states.loc[choosers.index, "offset"].copy()
    except (AttributeError, KeyError):
        return None


def _restore_rng_offsets(snapshot) -> None:
    if snapshot is None:
        return
    channel, offsets = snapshot
    channel.row_states.loc[offsets.index, "offset"] = offsets


def _choice_values(choices):
    """Return selected labels from either ActivitySim choice result shape."""
    if hasattr(choices, "columns") and "choice" in choices.columns:
        return np.asarray(choices["choice"])
    return np.asarray(choices)


def interaction_sample_simulate_choiceforge(
    state,
    choosers,
    alternatives,
    spec,
    choice_column,
    allow_zero_probs=False,
    zero_prob_choice_val=None,
    log_alt_losers=False,
    want_logsums=False,
    skims=None,
    locals_d=None,
    chunk_size=0,
    chunk_tag=None,
    trace_label=None,
    trace_choice_name=None,
    estimator=None,
    skip_choice=False,
    explicit_chunk_size=0,
    *,
    compute_settings=None,
    alts_context=None,
    fallback: Callable | None = None,
    return_telemetry: bool = False,
    precision_guard: str = "off",
):
    """Run the supported ActivitySim scheduling choice through ChoiceForge.

    Unsupported tracing, estimation, chunking, or skim paths delegate to the
    supplied ActivitySim fallback.  Normal mandatory-tour scheduling uses none
    of those paths at this boundary.
    """
    call_args = (state, choosers, alternatives, spec, choice_column)
    call_kwargs = dict(
        allow_zero_probs=allow_zero_probs,
        zero_prob_choice_val=zero_prob_choice_val,
        log_alt_losers=log_alt_losers,
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name,
        estimator=estimator,
        skip_choice=skip_choice,
        explicit_chunk_size=explicit_chunk_size,
        compute_settings=compute_settings,
        alts_context=alts_context,
    )
    has_trace_targets = bool(
        trace_label and hasattr(state, "tracing") and state.tracing.has_trace_targets(choosers)
    )
    unsupported = (
        allow_zero_probs
        or log_alt_losers
        or skims is not None
        or chunk_size
        or explicit_chunk_size
        or estimator is not None
        or skip_choice
        or alts_context is not None
        or has_trace_targets
    )
    if unsupported:
        return _fallback_or_raise(
            fallback, "call requires ActivitySim tracing/estimation/chunk behavior", call_args, call_kwargs
        )

    if precision_guard not in ("off", "shadow_fallback"):
        raise ValueError(
            "precision_guard must be 'off' or 'shadow_fallback', "
            f"not {precision_guard!r}"
        )
    if precision_guard == "shadow_fallback" and fallback is None:
        raise UnsupportedActivitySimPath(
            "precision guard requires the ActivitySim fallback"
        )

    start = time.perf_counter()
    lowered = lower_activitysim_spec(spec, choosers, alternatives)
    after_lower = time.perf_counter()
    stateful = evaluate_stateful_primitives(lowered, choosers, alternatives, locals_d)
    after_stateful = time.perf_counter()
    compact = pack_compact_scheduling_inputs(
        lowered,
        choosers,
        alternatives,
        choice_column=choice_column,
        stateful_values=stateful,
    )
    after_pack = time.perf_counter()

    rng_snapshot = _rng_offset_snapshot(state, choosers)
    if precision_guard == "shadow_fallback" and rng_snapshot is None:
        # A guard without an exact rewind point would make the CPU fallback
        # consume a different controlled draw.  Preserve replication by
        # declining the GPU path instead of guessing about a future RNG API.
        logger.warning(
            "%s ChoiceForge precision guard could not snapshot RNG offsets; returning ActivitySim",
            trace_label,
        )
        return fallback(*call_args, **call_kwargs)
    try:
        draws = np.asarray(
            state.get_rn_generator().random_for_df(choosers), dtype=np.float32
        ).reshape(-1)
        after_random = time.perf_counter()
        result = _model_for(lowered).choose(
            compact.chooser_values,
            compact.row_values,
            compact.alternative_values,
            compact.alternative_ids,
            compact.offsets,
            draws,
        )
    except Exception:
        if precision_guard == "shadow_fallback":
            _restore_rng_offsets(rng_snapshot)
            logger.exception(
                "%s ChoiceForge GPU failure under precision guard; returning ActivitySim",
                trace_label,
            )
            return fallback(*call_args, **call_kwargs)
        raise
    after_gpu = time.perf_counter()

    positions = np.asarray(result.choices, dtype=np.int64)
    selected_rows = compact.offsets[:-1] + positions
    labels = np.asarray(alternatives[choice_column])[selected_rows]
    import pandas as pd

    choices = pd.Series(labels, index=choosers.index)
    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = np.asarray(result.logsums)
    after_map = time.perf_counter()

    if precision_guard == "shadow_fallback":
        # The GPU has consumed the random draw.  Rewind precisely that advance
        # so ActivitySim's authoritative calculation sees the same draw and
        # leaves the stream in the normal post-choice state.
        _restore_rng_offsets(rng_snapshot)
        reference_choices = fallback(*call_args, **call_kwargs)
        mismatch_count = int(
            np.count_nonzero(_choice_values(choices) != _choice_values(reference_choices))
        )
        logger.info(
            "%s ChoiceForge precision guard compared %d choices; mismatches=%d",
            trace_label,
            len(choosers),
            mismatch_count,
        )
        if mismatch_count:
            logger.warning(
                "%s ChoiceForge precision guard returned ActivitySim for %d mismatched choices",
                trace_label,
                mismatch_count,
            )
            return reference_choices

    telemetry = SchedulingTelemetry(
        (after_lower - start) * 1000,
        (after_stateful - after_lower) * 1000,
        (after_pack - after_stateful) * 1000,
        (after_random - after_pack) * 1000,
        (after_gpu - after_random) * 1000,
        (after_map - after_gpu) * 1000,
        compact.nbytes + draws.nbytes,
    )
    logger.info(
        "%s ChoiceForge rows=%d choosers=%d compact=%.3fMB "
        "lower=%.3fms stateful=%.3fms pack=%.3fms rng=%.3fms gpu=%.3fms map=%.3fms total=%.3fms",
        trace_label,
        len(alternatives),
        len(choosers),
        telemetry.compact_bytes / 1_000_000,
        telemetry.lower_ms,
        telemetry.stateful_ms,
        telemetry.pack_ms,
        telemetry.random_ms,
        telemetry.gpu_ms,
        telemetry.map_ms,
        telemetry.total_ms,
    )
    if not return_telemetry:
        return choices
    return choices, telemetry
