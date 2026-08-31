"""Compact exact final-choice runtime for trip destination.

The runtime deliberately reuses Sharrow's authoritative compiled utility
evaluator.  It removes only ActivitySim's generic wide join, group-by count,
``np.insert`` padding, and generic position mapping around that evaluator.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


SUPPORTED_EXPRESSIONS = (
    "_od_DIST@od_skims['DIST']",
    "_dp_DIST@dp_skims['DIST']",
    "@np.log1p(size_terms.get(df.dest_taz, df.purpose)) # sharrow: np.log1p(size_terms['sizearray'])",
    "@size_terms.get(df.dest_taz, df.purpose) == 0 # sharrow: size_terms['sizearray'] == 0",
    "@(~df.is_joint & ~df.outbound) * (_od_DIST + _dp_DIST)",
    "@(~df.is_joint & df.outbound) * (_od_DIST + _dp_DIST)",
    "@df.is_joint * (_od_DIST + _dp_DIST)",
    "@df.outbound * _od_DIST",
    "@~df.outbound * _dp_DIST",
    "@df.outbound * _dp_DIST",
    "@~df.outbound * _od_DIST",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
    "od_logsum",
    "(od_logsum < -100)",
    "dp_logsum",
    "(dp_logsum < -100)",
)

REQUIRED_ALTERNATIVE_COLUMNS = (
    "dest_taz",
    "prob",
    "pick_count",
    "od_logsum",
    "dp_logsum",
)
REQUIRED_CHOOSER_COLUMNS = (
    "origin",
    "tour_leg_dest",
    "trip_period",
    "is_joint",
    "outbound",
    "purpose",
    "purpose_index_num",
)


if njit is not None:

    @njit(cache=True)
    def _pad_ragged_f32(values, offsets, width):
        result = np.full((offsets.size - 1, width), -999.0, dtype=np.float32)
        for chooser in range(offsets.size - 1):
            begin = offsets[chooser]
            end = offsets[chooser + 1]
            for row in range(begin, end):
                result[chooser, row - begin] = values[row]
        return result


def _spec_expressions(spec) -> tuple[str, ...]:
    from activitysim.core import simulate

    if isinstance(spec.index, pd.MultiIndex):
        values = spec.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
    else:
        values = spec.index
    return tuple(str(value) for value in values)


def validate_final_spec(spec) -> None:
    """Fail closed unless the reviewed public final-choice program is active."""
    if len(spec.columns) != 1:
        raise ValueError("Phase 44 requires one final-choice coefficient column")
    expressions = _spec_expressions(spec)
    if expressions != SUPPORTED_EXPRESSIONS:
        raise ValueError(
            "Phase 44 final-choice expression ABI differs from the reviewed program"
        )
    coefficients = np.asarray(spec.iloc[:, 0], dtype=np.float32)
    if coefficients.shape != (16,) or not np.isfinite(coefficients).all():
        raise ValueError("Phase 44 final-choice coefficients are invalid")


def ragged_offsets(alternatives, choosers) -> tuple[np.ndarray, np.ndarray]:
    """Return chooser offsets and counts after strict contiguity validation."""
    row_ids = alternatives.index.to_numpy(copy=False)
    if row_ids.ndim != 1 or row_ids.size == 0:
        raise ValueError("Phase 44 requires a nonempty sampled alternative table")
    starts = np.flatnonzero(np.r_[True, row_ids[1:] != row_ids[:-1]])
    chooser_ids = choosers.index.to_numpy(copy=False)
    if not np.array_equal(row_ids[starts], chooser_ids):
        raise ValueError("Phase 44 alternatives are not contiguous in chooser order")
    offsets = np.ascontiguousarray(np.r_[starts, row_ids.size], dtype=np.int64)
    counts = np.diff(offsets)
    if np.any(counts <= 0) or int(counts.max()) > 1024:
        raise ValueError("Phase 44 ragged alternative counts are unsupported")
    return offsets, counts


def _repeat_categorical(values: pd.Series, counts: np.ndarray):
    if isinstance(values.dtype, pd.CategoricalDtype):
        codes = np.repeat(values.cat.codes.to_numpy(copy=False), counts)
        return pd.Categorical.from_codes(
            codes,
            categories=values.cat.categories,
            ordered=values.cat.ordered,
        )
    return np.repeat(values.to_numpy(copy=False), counts)


def narrow_interaction_frame(alternatives, choosers, counts):
    """Construct only the columns referenced by the reviewed compiled flow."""
    missing_alts = set(REQUIRED_ALTERNATIVE_COLUMNS) - set(alternatives.columns)
    missing_choosers = set(REQUIRED_CHOOSER_COLUMNS) - set(choosers.columns)
    if missing_alts or missing_choosers:
        raise ValueError(
            f"Phase 44 missing columns alternatives={sorted(missing_alts)} "
            f"choosers={sorted(missing_choosers)}"
        )
    data = {
        name: alternatives[name].to_numpy(copy=False)
        for name in REQUIRED_ALTERNATIVE_COLUMNS
    }
    for name in REQUIRED_CHOOSER_COLUMNS:
        data[name] = _repeat_categorical(choosers[name], counts)
    return pd.DataFrame(data, index=alternatives.index, copy=False)


def compact_interaction_sample_simulate(
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
    alts_context=None,
    compute_settings=None,
    telemetry=None,
):
    """Execute exact Sharrow utilities through a compact ragged boundary."""
    from activitysim.core import interaction_simulate, logit, simulate

    started = time.perf_counter()
    if njit is None:
        raise RuntimeError("Phase 44 compact padding requires Numba")
    if estimator is not None or skip_choice or state.settings.use_explicit_error_terms:
        raise ValueError("Phase 44 does not support estimation, skip-choice, or EET")
    if chunk_size or explicit_chunk_size:
        raise ValueError("Phase 44 requires the qualified unchunked final boundary")
    validate_final_spec(spec)
    offsets, counts = ragged_offsets(alternatives, choosers)

    frame_started = time.perf_counter()
    interaction_df = narrow_interaction_frame(alternatives, choosers, counts)
    # The reviewed final program references only these two wrappers.  Keeping
    # unrelated wrappers in the namespace makes Sharrow require columns for
    # unused skim relationships, defeating the compact boundary.
    used_skims = {name: skims[name] for name in ("od_skims", "dp_skims")}
    compact_locals = {
        name: value for name, value in (locals_d or {}).items() if name not in skims
    }
    compact_locals.update(used_skims)
    # ActivitySim's trip-destination relationship template always declares
    # primary-origin relationships.  Sharrow needs this wrapper's keys to
    # name that relationship even though the reviewed expressions never read
    # from it; no dnt skim values are evaluated by the compiled program.
    compact_locals["dnt_skims"] = skims["dnt_skims"]
    simulate.set_skim_wrapper_targets(interaction_df, used_skims)
    frame_seconds = time.perf_counter() - frame_started

    utility_started = time.perf_counter()
    utilities, _ = interaction_simulate.eval_interaction_utilities(
        state,
        spec,
        interaction_df,
        compact_locals,
        trace_label,
        None,
        estimator=None,
        log_alt_losers=log_alt_losers,
        compute_settings=compute_settings,
    )
    raw = np.asarray(utilities.utility, dtype=np.float32)
    if raw.shape != (len(alternatives),):
        raise ValueError("Phase 44 compiled utility result has an invalid shape")
    utility_seconds = time.perf_counter() - utility_started

    pad_started = time.perf_counter()
    padded = _pad_ragged_f32(raw, offsets, int(counts.max()))
    utilities_df = pd.DataFrame(padded, index=choosers.index)
    pad_seconds = time.perf_counter() - pad_started

    probability_started = time.perf_counter()
    if want_logsums:
        probs, logsums = logit.utils_to_probs(
            state,
            utilities_df,
            allow_zero_probs=allow_zero_probs,
            trace_label=trace_label,
            trace_choosers=choosers,
            overflow_protection=not allow_zero_probs,
            return_logsums=True,
        )
    else:
        probs = logit.utils_to_probs(
            state,
            utilities_df,
            allow_zero_probs=allow_zero_probs,
            trace_label=trace_label,
            trace_choosers=choosers,
            overflow_protection=not allow_zero_probs,
        )
        logsums = None
    zero_probs = probs.sum(axis=1) == 0 if allow_zero_probs else None
    if zero_probs is not None and zero_probs.any():
        probs.loc[zero_probs, 0] = 1.0
    probability_seconds = time.perf_counter() - probability_started

    choice_started = time.perf_counter()
    positions, _ = logit.make_choices(
        state, probs, trace_label=trace_label, trace_choosers=choosers
    )
    selected_positions = (
        positions.clip(lower=0) if state.settings.skip_failed_choices else positions
    )
    selected_rows = offsets[:-1] + np.asarray(selected_positions, dtype=np.int64)
    choices = pd.Series(
        alternatives[choice_column].to_numpy(copy=False)[selected_rows],
        index=choosers.index,
    )
    if zero_probs is not None and zero_probs.any() and zero_prob_choice_val is not None:
        choices.loc[zero_probs] = zero_prob_choice_val
    choice_seconds = time.perf_counter() - choice_started
    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = logsums

    if telemetry is not None:
        telemetry.append(
            {
                "chooser_rows": len(choosers),
                "alternative_rows": len(alternatives),
                "max_alternatives": int(counts.max()),
                "expression_slots": 16,
                "effective_utility_terms": 14,
                "frame_seconds": frame_seconds,
                "utility_seconds": utility_seconds,
                "padding_seconds": pad_seconds,
                "probability_seconds": probability_seconds,
                "choice_seconds": choice_seconds,
                "total_seconds": time.perf_counter() - started,
            }
        )
    return choices
