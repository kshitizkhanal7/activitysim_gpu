"""Reusable compact sampled-choice boundary for Phase 45.

This module preserves ActivitySim's authoritative Sharrow utility evaluator,
probability implementation, keyed random manager, and output contract.  It
replaces the generic pandas join/group-by/``np.insert`` mechanics around those
operations with contiguous ragged offsets and direct typed-array expansion.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .trip_destination_final import _pad_ragged_f32, ragged_offsets


def _repeat_series(values: pd.Series, counts: np.ndarray):
    if isinstance(values.dtype, pd.CategoricalDtype):
        codes = np.repeat(values.cat.codes.to_numpy(copy=False), counts)
        return pd.Categorical.from_codes(
            codes,
            categories=values.cat.categories,
            ordered=values.cat.ordered,
        )
    return np.repeat(values.to_numpy(copy=False), counts)


def compact_interaction_frame(
    alternatives: pd.DataFrame,
    choosers: pd.DataFrame,
    counts: np.ndarray,
) -> pd.DataFrame:
    """Reproduce ``alternatives.join(choosers, rsuffix='_chooser')`` directly."""
    data = {
        name: (
            alternatives[name].array
            if isinstance(alternatives[name].dtype, pd.CategoricalDtype)
            else alternatives[name].to_numpy(copy=False)
        )
        for name in alternatives.columns
    }
    for name in choosers.columns:
        output_name = f"{name}_chooser" if name in data else name
        data[output_name] = _repeat_series(choosers[name], counts)
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
    component="unknown",
):
    """Run one unchunked sampled MNL choice through the compact exact boundary."""
    from activitysim.core import interaction_simulate, logit, tracing, util
    from activitysim.core.configuration.base import ComputeSettings
    from activitysim.core.simulate import set_skim_wrapper_targets

    started = time.perf_counter()
    if (
        estimator is not None
        or skip_choice
        or log_alt_losers
        or alts_context is not None
        or state.settings.use_explicit_error_terms
    ):
        raise ValueError(
            "Phase 45 compact choice does not support estimation, skip-choice, "
            "loser logging, alternative contexts, or explicit error terms"
        )
    if chunk_size or explicit_chunk_size:
        raise ValueError("Phase 45 requires the qualified unchunked choice boundary")
    if len(spec.columns) != 1:
        raise ValueError("Phase 45 requires one final-choice coefficient column")
    if state.tracing.has_trace_targets(choosers):
        raise ValueError("Phase 45 defers traced households to ActivitySim")
    if not choosers.index.is_monotonic_increasing:
        raise ValueError("Phase 45 chooser index is not monotonic")
    if not alternatives.index.is_monotonic_increasing:
        raise ValueError("Phase 45 alternative index is not monotonic")

    trace_label = tracing.extend_trace_label(
        trace_label, "interaction_sample_simulate"
    )
    compute_settings = compute_settings or ComputeSettings()
    offsets, counts = ragged_offsets(alternatives, choosers)

    prune_started = time.perf_counter()
    if compute_settings.drop_unused_columns:
        choosers = util.drop_unused_columns(
            choosers,
            spec,
            locals_d,
            custom_chooser=None,
            sharrow_enabled=state.settings.sharrow,
            additional_columns=compute_settings.protect_columns,
        )
        alternatives = util.drop_unused_columns(
            alternatives,
            spec,
            locals_d,
            custom_chooser=None,
            sharrow_enabled=state.settings.sharrow,
            additional_columns=list(
                dict.fromkeys([choice_column, "tdd", *compute_settings.protect_columns])
            ),
        )
    prune_seconds = time.perf_counter() - prune_started

    frame_started = time.perf_counter()
    interaction_df = compact_interaction_frame(alternatives, choosers, counts)
    if skims is not None:
        set_skim_wrapper_targets(interaction_df, skims)
    frame_seconds = time.perf_counter() - frame_started

    utility_started = time.perf_counter()
    utilities, _ = interaction_simulate.eval_interaction_utilities(
        state,
        spec,
        interaction_df,
        locals_d,
        trace_label,
        None,
        estimator=None,
        log_alt_losers=False,
        compute_settings=compute_settings,
    )
    raw = np.asarray(utilities.utility, dtype=np.float32)
    if raw.shape != (len(alternatives),):
        raise ValueError("Phase 45 compiled utility result has an invalid shape")
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
                "component": str(component),
                "trace_label": str(trace_label),
                "chooser_rows": len(choosers),
                "alternative_rows": len(alternatives),
                "max_alternatives": int(counts.max()),
                "chooser_columns": len(choosers.columns),
                "alternative_columns": len(alternatives.columns),
                "interaction_columns": len(interaction_df.columns),
                "prune_seconds": prune_seconds,
                "frame_seconds": frame_seconds,
                "utility_seconds": utility_seconds,
                "padding_seconds": pad_seconds,
                "probability_seconds": probability_seconds,
                "choice_seconds": choice_seconds,
                "total_seconds": time.perf_counter() - started,
            }
        )
    return choices


def summarize_telemetry(events) -> dict:
    events = list(events)
    groups = {}
    for event in events:
        group = groups.setdefault(
            event["component"],
            {"calls": 0, "chooser_rows": 0, "alternative_rows": 0, "seconds": 0.0},
        )
        group["calls"] += 1
        group["chooser_rows"] += int(event["chooser_rows"])
        group["alternative_rows"] += int(event["alternative_rows"])
        group["seconds"] += float(event["total_seconds"])
    return {
        "calls": len(events),
        "chooser_rows": sum(int(item["chooser_rows"]) for item in events),
        "alternative_rows": sum(int(item["alternative_rows"]) for item in events),
        "groups": groups,
        "events": events,
    }
