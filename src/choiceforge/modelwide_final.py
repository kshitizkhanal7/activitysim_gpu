"""Strict Phase 47 CUDA runtime for model-wide sampled final choice."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .arithmetic_abi import float32_reduction_cuda, grouped_left_reduction
from .cuda_backend import _cupy
from .cuda_skims import cuda_cube_from_activitysim
from .modelwide_sampling import (
    _compile_phase46_choice,
    _feature_source,
    numpy_preserved_order_choices,
)
from .trip_destination_final import ragged_offsets


_FINAL_UTILITY_KERNELS = {}
_TELEMETRY = []


SCHOOL_FINAL_EXPRESSIONS = (
    "_DIST@skims['DIST']",
    "@_DIST.clip(0,1)",
    "@(_DIST-1).clip(0,1)",
    "@(_DIST-2).clip(0,3)",
    "@(_DIST-5).clip(0,10)",
    "@(_DIST-15.0).clip(0)",
    "@(df['size_term'] * df['shadow_price_size_term_adjustment']).apply(np.log1p)",
    "@df['shadow_price_utility_adjustment']",
    "@df['size_term']==0",
    "mode_choice_logsum",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
)

WORK_FINAL_EXPRESSIONS = (
    "_DIST@skims['DIST']",
    "@_DIST.clip(0,1)",
    "@(_DIST-1).clip(0,1)",
    "@(_DIST-2).clip(0,3)",
    "@(_DIST-5).clip(0,10)",
    "@(_DIST-15.0).clip(0)",
    "@(df['income_segment']>=WORK_HIGH_SEGMENT_ID) * _DIST.clip(upper=5)",
    "@(df['income_segment']>=WORK_HIGH_SEGMENT_ID) * (_DIST-5).clip(0)",
    "@(df['size_term'] * df['shadow_price_size_term_adjustment']).apply(np.log1p)",
    "@df['shadow_price_utility_adjustment']",
    "@df['size_term']==0",
    "mode_choice_logsum",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
)

TOUR_FINAL_EXPRESSIONS = (
    "@skims['DIST'].clip(0,1)",
    "@(skims['DIST']-1).clip(0,1)",
    "@(skims['DIST']-2).clip(0,3)",
    "@(skims['DIST']-5).clip(0,10)",
    "@(skims['DIST']-15.0).clip(0)",
    "@df['size_term'].apply(np.log1p)",
    "@df['size_term']==0",
    "mode_choice_logsum",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
)

ATWORK_FINAL_EXPRESSIONS = TOUR_FINAL_EXPRESSIONS[:6] + (
    "size_term==0",
    "mode_choice_logsum",
    "@np.minimum(np.log(df.pick_count/df.prob), 60)",
)

PUBLIC_FINAL_PROGRAMS = (
    SCHOOL_FINAL_EXPRESSIONS,
    WORK_FINAL_EXPRESSIONS,
    TOUR_FINAL_EXPRESSIONS,
    ATWORK_FINAL_EXPRESSIONS,
)
PUBLIC_FINAL_WIDTHS = (21, 25, 29, 30)


def reset_phase47_telemetry():
    _TELEMETRY.clear()


def phase47_telemetry():
    return list(_TELEMETRY)


def _spec_expressions(spec) -> tuple[str, ...]:
    from activitysim.core import simulate

    if isinstance(spec.index, pd.MultiIndex):
        values = spec.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
    else:
        values = spec.index
    return tuple(str(value) for value in values)


def _final_feature_source(expression: str) -> str:
    compact = "".join(str(expression).split())
    if compact == "mode_choice_logsum":
        return "mode_logsum[alternative_row]"
    if "pick_count/df.prob" in compact:
        return "correction[alternative_row]"
    return _feature_source(expression).replace("[alternative]", "[alternative_row]")


def _compile_final_utility(cp, expressions):
    key = tuple(map(str, expressions))
    if key in _FINAL_UTILITY_KERNELS:
        return _FINAL_UTILITY_KERNELS[key], True
    if key not in PUBLIC_FINAL_PROGRAMS:
        raise ValueError("Phase 47 final expression ABI is not reviewed")
    features = [_final_feature_source(expression) for expression in key]
    intermediate = "const float intermediate[%d] = {%s};" % (
        len(features), ", ".join(features)
    )
    reduction = float32_reduction_cuda(
        grouped_left_reduction(len(features)),
        intermediate="intermediate",
        coefficients="coefficients",
    )
    source = r'''
extern "C" __global__ void phase47_compact_final_utility(
    const float* distance,
    const int* origins,
    const int* alternative_ids,
    const int* row_ids,
    const long long* offsets,
    const float* size_term,
    const float* size_log,
    const float* shadow_utility,
    const int* income,
    const float* mode_logsum,
    const float* correction,
    const float* coefficients,
    float* padded_utilities,
    long long alternative_rows,
    int width,
    int cube_width,
    int work_high)
{
    const long long alternative_row =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (alternative_row >= alternative_rows) return;
    const int row = row_ids[alternative_row];
    const int position = (int)(alternative_row - offsets[row]);
    const float dist = distance[
        (long long)origins[row] * cube_width + alternative_ids[alternative_row]];
    INTERMEDIATE
    TERMS
    padded_utilities[(long long)row * width + position] = utility;
}
'''.replace("INTERMEDIATE", intermediate).replace("TERMS", reduction)
    kernel = cp.RawKernel(
        source,
        "phase47_compact_final_utility",
        options=("--std=c++11", "--fmad=false"),
    )
    kernel.compile()
    _FINAL_UTILITY_KERNELS[key] = kernel
    return kernel, False


def prewarm_phase47_public_runtime(cp=None) -> dict:
    cp = cp or _cupy()
    started = time.perf_counter()
    compiled = 0
    for expressions in PUBLIC_FINAL_PROGRAMS:
        _, hit = _compile_final_utility(cp, expressions)
        compiled += int(not hit)
    for width in PUBLIC_FINAL_WIDTHS:
        _compile_phase46_choice(cp, width)
    cp.cuda.Stream.null.synchronize()
    return {
        "programs": len(PUBLIC_FINAL_PROGRAMS),
        "widths": list(PUBLIC_FINAL_WIDTHS),
        "new_programs_compiled": compiled,
        "seconds": time.perf_counter() - started,
    }


def _cpu_reference_utility(
    state, choosers, alternatives, counts, spec, skims, locals_d,
    trace_label, compute_settings,
):
    from activitysim.core import interaction_simulate
    from activitysim.core.simulate import set_skim_wrapper_targets
    from .modelwide_choice import compact_interaction_frame

    frame = compact_interaction_frame(alternatives, choosers, counts)
    if skims is not None:
        set_skim_wrapper_targets(frame, skims)
    utilities, _ = interaction_simulate.eval_interaction_utilities(
        state, spec, frame, locals_d, trace_label + ".phase47_shadow", None,
        estimator=None, log_alt_losers=False, compute_settings=compute_settings,
    )
    return np.asarray(utilities.utility, dtype=np.float32)


def device_compact_interaction_sample_simulate(
    state,
    choosers,
    alternatives,
    spec,
    choice_column,
    *,
    allow_zero_probs=False,
    zero_prob_choice_val=None,
    want_logsums=False,
    skims=None,
    locals_d=None,
    trace_label=None,
    compute_settings=None,
    telemetry=None,
    component="unknown",
    service,
    work_high_segment_id=3,
):
    """Evaluate reviewed compact final choice on CUDA with exact adjudication."""
    from activitysim.core import logit

    started = time.perf_counter()
    expressions = _spec_expressions(spec)
    coefficients = np.ascontiguousarray(spec.iloc[:, 0], dtype=np.float32)
    if expressions not in PUBLIC_FINAL_PROGRAMS:
        raise ValueError("Phase 47 received an unknown public final program")
    if not hasattr(skims, "dataset") or hasattr(skims, "skim_dict"):
        raise ValueError("Phase 47 requires the public Sharrow skim wrapper")
    offsets, counts = ragged_offsets(alternatives, choosers)
    width = int(counts.max())
    if width not in PUBLIC_FINAL_WIDTHS:
        raise ValueError(f"Phase 47 unsupported ragged width: {width}")
    cp = service.cp
    distance, destination_count, _, rank = cuda_cube_from_activitysim(skims, "DIST")
    if rank != 2:
        raise ValueError("Phase 47 distance skim is not two-dimensional")
    origins = np.ascontiguousarray(choosers[skims.orig_key], dtype=np.int32)
    alternative_ids = np.ascontiguousarray(
        alternatives[choice_column], dtype=np.int32
    )
    if (
        origins.min() < 0
        or origins.max() >= destination_count
        or alternative_ids.min() < 0
        or alternative_ids.max() >= destination_count
    ):
        raise ValueError("Phase 47 skim coordinates are outside the dense cube")
    row_ids = np.repeat(
        np.arange(len(choosers), dtype=np.int32), counts.astype(np.int32)
    )
    size_source = np.asarray(alternatives["size_term"])
    shadow_size_source = np.asarray(
        alternatives.get(
            "shadow_price_size_term_adjustment",
            pd.Series(1.0, index=alternatives.index),
        )
    )
    size = np.ascontiguousarray(size_source, dtype=np.float32)
    size_log = np.ascontiguousarray(
        np.log1p(size_source * shadow_size_source), dtype=np.float32
    )
    shadow_utility = np.ascontiguousarray(
        alternatives.get(
            "shadow_price_utility_adjustment",
            pd.Series(0.0, index=alternatives.index),
        ),
        dtype=np.float32,
    )
    income = np.ascontiguousarray(
        choosers.get("income_segment", pd.Series(0, index=choosers.index)),
        dtype=np.int32,
    )
    mode_logsum = np.ascontiguousarray(
        alternatives["mode_choice_logsum"], dtype=np.float32
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.ascontiguousarray(
            np.minimum(
                np.log(
                    np.asarray(alternatives["pick_count"])
                    / np.asarray(alternatives["prob"])
                ),
                60,
            ),
            dtype=np.float32,
        )
    if not np.isfinite(correction).all():
        raise ValueError("Phase 47 sample correction is not finite")
    prepared = time.perf_counter()

    workspace = service.final_workspace(
        len(choosers), len(alternatives), width
    )
    padded = workspace["utilities"]
    padded.fill(np.float32(-999.0))
    utility_kernel, cache_hit = _compile_final_utility(cp, expressions)
    block = 256
    utility_kernel(
        ((len(alternatives) + block - 1) // block,),
        (block,),
        (
            distance,
            cp.asarray(origins),
            cp.asarray(alternative_ids),
            cp.asarray(row_ids),
            cp.asarray(offsets),
            cp.asarray(size),
            cp.asarray(size_log),
            cp.asarray(shadow_utility),
            cp.asarray(income),
            cp.asarray(mode_logsum),
            cp.asarray(correction),
            cp.asarray(coefficients),
            padded,
            np.int64(len(alternatives)),
            np.int32(width),
            np.int32(destination_count),
            np.int32(work_high_segment_id),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    utility_complete = time.perf_counter()

    padded_host = cp.asnumpy(padded)
    shadow = os.environ.get("CHOICEFORGE_PHASE47_SHADOW", "0") == "1"
    utility_mismatches = 0
    utility_max_abs = 0.0
    if shadow:
        reference = _cpu_reference_utility(
            state, choosers, alternatives, counts, spec, skims, locals_d or {},
            str(trace_label), compute_settings,
        )
        actual = np.concatenate(
            [padded_host[row, : count] for row, count in enumerate(counts)]
        )
        utility_mismatches = int(np.count_nonzero(
            actual.view(np.uint32) != reference.view(np.uint32)
        ))
        utility_max_abs = float(np.max(np.abs(actual - reference)))
        print(
            "PHASE47_UTILITY_SHADOW "
            + repr({
                "trace_label": str(trace_label),
                "cells": len(reference),
                "bit_mismatches": utility_mismatches,
                "max_abs": utility_max_abs,
            }),
            flush=True,
        )
        if utility_mismatches:
            raise RuntimeError("Phase 47 utility shadow is not bit-identical")
    transfer_complete = time.perf_counter()

    utilities_df = pd.DataFrame(padded_host, index=choosers.index)
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
    probability_complete = time.perf_counter()

    draws, device_draws = service.random_for_df(state, choosers, 1)
    row_maxima = workspace["row_maxima"]
    cp.max(padded, axis=1, out=row_maxima)
    weight_kernel, choice_kernel, _ = _compile_phase46_choice(cp, width)
    cells = padded.size
    weight_kernel(
        ((cells + 255) // 256,),
        (256,),
        (
            padded,
            row_maxima,
            workspace["weights"],
            np.int64(cells),
            np.int32(width),
        ),
    )
    choice_kernel(
        ((len(choosers) + 127) // 128,),
        (128,),
        (
            workspace["weights"],
            device_draws,
            workspace["positions"],
            workspace["selected_probabilities"],
            workspace["guard"],
            workspace["bad"],
            np.int32(len(choosers)),
            np.int32(width),
            np.int32(1),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    if int(cp.count_nonzero(workspace["bad"]).get()):
        raise ValueError("Phase 47 produced zero or invalid probabilities")
    gpu_positions = cp.asnumpy(workspace["positions"])
    guard = cp.asnumpy(workspace["guard"]).astype(bool, copy=False)
    alternatives_by_position = np.arange(width, dtype=np.int32)
    selected_positions = gpu_positions
    if shadow:
        exact_positions, _ = numpy_preserved_order_choices(
            probs.to_numpy(copy=False), draws, alternatives_by_position
        )
        exact_positions = exact_positions[:, 0]
        mismatches = gpu_positions != exact_positions
        unguarded_mismatches = int(np.count_nonzero(mismatches & ~guard))
        if unguarded_mismatches:
            raise RuntimeError(
                f"Phase 47 has {unguarded_mismatches} unguarded final-choice mismatches"
            )
        selected_positions[guard] = exact_positions[guard]
        pre_guard_mismatches = int(np.count_nonzero(mismatches))
    else:
        guard_rows = np.flatnonzero(guard)
        pre_guard_mismatches = 0
        unguarded_mismatches = 0
        if len(guard_rows):
            exact_guard_positions, _ = numpy_preserved_order_choices(
                probs.to_numpy(copy=False)[guard_rows],
                draws[guard_rows],
                alternatives_by_position,
            )
            exact_guard_positions = exact_guard_positions[:, 0]
            pre_guard_mismatches = int(np.count_nonzero(
                selected_positions[guard_rows] != exact_guard_positions
            ))
            selected_positions[guard_rows] = exact_guard_positions
    choice_complete = time.perf_counter()

    selected_rows = offsets[:-1] + selected_positions.astype(np.int64)
    choices = pd.Series(
        alternatives[choice_column].to_numpy(copy=False)[selected_rows],
        index=choosers.index,
    )
    if zero_probs is not None and zero_probs.any() and zero_prob_choice_val is not None:
        choices.loc[zero_probs] = zero_prob_choice_val
    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = logsums
    finished = time.perf_counter()
    event = {
        "component": str(component),
        "trace_label": str(trace_label),
        "chooser_rows": len(choosers),
        "alternative_rows": len(alternatives),
        "max_alternatives": width,
        "program_terms": len(expressions),
        "program_cache_hit": cache_hit,
        "prepare_seconds": prepared - started,
        "utility_seconds": utility_complete - prepared,
        "transfer_seconds": transfer_complete - utility_complete,
        "probability_logsum_seconds": probability_complete - transfer_complete,
        "choice_seconds": choice_complete - probability_complete,
        "pack_seconds": finished - choice_complete,
        "total_seconds": finished - started,
        "exact_guard_rows": int(np.count_nonzero(guard)),
        "pre_guard_mismatches": pre_guard_mismatches,
        "unguarded_mismatches": unguarded_mismatches,
        "exhaustive_choice_shadow": shadow,
        "utility_shadow_bit_mismatches": utility_mismatches,
        "utility_shadow_max_abs": utility_max_abs,
        "runtime": "phase47_device_final",
    }
    _TELEMETRY.append(event)
    if telemetry is not None:
        telemetry.append(event)
    return choices


def summarize_phase47_telemetry(events=None) -> dict:
    events = list(_TELEMETRY if events is None else events)
    return {
        "calls": len(events),
        "chooser_rows": sum(item["chooser_rows"] for item in events),
        "alternative_rows": sum(item["alternative_rows"] for item in events),
        "guard_rows": sum(item["exact_guard_rows"] for item in events),
        "pre_guard_mismatches": sum(item["pre_guard_mismatches"] for item in events),
        "seconds": sum(item["total_seconds"] for item in events),
        "events": events,
    }
