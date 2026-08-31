"""Phase 45 resident sampler for the public one-zone destination programs.

The qualified programs all reduce to distance splines, a destination size
term, and (for shadow-priced location choice) two adjustment columns.  This
module evaluates that reviewed contract directly over the dense 1,454-zone
universe and reuses the Phase 41 exact inverse-CDF/duplicate CUDA runtime.
"""

from __future__ import annotations

import logging
import os
import re
import time

import numpy as np
import pandas as pd

from .cuda_backend import _cupy
from .cuda_skims import cuda_cube_from_activitysim
from .arithmetic_abi import float32_reduction_cuda, grouped_left_reduction
from .trip_destination_resident import (
    _compile_resident_kernels,
    _host_duplicate_contract,
    _pack_sample,
    _preserved_order_choices,
)


_UTILITY_KERNELS = {}
_TELEMETRY = []
logger = logging.getLogger(__name__)


class Phase45Unsupported(ValueError):
    pass


def reset_phase45_sampling_telemetry():
    _TELEMETRY.clear()


def phase45_sampling_telemetry():
    return list(_TELEMETRY)


def _feature_source(expression: str) -> str:
    text = re.sub(r"\s+", "", str(expression))
    distance = "dist"
    if text.startswith("_DIST@"):
        # Sharrow's ``name@expression`` row defines a reusable local feature;
        # it is not itself accumulated into utility despite the CSV sentinel 1.
        return "0.0f"
    if "income_segment']>=WORK_HIGH_SEGMENT_ID" in text:
        body = (
            "(((income[row] >= work_high) ? 1.0f : 0.0f) * ("
            + _distance_spline(text, distance)
            + "))"
        )
        return body
    if "skims['DIST']" in text or "_DIST" in text:
        return _distance_spline(text, distance)
    if "log1p" in text and "size_term" in text:
        return "size_log[alternative]"
    if "shadow_price_utility_adjustment" in text:
        return "shadow_utility[alternative]"
    if "size_term" in text and "==0" in text:
        return "(size_term[alternative] == 0.0f ? 1.0f : 0.0f)"
    raise Phase45Unsupported(f"unsupported sample expression: {expression}")


def _distance_spline(text: str, variable: str) -> str:
    match = re.search(r"(?:skims\['DIST'\]|_DIST)(?:-([0-9.]+))?\)??\.clip\(([^)]*)\)", text)
    if not match:
        if text in {"skims['DIST']", "_DIST", "_DIST@skims['DIST']"}:
            return variable
        raise Phase45Unsupported(f"unsupported distance expression: {text}")
    subtract = np.float32(float(match.group(1) or 0.0))
    args = match.group(2)
    lower = None
    upper = None
    for item in args.split(","):
        if not item:
            continue
        if "=" in item:
            name, value = item.split("=", 1)
            if name == "lower": lower = float(value)
            elif name == "upper": upper = float(value)
        elif lower is None:
            lower = float(item)
        elif upper is None:
            upper = float(item)
    literal = lambda value: f"{float(np.float32(value)):.9g}" + (
        "f" if "." in f"{float(np.float32(value)):.9g}" else ".0f"
    )
    code = variable if subtract == 0 else f"__fsub_rn({variable}, {literal(subtract)})"
    if lower is not None:
        code = f"(({code}) < {literal(lower)} ? {literal(lower)} : ({code}))"
    if upper is not None:
        code = f"(({code}) > {literal(upper)} ? {literal(upper)} : ({code}))"
    return code


def _compile_utility(cp, expressions):
    key = tuple(map(str, expressions))
    if key in _UTILITY_KERNELS:
        return _UTILITY_KERNELS[key], True
    features = [_feature_source(expression) for expression in expressions]
    intermediate = "const float intermediate[%d] = {%s};" % (
        len(features), ", ".join(features)
    )
    reduction = float32_reduction_cuda(
        grouped_left_reduction(len(features)),
        intermediate="intermediate",
        coefficients="coefficients",
    )
    source = r'''
extern "C" __global__ void phase45_dense_destination_utility(
    const float* distance, const int* origins, const int* alternative_ids, const float* size_term,
    const float* size_log, const float* shadow_utility, const int* income,
    const float* coefficients, float* utilities, int rows, int alternatives,
    int cube_width, int work_high)
{
    const long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long cells = (long long)rows * alternatives;
    if (cell >= cells) return;
    const int row = (int)(cell / alternatives);
    const int alternative = (int)(cell - (long long)row * alternatives);
    const float dist = distance[(long long)origins[row] * cube_width + alternative_ids[alternative]];
    INTERMEDIATE
    TERMS
    utilities[cell] = utility;
}
'''.replace("INTERMEDIATE", intermediate).replace("TERMS", reduction)
    kernel = cp.RawKernel(source, "phase45_dense_destination_utility", options=("--std=c++11", "--fmad=false"))
    _UTILITY_KERNELS[key] = kernel
    return kernel, False


def sample_destinations_resident(
    state, choosers, alternatives, spec, sample_size, alt_col_name, *, skims,
    trace_label, component, locals_d=None, zone_layer=None, compute_settings=None,
    work_high_segment_id=3,
):
    """Return ActivitySim's exact narrow sample for one reviewed segment."""
    started = time.perf_counter()
    if sample_size <= 0 or sample_size > 32 or len(spec.columns) != 1:
        raise Phase45Unsupported("resident sampler requires 1..32 draws and one coefficient column")
    if not hasattr(skims, "dataset") or hasattr(skims, "skim_dict"):
        raise Phase45Unsupported("resident sampler requires the public Sharrow skim wrapper")
    expressions = tuple(
        str(item[0] if isinstance(item, tuple) else item) for item in spec.index
    )
    coefficients = np.ascontiguousarray(spec.iloc[:, 0], dtype=np.float32)
    cp = _cupy()
    distance, dest_count, time_count, rank = cuda_cube_from_activitysim(skims, "DIST")
    if rank != 2:
        raise Phase45Unsupported("distance cube is not two-dimensional")
    origin_name = skims.orig_key
    origins = np.ascontiguousarray(choosers[origin_name], dtype=np.int32)
    if origins.min() < 0 or origins.max() >= dest_count:
        raise Phase45Unsupported("origin coordinates are outside the dense skim cube")
    size_source = np.asarray(alternatives["size_term"])
    shadow_size_source = np.asarray(
        alternatives.get(
            "shadow_price_size_term_adjustment",
            pd.Series(1.0, index=alternatives.index),
        )
    )
    size = np.ascontiguousarray(size_source, dtype=np.float32)
    # This 1,454-value feature is cheaper to calculate once than 274 million
    # times.  More importantly, using NumPy on the original column dtypes
    # reproduces Sharrow's log1p-then-float32-cast contract exactly.
    size_log = np.ascontiguousarray(
        np.log1p(size_source * shadow_size_source), dtype=np.float32
    )
    shadow_utility = np.ascontiguousarray(
        alternatives.get("shadow_price_utility_adjustment", pd.Series(0.0, index=alternatives.index)),
        dtype=np.float32,
    )
    income = np.ascontiguousarray(
        choosers.get("income_segment", pd.Series(0, index=choosers.index)), dtype=np.int32
    )
    alternative_ids = np.ascontiguousarray(alternatives.index, dtype=np.int32)
    if alternative_ids.min() < 0 or alternative_ids.max() >= dest_count:
        raise Phase45Unsupported("alternative coordinates are outside the dense skim cube")
    utility = cp.empty((len(choosers), len(alternatives)), dtype=cp.float32)
    kernel, cache_hit = _compile_utility(cp, expressions)
    block = 256
    cells = utility.size
    kernel(
        ((cells + block - 1) // block,), (block,),
        (distance, cp.asarray(origins), cp.asarray(alternative_ids), cp.asarray(size), cp.asarray(size_log),
         cp.asarray(shadow_utility), cp.asarray(income), cp.asarray(coefficients), utility,
         np.int32(len(choosers)), np.int32(len(alternatives)), np.int32(dest_count),
         np.int32(work_high_segment_id)),
    )
    cp.cuda.Stream.null.synchronize()
    utility_complete = time.perf_counter()

    diagnostic_exact_values = None
    diagnostic_match = os.environ.get("CHOICEFORGE_PHASE45_DIAGNOSTIC_MATCH")
    if diagnostic_match and diagnostic_match in str(trace_label):
        from activitysim.core import interaction_simulate

        exact, _ = interaction_simulate.eval_interaction_utilities(
            state,
            spec,
            choosers,
            locals_d,
            str(trace_label) + ".phase45_full_diagnostic",
            None,
            estimator=None,
            log_alt_losers=False,
            extra_data=alternatives,
            zone_layer=zone_layer,
            compute_settings=compute_settings,
        )
        exact_values = np.asarray(exact.utility, dtype=np.float32).reshape(
            len(choosers), len(alternatives)
        )
        diagnostic_exact_values = exact_values
        approximate_values = cp.asnumpy(utility)
        differences = np.abs(approximate_values - exact_values)
        logger.warning(
            "Phase45 utility diagnostic %s cells=%d max_abs=%.9g p99=%.9g "
            "p999=%.9g mismatches_gt_2e6=%d mismatches_gt_4e6=%d",
            trace_label,
            differences.size,
            float(np.max(differences)),
            float(np.quantile(differences, 0.99)),
            float(np.quantile(differences, 0.999)),
            int(np.count_nonzero(differences > 2.0e-6)),
            int(np.count_nonzero(differences > 4.0e-6)),
        )

    random_draws = np.ascontiguousarray(
        state.get_rn_generator().random_for_df(choosers, n=sample_size), dtype=np.float64
    )
    choices = cp.empty((len(choosers), sample_size), dtype=cp.int32)
    probabilities = cp.empty((len(choosers), sample_size), dtype=cp.float32)
    guard = cp.zeros(len(choosers), dtype=cp.uint8)
    bad = cp.zeros(len(choosers), dtype=cp.uint8)
    risk = cp.zeros(len(choosers), dtype=cp.float32)
    choice_kernel, duplicate_kernel = _compile_resident_kernels(cp, len(alternatives))
    # Manual spline/log1p code is bounded against Sharrow, then the existing
    # Phase 40 interval proof decides which rows are invariant.  Only risky
    # rows cross back to the authoritative evaluator.
    # ActivitySim's probability path subtracts each row maximum before its
    # float32 exponential.  This is numerically meaningful even though the
    # mathematical probabilities are shift invariant.
    shifted_utility = utility - cp.max(utility, axis=1, keepdims=True)
    # The grouped OpenBLAS reduction and host-precomputed NumPy log1p feature
    # make this utility surface bit-identical to Sharrow.  The choice kernel's
    # fixed CDF reserve still catches the rare exp/divide rounding boundary.
    # Mode 3 never reads the error surface, so reuse this already-resident
    # pointer instead of allocating another dense matrix.
    error_bounds = shifted_utility
    choice_kernel(
        ((len(choosers) + 127) // 128,), (128,),
        (shifted_utility, error_bounds, cp.asarray(random_draws), choices, probabilities,
         guard, bad, risk, np.int32(len(choosers)), np.int32(len(alternatives)),
         np.int32(sample_size), np.int32(3)),
    )
    first = cp.empty_like(choices, dtype=cp.uint8)
    counts = cp.empty_like(choices, dtype=cp.uint32)
    duplicate_kernel(
        ((len(choosers) + 127) // 128,), (128,),
        (choices, first, counts, np.int32(len(choosers)), np.int32(sample_size)),
    )
    cp.cuda.Stream.null.synchronize()
    if int(cp.count_nonzero(bad).get()):
        raise Phase45Unsupported("resident choice produced zero or invalid probabilities")
    host_choices = cp.asnumpy(choices)
    host_choices = alternative_ids[host_choices]
    host_probabilities = cp.asnumpy(probabilities)
    host_first = cp.asnumpy(first)
    host_counts = cp.asnumpy(counts)
    if diagnostic_exact_values is not None:
        from activitysim.core import logit

        diagnostic_probs = logit.utils_to_probs(
            state,
            pd.DataFrame(diagnostic_exact_values.copy(), index=choosers.index),
            allow_zero_probs=False,
            trace_label=str(trace_label) + ".phase45_probability_diagnostic",
            trace_choosers=choosers,
            overflow_protection=True,
        ).to_numpy(copy=False)
        diagnostic_choices, diagnostic_choice_probs = _preserved_order_choices(
            diagnostic_probs, random_draws, alternative_ids
        )
        same_choice = host_choices == diagnostic_choices
        mismatch_positions = np.argwhere(~same_choice)
        mismatch_boundary_max = 0.0
        if len(mismatch_positions):
            diagnostic_cdf = np.cumsum(
                diagnostic_probs, axis=1, dtype=np.float64
            )
            margins = []
            alternative_positions = {
                int(value): position for position, value in enumerate(alternative_ids)
            }
            for row, draw in mismatch_positions:
                position = alternative_positions[int(host_choices[row, draw])]
                upper = diagnostic_cdf[row, position]
                lower = 0.0 if position == 0 else diagnostic_cdf[row, position - 1]
                margins.append(min(
                    abs(float(random_draws[row, draw]) - upper),
                    abs(float(random_draws[row, draw]) - lower),
                ))
            mismatch_boundary_max = float(np.max(margins))
        positive = (
            same_choice
            & (host_probabilities > 0.0)
            & (diagnostic_choice_probs > 0.0)
        )
        probability_log_error = np.abs(
            np.log(host_probabilities[positive])
            - np.log(diagnostic_choice_probs[positive])
        )
        logger.warning(
            "Phase45 probability diagnostic %s choice_mismatches=%d "
            "unguarded_mismatch_rows=%d "
            "max_exact_boundary_margin=%.9g "
            "max_selected_log_error=%.9g p999_selected_log_error=%.9g",
            trace_label,
            int(np.count_nonzero(~same_choice)),
            int(np.count_nonzero(
                np.any(~same_choice, axis=1)
                & ~cp.asnumpy(guard).astype(bool, copy=False)
            )),
            mismatch_boundary_max,
            float(np.max(probability_log_error)),
            float(np.quantile(probability_log_error, 0.999)),
        )
        if os.environ.get("CHOICEFORGE_PHASE45_DIAGNOSTIC_STOP", "0") == "1":
            raise RuntimeError("Phase45 utility/probability diagnostic completed")
    guard_host = cp.asnumpy(guard).astype(bool, copy=False)
    guard_count = int(np.count_nonzero(guard_host))
    if guard_count:
        from activitysim.core import logit

        exact_choosers = choosers.iloc[np.flatnonzero(guard_host)]
        # Utilities are already exact and row-max shifted.  Transfer only the
        # ambiguous rows and let NumPy reproduce ActivitySim's final exp,
        # reduction and division; no Sharrow flow construction is necessary.
        exact_values = cp.asnumpy(utility[cp.asarray(guard_host)])
        exact_probs = logit.utils_to_probs(
            state,
            pd.DataFrame(exact_values, index=exact_choosers.index),
            allow_zero_probs=False,
            trace_label=str(trace_label) + ".phase45_exact_guard",
            trace_choosers=exact_choosers,
            overflow_protection=True,
        ).to_numpy(copy=False)
        exact_choices, exact_probabilities = _preserved_order_choices(
            exact_probs, random_draws[guard_host], alternative_ids
        )
        host_choices[guard_host] = exact_choices
        host_probabilities[guard_host] = exact_probabilities
        exact_first, exact_counts = _host_duplicate_contract(exact_choices)
        host_first[guard_host] = exact_first
        host_counts[guard_host] = exact_counts
    sample = _pack_sample(
        choosers, host_choices, host_probabilities, random_draws,
        host_first, host_counts, alt_col_name,
    )
    finished = time.perf_counter()
    _TELEMETRY.append({
        "component": str(component), "trace_label": str(trace_label),
        "chooser_rows": len(choosers), "alternatives": len(alternatives),
        "utility_cells": int(cells), "sampled_rows": len(sample),
        "random_draws": int(random_draws.size), "program_terms": len(expressions),
        "program_cache_hit": cache_hit, "utility_seconds": utility_complete - started,
        "exact_guard_rows": guard_count,
        "total_seconds": finished - started, "fallback": False,
    })
    return sample
