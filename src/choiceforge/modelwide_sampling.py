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
from .arithmetic_abi import (
    float32_reduction_cuda,
    grouped_left_reduction,
    numpy_float32_choice_cuda_helpers,
    numpy_float32_pairwise_sum_cuda_helpers,
)
from .trip_destination_resident import (
    _compile_resident_kernels,
    _host_duplicate_contract,
    _pack_sample,
    _preserved_order_choices,
)


_UTILITY_KERNELS = {}
_PHASE46_CHOICE_KERNELS = {}
_PHASE46_WEIGHT_KERNEL = None
_PHASE46_DUPLICATE_KERNEL = None
_TELEMETRY = []
logger = logging.getLogger(__name__)


_PHASE46_PUBLIC_PROGRAMS = (
    (
        "_DIST@skims['DIST']",
        "@_DIST.clip(0,1)",
        "@(_DIST-1).clip(0,1)",
        "@(_DIST-2).clip(0,3)",
        "@(_DIST-5).clip(0,10)",
        "@(_DIST-15.0).clip(0)",
        "@(df['size_term'] * df['shadow_price_size_term_adjustment']).apply(np.log1p)",
        "@df['shadow_price_utility_adjustment']",
        "@df['size_term']==0",
    ),
    (
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
    ),
    (
        "@skims['DIST'].clip(0,1)",
        "@(skims['DIST']-1).clip(0,1)",
        "@(skims['DIST']-2).clip(0,3)",
        "@(skims['DIST']-5).clip(0,10)",
        "@(skims['DIST']-15.0).clip(0)",
        "@df['size_term'].apply(np.log1p)",
        "@df['size_term']==0",
    ),
    (
        "@skims['DIST'].clip(0,1)",
        "@(skims['DIST']-1).clip(0,1)",
        "@(skims['DIST']-2).clip(0,3)",
        "@(skims['DIST']-5).clip(0,10)",
        "@(skims['DIST']-15.0).clip(0)",
        "@df['size_term'].apply(np.log1p)",
        "size_term==0",
    ),
)


class Phase45Unsupported(ValueError):
    pass


def reset_phase45_sampling_telemetry():
    _TELEMETRY.clear()


def phase45_sampling_telemetry():
    return list(_TELEMETRY)


def _pack_sample_phase46(
    choosers,
    choices,
    probabilities,
    random_draws,
    first_occurrence,
    pick_counts,
    alt_col_name,
):
    """Pack chooser-major samples without pandas' two global stable sorts."""
    valid = np.asarray(first_occurrence, dtype=bool)
    sentinel = np.iinfo(np.int32).max
    # ActivitySim's output contract is chooser index first, then alternative
    # id.  Choosers are already monotonic and each row is at most 30 draws, so
    # sort those tiny rows directly and keep invalid duplicates at the end.
    order = np.argsort(
        np.where(valid, choices, sentinel), axis=1, kind="stable"
    )
    sorted_valid = np.take_along_axis(valid, order, axis=1)
    selected = sorted_valid.reshape(-1)
    counts_per_chooser = sorted_valid.sum(axis=1, dtype=np.int32)

    def compact(values):
        return np.take_along_axis(values, order, axis=1).reshape(-1)[selected]

    index_name = choosers.index.name
    index = pd.Index(
        np.repeat(np.asarray(choosers.index), counts_per_chooser),
        name=index_name,
    )
    return pd.DataFrame(
        {
            alt_col_name: compact(choices),
            "rand": compact(random_draws),
            "prob": compact(probabilities).astype(np.float32, copy=False),
            "pick_count": compact(pick_counts).astype(np.uint32, copy=False),
        },
        index=index,
        copy=False,
    )


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


def _compile_phase46_choice(cp, alternatives):
    """Compile the one-exp-per-cell exact Phase 46 probability runtime."""
    global _PHASE46_WEIGHT_KERNEL, _PHASE46_DUPLICATE_KERNEL
    alternatives = int(alternatives)
    if _PHASE46_WEIGHT_KERNEL is None:
        exp_helpers = numpy_float32_choice_cuda_helpers(1)
        source = exp_helpers + r'''
extern "C" __global__ void phase46_destination_weights(
    const float* utilities,
    const float* row_maxima,
    float* weights,
    long long cells,
    int alternative_count)
{
    const long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < cells) {
        const int row = (int)(cell / alternative_count);
        weights[cell] = numpy_avx2_expf(
            __fsub_rn(utilities[cell], row_maxima[row]));
    }
}
'''
        _PHASE46_WEIGHT_KERNEL = cp.RawKernel(
            source,
            "phase46_destination_weights",
            options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
        )
        _PHASE46_WEIGHT_KERNEL.compile()
    if _PHASE46_DUPLICATE_KERNEL is None:
        duplicate_source = r'''
extern "C" __global__ void phase46_duplicate_counts(
    const int* choices,
    unsigned char* first_occurrence,
    unsigned int* pick_counts,
    int chooser_rows,
    int sample_size)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= chooser_rows) return;
    const long long base = (long long)row * sample_size;
    for (int draw = 0; draw < sample_size; ++draw) {
        bool first = true;
        unsigned int count = 0;
        const int choice = choices[base + draw];
        for (int other = 0; other < sample_size; ++other) {
            if (choices[base + other] == choice) {
                ++count;
                if (other < draw) first = false;
            }
        }
        first_occurrence[base + draw] = first ? 1 : 0;
        pick_counts[base + draw] = count;
    }
}
'''
        _PHASE46_DUPLICATE_KERNEL = cp.RawKernel(
            duplicate_source,
            "phase46_duplicate_counts",
            options=("--std=c++11",),
        )
        _PHASE46_DUPLICATE_KERNEL.compile()
    if alternatives not in _PHASE46_CHOICE_KERNELS:
        sum_helpers = numpy_float32_pairwise_sum_cuda_helpers(alternatives)
        source = sum_helpers + r'''
extern "C" __global__ void phase46_destination_inverse_cdf(
    const float* weights,
    const double* random_draws,
    int* choices,
    float* choice_probabilities,
    unsigned char* guard_rows,
    unsigned char* bad_rows,
    int chooser_rows,
    int alternative_count,
    int sample_size)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= chooser_rows) return;
    if (sample_size <= 0 || sample_size > 32) {
        bad_rows[row] = 1;
        return;
    }
    const long long base = (long long)row * alternative_count;
    const long long draw_base = (long long)row * sample_size;
    int order[32];
    double sorted_draws[32];
    for (int draw = 0; draw < sample_size; ++draw) {
        const double value = random_draws[draw_base + draw];
        int position = draw;
        while (position > 0 && value < sorted_draws[position - 1]) {
            sorted_draws[position] = sorted_draws[position - 1];
            order[position] = order[position - 1];
            --position;
        }
        sorted_draws[position] = value;
        order[position] = draw;
    }
    const float total = PAIRWISE_SUM(weights + base);
    if (!(total > 0.0f) || !isfinite(total)) {
        bad_rows[row] = 1;
        return;
    }
    const double reserve = 5.0e-7;
    int sorted_position = 0;
    int last_nontrivial = alternative_count - 1;
    double prefix = 0.0;
    bool guarded = false;
    for (int alternative = 0; alternative < alternative_count; ++alternative) {
        const float weight = weights[base + alternative];
        const float probability = weight > 0.0f ? weight / total : 0.0f;
        const double previous = prefix;
        prefix += (double)probability;
        if (probability >= 1.0e-30f) last_nontrivial = alternative;
        while (sorted_position < sample_size &&
               prefix > sorted_draws[sorted_position]) {
            const int original = order[sorted_position];
            const double random_draw = sorted_draws[sorted_position];
            choices[draw_base + original] = alternative;
            choice_probabilities[draw_base + original] = probability;
            if (fabs(prefix - random_draw) <= reserve ||
                fabs(random_draw - previous) <= reserve) guarded = true;
            ++sorted_position;
        }
        if (sorted_position == sample_size) break;
    }
    while (sorted_position < sample_size) {
        const int original = order[sorted_position];
        const float probability = weights[base + last_nontrivial] / total;
        choices[draw_base + original] = last_nontrivial;
        choice_probabilities[draw_base + original] = probability;
        guarded = true;
        ++sorted_position;
    }
    guard_rows[row] = guarded ? 1 : 0;
}
'''.replace("PAIRWISE_SUM", f"numpy_pairwise_sum_{alternatives}")
        kernel = cp.RawKernel(
            source,
            "phase46_destination_inverse_cdf",
            options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
        )
        kernel.compile()
        _PHASE46_CHOICE_KERNELS[alternatives] = kernel
    return (
        _PHASE46_WEIGHT_KERNEL,
        _PHASE46_CHOICE_KERNELS[alternatives],
        _PHASE46_DUPLICATE_KERNEL,
    )


def prewarm_phase46_public_runtime(cp=None) -> dict:
    """Compile all four reviewed public program shapes before model steps."""
    from activitysim.core.choosing import sample_choices_maker_preserve_ordering

    cp = cp or _cupy()
    started = time.perf_counter()
    compiled = 0
    for expressions in _PHASE46_PUBLIC_PROGRAMS:
        _, cache_hit = _compile_utility(cp, expressions)
        compiled += int(not cache_hit)
    _compile_phase46_choice(cp, 1454)
    # The exact boundary is sparse, but its authoritative ActivitySim Numba
    # implementation must not pay a lazy compile inside the first model step.
    sample_choices_maker_preserve_ordering(
        np.ones((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float64),
        np.zeros(1, dtype=np.int32),
    )
    cp.cuda.Stream.null.synchronize()
    return {
        "programs": len(_PHASE46_PUBLIC_PROGRAMS),
        "new_programs_compiled": compiled,
        "seconds": time.perf_counter() - started,
    }


def sample_destinations_resident(
    state, choosers, alternatives, spec, sample_size, alt_col_name, *, skims,
    trace_label, component, locals_d=None, zone_layer=None, compute_settings=None,
    work_high_segment_id=3, service=None,
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
    workspace = (
        service.sample_workspace(len(choosers), len(alternatives), sample_size)
        if service is not None
        else None
    )
    utility = (
        workspace["utility"]
        if workspace is not None
        else cp.empty((len(choosers), len(alternatives)), dtype=cp.float32)
    )
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

    if service is None:
        random_draws = np.ascontiguousarray(
            state.get_rn_generator().random_for_df(choosers, n=sample_size),
            dtype=np.float64,
        )
        device_random = cp.asarray(random_draws)
        choices = cp.empty((len(choosers), sample_size), dtype=cp.int32)
        probabilities = cp.empty((len(choosers), sample_size), dtype=cp.float32)
        guard = cp.zeros(len(choosers), dtype=cp.uint8)
        bad = cp.zeros(len(choosers), dtype=cp.uint8)
        risk = cp.zeros(len(choosers), dtype=cp.float32)
    else:
        random_draws, device_random = service.random_for_df(
            state, choosers, sample_size
        )
        choices = workspace["choices"]
        probabilities = workspace["probabilities"]
        guard = workspace["guard"]
        bad = workspace["bad"]
        risk = workspace["risk"]
    choice_kernel = None
    duplicate_kernel = None
    if service is None:
        choice_kernel, duplicate_kernel = _compile_resident_kernels(
            cp, len(alternatives)
        )
    # Manual spline/log1p code is bounded against Sharrow, then the existing
    # Phase 40 interval proof decides which rows are invariant.  Only risky
    # rows cross back to the authoritative evaluator.
    # ActivitySim's probability path subtracts each row maximum before its
    # float32 exponential.  This is numerically meaningful even though the
    # mathematical probabilities are shift invariant.
    if service is None:
        shifted_utility = utility - cp.max(utility, axis=1, keepdims=True)
    else:
        # Keep the authoritative, unshifted utility surface intact for sparse
        # exact adjudication.  Phase 45's CPU probability implementation does
        # its own row-max subtraction; shifting this buffer in place changed
        # a boundary decision even though the operation is mathematically
        # invariant.  The reusable risk vector doubles as row-max workspace.
        row_maxima = risk
        cp.max(utility, axis=1, out=row_maxima)
        shifted_utility = utility
    # The grouped OpenBLAS reduction and host-precomputed NumPy log1p feature
    # make this utility surface bit-identical to Sharrow.  The choice kernel's
    # fixed CDF reserve still catches the rare exp/divide rounding boundary.
    # Mode 3 never reads the error surface, so reuse this already-resident
    # pointer instead of allocating another dense matrix.
    error_bounds = shifted_utility
    if service is None:
        choice_kernel(
            ((len(choosers) + 127) // 128,), (128,),
            (shifted_utility, error_bounds, device_random, choices, probabilities,
             guard, bad, risk, np.int32(len(choosers)), np.int32(len(alternatives)),
             np.int32(sample_size), np.int32(3)),
        )
    else:
        weight_kernel, phase46_choice, duplicate_kernel = _compile_phase46_choice(
            cp, len(alternatives)
        )
        weights = workspace["weights"]
        weight_kernel(
            ((cells + 255) // 256,),
            (256,),
            (
                shifted_utility,
                row_maxima,
                weights,
                np.int64(cells),
                np.int32(len(alternatives)),
            ),
        )
        phase46_choice(
            ((len(choosers) + 127) // 128,),
            (128,),
            (
                weights,
                device_random,
                choices,
                probabilities,
                guard,
                bad,
                np.int32(len(choosers)),
                np.int32(len(alternatives)),
                np.int32(sample_size),
            ),
        )
    first = (
        workspace["first"]
        if workspace is not None
        else cp.empty_like(choices, dtype=cp.uint8)
    )
    counts = (
        workspace["counts"]
        if workspace is not None
        else cp.empty_like(choices, dtype=cp.uint32)
    )
    duplicate_kernel(
        ((len(choosers) + 127) // 128,), (128,),
        (choices, first, counts, np.int32(len(choosers)), np.int32(sample_size)),
    )
    cp.cuda.Stream.null.synchronize()
    device_choice_complete = time.perf_counter()
    if int(cp.count_nonzero(bad).get()):
        raise Phase45Unsupported("resident choice produced zero or invalid probabilities")
    host_choices = cp.asnumpy(choices)
    host_choices = alternative_ids[host_choices]
    host_probabilities = cp.asnumpy(probabilities)
    host_first = cp.asnumpy(first)
    host_counts = cp.asnumpy(counts)
    transfer_complete = time.perf_counter()
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
        diagnostic_summary = {
            "trace_label": str(trace_label),
            "choice_mismatches": int(np.count_nonzero(~same_choice)),
            "unguarded_mismatch_rows": int(np.count_nonzero(
                np.any(~same_choice, axis=1)
                & ~cp.asnumpy(guard).astype(bool, copy=False)
            )),
            "max_exact_boundary_margin": mismatch_boundary_max,
            "max_selected_log_error": float(np.max(probability_log_error)),
        }
        print("PHASE46_DIAGNOSTIC " + repr(diagnostic_summary), flush=True)
        if os.environ.get("CHOICEFORGE_PHASE45_DIAGNOSTIC_STOP", "0") == "1":
            raise RuntimeError(
                "Phase45 utility/probability diagnostic completed: "
                + repr(diagnostic_summary)
            )
    guard_host = cp.asnumpy(guard).astype(bool, copy=False)
    guard_count = int(np.count_nonzero(guard_host))
    if guard_count:
        from activitysim.core import logit
        from activitysim.core.choosing import sample_choices_maker_preserve_ordering

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
        exact_choices, exact_probabilities = sample_choices_maker_preserve_ordering(
            exact_probs, random_draws[guard_host], alternative_ids
        )
        # ActivitySim's public helper stores sample-major arrays; the resident
        # contract is chooser-major.  Transposition changes layout, not values.
        exact_choices = exact_choices.T
        exact_probabilities = exact_probabilities.T
        host_choices[guard_host] = exact_choices
        host_probabilities[guard_host] = exact_probabilities
        exact_first, exact_counts = _host_duplicate_contract(exact_choices)
        host_first[guard_host] = exact_first
        host_counts[guard_host] = exact_counts
    guard_complete = time.perf_counter()
    diagnostic_id = os.environ.get("CHOICEFORGE_PHASE46_SAMPLE_DIAGNOSTIC_ID")
    if diagnostic_id is not None:
        target = int(diagnostic_id)
        target_positions = np.flatnonzero(np.asarray(choosers.index) == target)
        if len(target_positions):
            target_position = int(target_positions[0])
            print(
                "PHASE46_SAMPLE_DIAGNOSTIC "
                + repr(
                    {
                        "trace_label": str(trace_label),
                        "id": target,
                        "guarded": bool(guard_host[target_position]),
                        "choices": host_choices[target_position].tolist(),
                        "probabilities": host_probabilities[target_position].tolist(),
                        "draws": random_draws[target_position].tolist(),
                        "first": host_first[target_position].tolist(),
                        "counts": host_counts[target_position].tolist(),
                    }
                ),
                flush=True,
            )
    sample = (
        _pack_sample(
            choosers, host_choices, host_probabilities, random_draws,
            host_first, host_counts, alt_col_name,
        )
        if service is None
        else _pack_sample_phase46(
            choosers, host_choices, host_probabilities, random_draws,
            host_first, host_counts, alt_col_name,
        )
    )
    finished = time.perf_counter()
    _TELEMETRY.append({
        "component": str(component), "trace_label": str(trace_label),
        "chooser_rows": len(choosers), "alternatives": len(alternatives),
        "utility_cells": int(cells), "sampled_rows": len(sample),
        "random_draws": int(random_draws.size), "program_terms": len(expressions),
        "program_cache_hit": cache_hit, "utility_seconds": utility_complete - started,
        "device_choice_seconds": device_choice_complete - utility_complete,
        "transfer_seconds": transfer_complete - device_choice_complete,
        "exact_guard_seconds": guard_complete - transfer_complete,
        "pack_seconds": finished - guard_complete,
        "exact_guard_rows": guard_count,
        "total_seconds": finished - started, "fallback": False,
        "runtime": "phase46_persistent" if service is not None else "phase45",
    })
    return sample
