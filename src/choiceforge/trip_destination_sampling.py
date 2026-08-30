"""Phase 39 CUDA utility factory for public MTC trip-destination sampling.

The production boundary deliberately leaves ActivitySim's probability
normalization, keyed random-number ledger, inverse-CDF selection, duplicate
collapse, and retry orchestration on the CPU.  CUDA generates the complete
133-million-cell utility matrix without a dense host Cartesian table.  A
conservative arithmetic envelope identifies the small set of chooser rows
whose inverse-CDF decisions could be changed by legal float32 reduction
differences; only those rows are adjudicated with the live Sharrow evaluator.
Unsupported specifications fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import time

import numba as nb
import numpy as np
import pandas as pd

from .cuda_backend import _cupy
from .cuda_skims import cuda_cube_from_activitysim


_KERNEL = None
_TELEMETRY = []
_EXPECTED_EXPRESSIONS = (
    "_od_DIST@od_skims['DIST']",
    "_dp_DIST@dp_skims['DIST']",
    "@(df.tour_mode_is_walk) & (od_skims['DISTWALK'] > max_walk_distance)",
    "@(df.tour_mode_is_walk) & (dp_skims['DISTWALK'] > max_walk_distance)",
    "@(df.tour_mode_is_bike) & (od_skims['DISTBIKE'] > max_bike_distance)",
    "@(df.tour_mode_is_bike) & (dp_skims['DISTBIKE'] > max_bike_distance)",
    "@np.log1p(size_terms.get(df.dest_taz, df.purpose)) # sharrow: np.log1p(size_terms['sizearray'])",
    "@size_terms.get(df.dest_taz, df.purpose) == 0 # sharrow: size_terms['sizearray'] == 0",
    "@(~df.is_joint & ~df.outbound) * (_od_DIST + _dp_DIST)",
    "@(~df.is_joint & df.outbound) * (_od_DIST + _dp_DIST)",
    "@df.is_joint * (_od_DIST + _dp_DIST)",
    "@df.outbound * _od_DIST",
    "@~df.outbound * _od_DIST",
    "@df.outbound * _dp_DIST",
    "@~df.outbound * _od_DIST",
)


class Phase39Unsupported(ValueError):
    """The live model does not satisfy the qualified Phase 39 contract."""


@dataclass(frozen=True)
class Phase39SamplingTelemetry:
    trace_label: str
    purpose: str
    backend: str
    chooser_rows: int
    alternatives: int
    utility_cells: int
    sampled_rows: int
    random_draws: int
    host_cross_join_rows_avoided: int
    utility_host_bytes: int
    arithmetic_guard_rows: int
    arithmetic_guard_cells: int
    arithmetic_guard_seconds: float
    utility_error_bound_max: float
    host_prepare_seconds: float
    upload_seconds: float
    kernel_seconds: float
    download_seconds: float
    probability_and_choice_seconds: float
    total_seconds: float
    fallback_calls: int
    specification_sha256: str
    contract_valid: bool
    shadow_utility_mismatches: int = 0
    shadow_utility_max_abs_difference: float = 0.0
    shadow_bound_violations: int = 0
    shadow_utility_cells_compared: int = 0


def reset_phase39_sampling_telemetry():
    _TELEMETRY.clear()


def phase39_sampling_telemetry():
    return [asdict(item) for item in _TELEMETRY]


def _specification_contract(spec):
    expressions = tuple(str(item).split("#", 1)[0].strip() for item in spec.index)
    expected = tuple(item.split("#", 1)[0].strip() for item in _EXPECTED_EXPRESSIONS)
    if expressions != expected:
        raise Phase39Unsupported("Phase 39 requires the reviewed 15-row public sample specification")
    if spec.shape != (15, 1):
        raise Phase39Unsupported("Phase 39 requires one coefficient column and 15 expressions")
    coefficients = np.asarray(spec.iloc[:, 0], dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise Phase39Unsupported("Phase 39 sample coefficients must be finite")
    document = "\n".join(
        f"{expression}\t{float(np.float32(coefficient)).hex()}"
        for expression, coefficient in zip(expressions, coefficients)
    )
    # Sharrow treats the first two ``name@expression`` rows as reusable named
    # intermediates.  Their CSV values are not dot-product coefficients.
    utility_coefficients = coefficients.copy()
    utility_coefficients[:2] = 0.0
    return utility_coefficients, hashlib.sha256(document.encode("utf-8")).hexdigest()


def _checked_zone(values, label, alternatives):
    values = np.asarray(values, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= alternatives):
        raise Phase39Unsupported(f"Phase 39 {label} is outside the dense skim universe")
    return values.astype(np.int32, copy=False)


def _compile_kernel(cp):
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    source = r'''
extern "C" __global__ void phase39_trip_sample_utilities(
    const int* origins,
    const int* primary_destinations,
    const int* purpose_indices,
    const unsigned char* walk_tours,
    const unsigned char* bike_tours,
    const unsigned char* joint_tours,
    const unsigned char* outbound_trips,
    const double* size_terms,
    const float* coefficients,
    const float* distance,
    const float* walk_distance,
    const float* bike_distance,
    float* utilities,
    float* utility_error_bounds,
    long long chooser_rows,
    int alternative_count,
    int purpose_count,
    float max_walk_distance,
    float max_bike_distance)
{
    const long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long cells = chooser_rows * (long long)alternative_count;
    if (cell >= cells) return;
    const long long row = cell / alternative_count;
    const int alternative = (int)(cell - row * alternative_count);
    const int origin = origins[row];
    const int primary = primary_destinations[row];
    const float od = distance[(long long)origin * alternative_count + alternative];
    const float dp = distance[(long long)alternative * alternative_count + primary];
    const int purpose_index = purpose_indices[row];
    if (purpose_index < 0 || purpose_index >= purpose_count) return;
    const double size = size_terms[(long long)alternative * purpose_count + purpose_index];
    const bool walk = walk_tours[row] != 0;
    const bool bike = bike_tours[row] != 0;
    const bool joint = joint_tours[row] != 0;
    const bool outbound = outbound_trips[row] != 0;
    const float intermediate[15] = {
        od,
        dp,
        (float)(walk && (walk_distance[(long long)origin * alternative_count + alternative] > max_walk_distance)),
        (float)(walk && (walk_distance[(long long)alternative * alternative_count + primary] > max_walk_distance)),
        (float)(bike && (bike_distance[(long long)origin * alternative_count + alternative] > max_bike_distance)),
        (float)(bike && (bike_distance[(long long)alternative * alternative_count + primary] > max_bike_distance)),
        (float)log1p(size),
        (float)(size == 0.0),
        (float)(((!joint) && (!outbound)) * (od + dp)),
        (float)(((!joint) && outbound) * (od + dp)),
        (float)(joint * (od + dp)),
        (float)(outbound * od),
        (float)((!outbound) * od),
        (float)(outbound * dp),
        (float)((!outbound) * od)
    };
    // Products are rounded exactly as in Sharrow's float32 intermediate
    // vector.  A double accumulator gives the closest reviewed GPU result;
    // the arithmetic envelope and guarded Numba adjudicator handle the few
    // legal CPU-dot reduction differences that remain.
    double utility = 0.0;
    double absolute_product_sum = 0.0;
    #pragma unroll 1
    for (int term = 0; term < 15; ++term) {
        const float product = coefficients[term] * intermediate[term];
        utility += (double)product;
        absolute_product_sum += (double)fabsf(product);
    }
    utilities[cell] = (float)utility;
    // The reviewed public spec has at most seven nonzero products in any cell.
    // This condition-aware envelope covers its float32 reduction error and a
    // small elementary-function/cast reserve.  Qualification exhaustively
    // checks the bound against live Sharrow for every benchmark utility cell.
    utility_error_bounds[cell] = (float)(2.0e-6 * absolute_product_sum + 2.0e-6);
}
'''
    _KERNEL = cp.RawKernel(
        source,
        "phase39_trip_sample_utilities",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    _KERNEL.compile()
    return _KERNEL


@nb.jit(
    cache=True,
    parallel=False,
    error_model="numpy",
    boundscheck=False,
    nopython=True,
    fastmath=False,
    nogil=True,
)
def _phase39_sharrow_arithmetic(
    origins,
    primary_destinations,
    purpose_indices,
    walk_tours,
    bike_tours,
    joint_tours,
    outbound_trips,
    size_terms,
    coefficients,
    distance,
    walk_distance,
    bike_distance,
    max_walk_distance,
    max_bike_distance,
):
    """Reproduce the reviewed Sharrow ``idotter`` for guarded rows.

    The generated public-model flow uses a 15-element float32 intermediate
    vector and ``np.dot(intermediate, coefficients, out=...)`` with Numba
    ``fastmath=False``.  Keeping that exact shape and call form makes this a
    small, cached arithmetic adjudicator instead of invoking Sharrow's full
    flow-construction and data-tree machinery for every guarded segment.
    """
    row_count = len(origins)
    alternative_count = distance.shape[1]
    result = np.empty((row_count, alternative_count, 1), dtype=np.float32)
    for row in range(row_count):
        intermediate = np.empty(15, dtype=np.float32)
        for alternative in range(alternative_count):
            od = distance[origins[row], alternative]
            dp = distance[alternative, primary_destinations[row]]
            intermediate[0] = od
            intermediate[1] = dp
            intermediate[2] = walk_tours[row] & (
                walk_distance[origins[row], alternative] > max_walk_distance
            )
            intermediate[3] = walk_tours[row] & (
                walk_distance[alternative, primary_destinations[row]]
                > max_walk_distance
            )
            intermediate[4] = bike_tours[row] & (
                bike_distance[origins[row], alternative] > max_bike_distance
            )
            intermediate[5] = bike_tours[row] & (
                bike_distance[alternative, primary_destinations[row]]
                > max_bike_distance
            )
            size = size_terms[alternative, purpose_indices[row]]
            intermediate[6] = np.log1p(size)
            intermediate[7] = size == 0.0
            intermediate[8] = (
                (~joint_tours[row] & ~outbound_trips[row]) * (od + dp)
            )
            intermediate[9] = (
                (~joint_tours[row] & outbound_trips[row]) * (od + dp)
            )
            intermediate[10] = joint_tours[row] * (od + dp)
            intermediate[11] = outbound_trips[row] * od
            intermediate[12] = ~outbound_trips[row] * od
            intermediate[13] = outbound_trips[row] * dp
            intermediate[14] = ~outbound_trips[row] * od
            np.dot(intermediate, coefficients, out=result[row, alternative, :])
    return result[:, :, 0]


def _guard_reference_utilities(
    trips, alternatives, size_term_matrix, skim_hotel, spec, constants
):
    """Evaluate only precision-ambiguous rows with Sharrow's numeric ABI."""
    coefficients, _ = _specification_contract(spec)
    coefficient_column = np.ascontiguousarray(coefficients.reshape(15, 1))
    alternative_count = len(alternatives)
    origins = _checked_zone(trips["origin"], "origin", alternative_count)
    primary = _checked_zone(
        trips["tour_leg_dest"], "primary destination", alternative_count
    )
    purpose_indices = np.asarray(trips["purpose_index_num"], dtype=np.int32)
    flags = [
        np.asarray(trips[column], dtype=np.bool_)
        for column in (
            "tour_mode_is_walk",
            "tour_mode_is_bike",
            "is_joint",
            "outbound",
        )
    ]
    sizes = np.asarray(size_term_matrix.df, dtype=np.float64, order="C")
    skims = skim_hotel.sample_skims(presample=False)

    def host_cube(key):
        wrapper = skims["od_skims"]
        array = wrapper.dataset[key]
        values = np.asarray(array.transpose(wrapper.odim, wrapper.ddim).values)
        if values.dtype != np.float32 or values.shape != (
            alternative_count,
            alternative_count,
        ):
            raise Phase39Unsupported(
                f"Phase 39 guarded skim {key!r} has an unsupported layout"
            )
        return np.ascontiguousarray(values)

    return _phase39_sharrow_arithmetic(
        origins,
        primary,
        purpose_indices,
        *flags,
        sizes,
        coefficient_column,
        host_cube("DIST"),
        host_cube("DISTWALK"),
        host_cube("DISTBIKE"),
        np.float32(constants["max_walk_distance"]),
        np.float32(constants["max_bike_distance"]),
    )


def _gpu_utilities(
    trips,
    alternatives,
    size_term_matrix,
    skim_hotel,
    spec,
    constants,
    *,
    device_only=False,
):
    cp = _cupy()
    coefficients, fingerprint = _specification_contract(spec)
    alternative_ids = np.asarray(alternatives.index, dtype=np.int64)
    alternative_count = len(alternative_ids)
    if alternative_count == 0 or not np.array_equal(
        alternative_ids, np.arange(alternative_count, dtype=np.int64)
    ):
        raise Phase39Unsupported("Phase 39 requires the canonical dense zero-based zone universe")
    required = (
        "origin", "tour_leg_dest", "tour_mode_is_walk", "tour_mode_is_bike",
        "is_joint", "outbound", "purpose_index_num",
    )
    missing = [column for column in required if column not in trips]
    if missing:
        raise Phase39Unsupported(f"Phase 39 chooser columns are missing: {missing}")
    origins = _checked_zone(trips["origin"], "origin", alternative_count)
    primary = _checked_zone(trips["tour_leg_dest"], "primary destination", alternative_count)
    purpose_indices = np.asarray(trips["purpose_index_num"], dtype=np.int64)
    purpose_count = int(size_term_matrix.df.shape[1])
    if purpose_indices.size and (
        purpose_indices.min() < 0 or purpose_indices.max() >= purpose_count
    ):
        raise Phase39Unsupported("Phase 39 purpose index is outside the size-term matrix")
    purpose_indices = purpose_indices.astype(np.int32, copy=False)
    flags = [
        np.asarray(trips[column], dtype=np.uint8)
        for column in ("tour_mode_is_walk", "tour_mode_is_bike", "is_joint", "outbound")
    ]
    sizes = np.asarray(
        size_term_matrix.df.reindex(alternative_ids), dtype=np.float64, order="C"
    )
    if not np.isfinite(sizes).all() or (sizes < 0).any():
        raise Phase39Unsupported("Phase 39 size terms must be finite and nonnegative")
    skims = skim_hotel.sample_skims(presample=False)
    distance, destination_count, time_count, rank = cuda_cube_from_activitysim(
        skims["od_skims"], "DIST"
    )
    walk_distance, walk_count, walk_time_count, walk_rank = cuda_cube_from_activitysim(
        skims["od_skims"], "DISTWALK"
    )
    bike_distance, bike_count, bike_time_count, bike_rank = cuda_cube_from_activitysim(
        skims["od_skims"], "DISTBIKE"
    )
    if (
        destination_count != alternative_count
        or walk_count != alternative_count
        or bike_count != alternative_count
        or (rank, walk_rank, bike_rank) != (2, 2, 2)
        or (time_count, walk_time_count, bike_time_count) != (1, 1, 1)
    ):
        raise Phase39Unsupported("Phase 39 requires three aligned 2-D public skim cubes")
    prepared = time.perf_counter()
    device_inputs = [
        cp.asarray(origins), cp.asarray(primary), cp.asarray(purpose_indices),
        *(cp.asarray(item) for item in flags),
        cp.asarray(sizes), cp.asarray(coefficients),
    ]
    utilities = cp.empty((len(trips), alternative_count), dtype=cp.float32)
    utility_error_bounds = cp.empty_like(utilities)
    cp.cuda.Stream.null.synchronize()
    uploaded = time.perf_counter()
    kernel = _compile_kernel(cp)
    cells = int(len(trips) * alternative_count)
    block = 256
    kernel(
        ((cells + block - 1) // block,),
        (block,),
        (
            *device_inputs, distance, walk_distance, bike_distance, utilities,
            utility_error_bounds,
            np.int64(len(trips)), np.int32(alternative_count),
            np.int32(purpose_count),
            np.float32(constants["max_walk_distance"]),
            np.float32(constants["max_bike_distance"]),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    completed = time.perf_counter()
    if device_only:
        return utilities, utility_error_bounds, fingerprint, {
            "prepared": prepared,
            "uploaded": uploaded,
            "completed": completed,
            "downloaded": completed,
            "utility_host_bytes": 0,
            "first_debug": None,
        }
    host = cp.asnumpy(utilities)
    host_error_bounds = cp.asnumpy(utility_error_bounds)
    downloaded = time.perf_counter()
    first_debug = None
    if len(trips) and alternative_count:
        origin0 = int(origins[0])
        primary0 = int(primary[0])
        purpose0 = int(purpose_indices[0])
        first_debug = {
            "origin": origin0,
            "primary": primary0,
            "purpose_index": purpose0,
            "outbound": int(flags[3][0]),
            "joint": int(flags[2][0]),
            "size": float(sizes[0, purpose0]),
            "od": float(cp.asnumpy(distance[origin0, 0])),
            "dp": float(cp.asnumpy(distance[0, primary0])),
            "coefficients": [float(item) for item in coefficients],
        }
    return host, host_error_bounds, fingerprint, {
        "prepared": prepared,
        "uploaded": uploaded,
        "completed": completed,
        "downloaded": downloaded,
        "utility_host_bytes": int(host.nbytes),
        "first_debug": first_debug,
    }


def _activitysim_inverse_cdf(
    state,
    utilities,
    trips,
    alternatives,
    sample_size,
    alt_col_name,
    *,
    utility_error_bounds=None,
    exact_utility_resolver=None,
):
    """Apply ActivitySim's existing probability, RNG, and duplicate semantics."""
    from activitysim.core import logit
    from activitysim.core.choosing import sample_choices_maker_preserve_ordering

    utilities = pd.DataFrame(utilities, index=trips.index)
    probs = logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=True,
        trace_label="phase39_trip_destination_sample",
        trace_choosers=trips,
        overflow_protection=False,
    )
    nonzero = probs.sum(axis=1) != 0
    if not nonzero.any():
        return pd.DataFrame(
            columns=[alt_col_name, "rand", "prob", "pick_count"],
            index=pd.Index([], name=trips.index.name),
        ), 0, 0, 0.0
    if not nonzero.all():
        probs = probs[nonzero]
        if utility_error_bounds is not None:
            utility_error_bounds = np.asarray(utility_error_bounds)[nonzero.to_numpy()]
    rands = state.get_rn_generator().random_for_df(probs, n=sample_size)
    choices, choice_probs = sample_choices_maker_preserve_ordering(
        probs.to_numpy(copy=False), rands, np.asarray(alternatives.index)
    )
    guard_rows = 0
    guard_started = time.perf_counter()
    if exact_utility_resolver is not None and utility_error_bounds is not None:
        probability_values = probs.to_numpy(copy=False)
        bounds = np.asarray(utility_error_bounds, dtype=np.float64)
        low_weights = probability_values * np.exp(-bounds)
        high_weights = probability_values * np.exp(bounds)
        low_prefix = np.cumsum(low_weights, axis=1, dtype=np.float64)
        high_prefix = np.cumsum(high_weights, axis=1, dtype=np.float64)
        low_total = low_prefix[:, -1]
        high_total = high_prefix[:, -1]
        # The qualified alternative universe is dense and zero-based, so the
        # sampled alternative identifier is also its CDF position.
        positions = choices.T.astype(np.int64, copy=False)
        rows = np.arange(len(probs), dtype=np.int64)[:, None]
        prefix_low = low_prefix[rows, positions]
        suffix_high = high_total[:, None] - high_prefix[rows, positions]
        cdf_low = prefix_low / (prefix_low + suffix_high)
        previous = np.maximum(positions - 1, 0)
        previous_high = high_prefix[rows, previous]
        previous_suffix_low = low_total[:, None] - low_prefix[rows, previous]
        previous_cdf_high = previous_high / (previous_high + previous_suffix_low)
        previous_cdf_high = np.where(positions == 0, 0.0, previous_cdf_high)
        guaranteed = (cdf_low > rands) & (previous_cdf_high <= rands)
        guard_mask = ~guaranteed.all(axis=1)
        guard_rows = int(np.count_nonzero(guard_mask))
        if guard_rows:
            exact_utilities = exact_utility_resolver(probs.index[guard_mask])
            exact_probs = logit.utils_to_probs(
                state,
                pd.DataFrame(exact_utilities, index=probs.index[guard_mask]),
                allow_zero_probs=True,
                trace_label="phase39_trip_destination_precision_guard",
                trace_choosers=trips.loc[probs.index[guard_mask]],
                overflow_protection=False,
            )
            probs.iloc[np.flatnonzero(guard_mask), :] = exact_probs.to_numpy(copy=False)
            choices, choice_probs = sample_choices_maker_preserve_ordering(
                probs.to_numpy(copy=False), rands, np.asarray(alternatives.index)
            )
    guard_seconds = time.perf_counter() - guard_started
    result = pd.DataFrame(
        {
            alt_col_name: choices.flatten(order="F"),
            "rand": rands.T.flatten(order="F"),
            "prob": choice_probs.flatten(order="F"),
            trips.index.name: np.repeat(np.asarray(probs.index), sample_size),
        }
    )
    groups = result.groupby([trips.index.name, alt_col_name])
    result["pick_count"] = groups.cumcount(ascending=True)
    result["pick_dup"] = result["pick_count"] > 0
    result["pick_count"] += groups.cumcount(ascending=False) + 1
    result = result[~result.pop("pick_dup")].copy()
    result.set_index(trips.index.name, inplace=True)
    result["prob"] = result["prob"].astype(np.float32)
    result["pick_count"] = result["pick_count"].astype(np.uint32)
    result = result.sort_values(by=alt_col_name).sort_index(kind="mergesort")
    return result, int(rands.size), guard_rows, guard_seconds


def _sharrow_reference_utilities(
    state, trips, alternatives, model_settings, size_term_matrix, skim_hotel, spec, trace_label
):
    """Evaluate the live public Sharrow program without drawing random numbers."""
    from activitysim.core import expressions, interaction_simulate

    skims = skim_hotel.sample_skims(presample=False)
    alternatives = alternatives.copy()
    if alternatives.index.name not in alternatives:
        alternatives[alternatives.index.name] = alternatives.index
    locals_dict = state.get_global_constants().copy()
    locals_dict.update(model_settings.CONSTANTS)
    locals_dict.update(
        {
            "size_terms": size_term_matrix,
            "size_terms_array": size_term_matrix.df.to_numpy(),
            "timeframe": "trip",
            "land_use": state.get_dataframe("land_use"),
        }
    )
    locals_dict.update(skims)
    expressions.annotate_preprocessors(
        state,
        df=alternatives,
        locals_dict=locals_dict,
        skims=skims,
        model_settings=model_settings,
        trace_label=trace_label,
        preprocessor_setting_name="alts_preprocessor_sample",
    )
    reference, _ = interaction_simulate.eval_interaction_utilities(
        state,
        spec,
        trips,
        locals_dict,
        trace_label,
        None,
        estimator=None,
        log_alt_losers=state.settings.log_alt_losers,
        extra_data=alternatives,
        zone_layer=None,
        compute_settings=model_settings.compute_settings.subcomponent_settings("sample"),
    )
    return np.asarray(reference).reshape(len(trips), len(alternatives))


def sample_trip_destinations_cuda(
    state,
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skim_hotel,
    estimator,
    trace_label,
):
    """Return the public trip-destination sample with GPU utility generation."""
    from activitysim.core import estimation, simulate
    from activitysim.core.interaction_sample import resolve_sample_method

    started = time.perf_counter()
    if estimator is not None or estimation.manager.enabled:
        raise Phase39Unsupported("Phase 39 does not support estimation mode")
    if resolve_sample_method(
        state, model_settings.compute_settings.subcomponent_settings("sample")
    ) != "inverse_cdf":
        raise Phase39Unsupported("Phase 39 requires inverse-CDF destination sampling")
    sample_size = int(model_settings.SAMPLE_SIZE)
    if state.settings.disable_destination_sampling or sample_size <= 0:
        raise Phase39Unsupported("Phase 39 requires enabled positive-size sampling")
    spec = simulate.spec_for_segment(
        state,
        None,
        spec_id="SAMPLE_SPEC",
        segment_name=primary_purpose,
        estimator=None,
        spec_file_name=model_settings.SAMPLE_SPEC,
        coefficients_file_name=model_settings.COEFFICIENTS,
    )
    constants = dict(model_settings.CONSTANTS)
    if set(constants) < {"max_walk_distance", "max_bike_distance"}:
        raise Phase39Unsupported("Phase 39 distance limits are absent")
    utilities, utility_error_bounds, fingerprint, clocks = _gpu_utilities(
        trips, alternatives, size_term_matrix, skim_hotel, spec, constants
    )
    shadow_mismatches = 0
    shadow_max_abs = 0.0
    shadow_bound_violations = 0
    shadow_cells_compared = 0
    shadow_setting = os.environ.get("CHOICEFORGE_PHASE39_UTILITY_SHADOW", "0")
    if shadow_setting in {"1", "exact"}:
        reference = _sharrow_reference_utilities(
            state,
            trips,
            alternatives,
            model_settings,
            size_term_matrix,
            skim_hotel,
            spec,
            trace_label,
        )
        shadow_cells_compared = int(reference.size)
        equal = np.equal(utilities, reference)
        shadow_mismatches = int(np.count_nonzero(~equal))
        if shadow_mismatches:
            finite = np.isfinite(utilities) & np.isfinite(reference)
            differences = np.abs(utilities - reference)
            shadow_bound_violations = int(
                np.count_nonzero(~finite | (differences > utility_error_bounds))
            )
            shadow_max_abs = float(
                np.max(np.abs(utilities[finite] - reference[finite]))
                if finite.any()
                else np.inf
            )
            shadow_bound = 0.0 if shadow_setting == "exact" else 0.00025
            if (
                not np.isfinite(shadow_max_abs)
                or shadow_max_abs > shadow_bound
                or shadow_bound_violations
            ):
                row, column = np.argwhere(~equal)[0]
                row_data = trips.iloc[int(row)]
                origin = int(row_data["origin"])
                primary = int(row_data["tour_leg_dest"])
                purpose_index = int(row_data["purpose_index_num"])
                skims = skim_hotel.sample_skims(presample=False)
                od_distance = np.asarray(skims["od_skims"].dataset["DIST"].values)
                walk_distance = np.asarray(skims["od_skims"].dataset["DISTWALK"].values)
                bike_distance = np.asarray(skims["od_skims"].dataset["DISTBIKE"].values)
                od = np.float32(od_distance[origin, column])
                dp = np.float32(od_distance[column, primary])
                size = float(size_term_matrix.df.iloc[column, purpose_index])
                walk = bool(row_data["tour_mode_is_walk"])
                bike = bool(row_data["tour_mode_is_bike"])
                joint = bool(row_data["is_joint"])
                outbound = bool(row_data["outbound"])
                features = np.asarray(
                    [
                        od, dp,
                        walk and walk_distance[origin, column] > constants["max_walk_distance"],
                        walk and walk_distance[column, primary] > constants["max_walk_distance"],
                        bike and bike_distance[origin, column] > constants["max_bike_distance"],
                        bike and bike_distance[column, primary] > constants["max_bike_distance"],
                        np.log1p(size), size == 0,
                        (not joint and not outbound) * (od + dp),
                        (not joint and outbound) * (od + dp),
                        joint * (od + dp), outbound * od, (not outbound) * od,
                        outbound * dp, (not outbound) * od,
                    ],
                    dtype=np.float32,
                )
                live_coefficients, _ = _specification_contract(spec)
                products = np.asarray(features * live_coefficients, dtype=np.float32)
                raise AssertionError(
                    "Phase 39 GPU/Sharrow utility shadow exceeded its reviewed "
                    "float32 reduction bound: "
                    f"count={shadow_mismatches} max_abs={shadow_max_abs:.12g} "
                    f"bound_violations={shadow_bound_violations} "
                    f"row={row} alternative={column} gpu={utilities[row, column]:.17g} "
                    f"sharrow={reference[row, column]:.17g} "
                    f"gpu_dtype={utilities.dtype} sharrow_dtype={reference.dtype}"
                    f" inputs={clocks['first_debug']} nonzero_products="
                    f"{[(i, float(v)) for i, v in enumerate(products) if v != 0]} "
                    f"numpy_dot={float(np.dot(features, live_coefficients)):.17g} "
                    f"double_sum={float(products.astype(np.float64).sum(dtype=np.float64).astype(np.float32)):.17g}"
                )
    probability_started = time.perf_counter()

    def exact_utility_resolver(index):
        guarded_trips = trips.loc[index]
        # The live evaluator is the authoritative arithmetic ABI and is faster
        # than the dedicated Numba reconstruction on this benchmark.  The
        # reconstruction remains available as an independently implemented
        # shadow oracle, never as an unverified production substitute.
        exact = _sharrow_reference_utilities(
            state,
            guarded_trips,
            alternatives,
            model_settings,
            size_term_matrix,
            skim_hotel,
            spec,
            trace_label,
        )
        if os.environ.get("CHOICEFORGE_PHASE39_GUARD_SHADOW", "0") == "1":
            dedicated = _guard_reference_utilities(
                guarded_trips,
                alternatives,
                size_term_matrix,
                skim_hotel,
                spec,
                constants,
            )
            if not np.array_equal(exact, dedicated):
                differences = np.abs(exact - dedicated)
                raise AssertionError(
                    "Phase 39 dedicated guard/Sharrow shadow mismatch: "
                    f"cells={int(np.count_nonzero(exact != dedicated))} "
                    f"max_abs={float(np.max(differences)):.12g}"
                )
        return exact

    sample, random_draws, guard_rows, guard_seconds = _activitysim_inverse_cdf(
        state,
        utilities,
        trips,
        alternatives,
        min(sample_size, len(alternatives)),
        model_settings.ALT_DEST_COL_NAME,
        utility_error_bounds=utility_error_bounds,
        exact_utility_resolver=exact_utility_resolver,
    )
    finished = time.perf_counter()
    _TELEMETRY.append(
        Phase39SamplingTelemetry(
            trace_label=str(trace_label),
            purpose=str(primary_purpose),
            backend="phase39_cuda_utility_activitysim_probability_rng",
            chooser_rows=int(len(trips)),
            alternatives=int(len(alternatives)),
            utility_cells=int(len(trips) * len(alternatives)),
            sampled_rows=int(len(sample)),
            random_draws=random_draws,
            host_cross_join_rows_avoided=int(len(trips) * len(alternatives)),
            utility_host_bytes=clocks["utility_host_bytes"],
            arithmetic_guard_rows=guard_rows,
            arithmetic_guard_cells=int(guard_rows * len(alternatives)),
            arithmetic_guard_seconds=guard_seconds,
            utility_error_bound_max=float(np.max(utility_error_bounds, initial=0.0)),
            host_prepare_seconds=clocks["prepared"] - started,
            upload_seconds=clocks["uploaded"] - clocks["prepared"],
            kernel_seconds=clocks["completed"] - clocks["uploaded"],
            download_seconds=clocks["downloaded"] - clocks["completed"],
            probability_and_choice_seconds=finished - probability_started,
            total_seconds=finished - started,
            fallback_calls=0,
            specification_sha256=fingerprint,
            contract_valid=True,
            shadow_utility_mismatches=shadow_mismatches,
            shadow_utility_max_abs_difference=shadow_max_abs,
            shadow_bound_violations=shadow_bound_violations,
            shadow_utility_cells_compared=shadow_cells_compared,
        )
    )
    return sample
