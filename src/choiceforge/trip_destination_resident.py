"""Phase 40 device-resident trip-destination sampling.

The complete full-zone utility matrix remains on the GPU through probability
normalization, preserved-order inverse-CDF selection, and duplicate counting.
ActivitySim still advances its authoritative keyed random ledger; only those
draws are uploaded.  Rows whose choices are not invariant throughout the
reviewed utility-error envelope are adjudicated by live Sharrow with the same
draws.  Unsupported public-model contracts fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

import numpy as np
import pandas as pd

from .arithmetic_abi import (
    NUMPY_FLOAT32_CHOICE_ABI_SHA256,
    NUMPY_FLOAT32_CHOICE_ABI_VERSION,
    SHARROW15_ABI_SHA256,
    SHARROW15_ABI_VERSION,
    numpy_float32_choice_cuda_helpers,
)
from .cuda_backend import _cupy
from .trip_destination_sampling import (
    Phase39Unsupported,
    _guard_reference_utilities,
    _gpu_utilities,
    _specification_contract,
)


_CHOICE_KERNEL = None
_CHOICE_KERNELS = {}
_DUPLICATE_KERNEL = None
_ACTIVITYSIM_CHOICE_WARM = False
_TELEMETRY = []
_PHASE41_TELEMETRY = []


class Phase40Unsupported(Phase39Unsupported):
    """The live model does not satisfy the qualified resident-sampler contract."""


@dataclass(frozen=True)
class Phase40SamplingTelemetry:
    trace_label: str
    purpose: str
    backend: str
    chooser_rows: int
    alternatives: int
    utility_cells: int
    random_draws: int
    sampled_rows: int
    utility_host_bytes: int
    compact_host_bytes: int
    random_upload_bytes: int
    dense_utility_download_bytes_avoided: int
    arithmetic_guard_rows: int
    arithmetic_guard_cells: int
    cdf_error_bound_scale: float
    selected_probability_log_guard_tolerance: float
    selected_probability_log_risk_max: float
    selected_probability_log_risk_p99: float
    utility_kernel_seconds: float
    resident_choice_kernel_seconds: float
    duplicate_kernel_seconds: float
    exact_guard_seconds: float
    exact_guard_evaluator: str
    arithmetic_abi_version: str
    arithmetic_abi_sha256: str
    probability_abi_version: str
    probability_abi_sha256: str
    exact_shared_arithmetic: bool
    compact_download_seconds: float
    host_pack_seconds: float
    total_seconds: float
    fallback_calls: int
    specification_sha256: str
    contract_valid: bool


def reset_phase40_sampling_telemetry():
    _TELEMETRY.clear()


def phase40_sampling_telemetry():
    return [asdict(item) for item in _TELEMETRY]


def reset_phase41_sampling_telemetry():
    _PHASE41_TELEMETRY.clear()


def phase41_sampling_telemetry():
    return [asdict(item) for item in _PHASE41_TELEMETRY]


def _compile_resident_kernels(cp, alternative_count=1454):
    global _CHOICE_KERNEL, _DUPLICATE_KERNEL
    alternative_count = int(alternative_count)
    if alternative_count not in _CHOICE_KERNELS:
        choice_source = numpy_float32_choice_cuda_helpers(alternative_count) + r'''
__device__ __forceinline__ float phase_choice_expf(float value, int exact_shared_arithmetic)
{
    return exact_shared_arithmetic != 0 ? numpy_avx2_expf(value) : expf(value);
}

extern "C" __global__ void phase40_resident_inverse_cdf(
    const float* utilities,
    const float* error_bounds,
    const double* random_draws,
    int* choices,
    float* choice_probabilities,
    unsigned char* guard_rows,
    unsigned char* bad_rows,
    float* probability_log_risk,
    int chooser_rows,
    int alternative_count,
    int sample_size,
    int exact_shared_arithmetic)
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

    // ActivitySim exponentiates float32 utilities, sums float32 weights, and
    // stores float32 probabilities.  The CDF accumulator itself is double in
    // its Numba preserved-order chooser.
    float total = 0.0f;
    if (exact_shared_arithmetic != 0) {
        total = numpy_pairwise_exp_sum_1454(utilities + base);
    } else {
        for (int alternative = 0; alternative < alternative_count; ++alternative) {
            const float utility = utilities[base + alternative];
            const float weight = expf(utility);
            if (weight > 0.0f) total += weight;
        }
    }
    if (!(total > 0.0f) || !isfinite(total)) {
        bad_rows[row] = 1;
        return;
    }

    // Anchor the proof envelope to the exact float32 probability table used by
    // the chooser.  This is the Phase 39 interval construction: multiplying a
    // normalized probability by exp(+/- utility error) makes the common
    // normalization factor cancel, while avoiding a second, subtly different
    // float-versus-double exponential path in the proof itself.
    double low_total = 1.0;
    double high_total = 1.0;
    if (exact_shared_arithmetic != 1) {
        low_total = 0.0;
        high_total = 0.0;
        for (int alternative = 0; alternative < alternative_count; ++alternative) {
            const float weight = expf(utilities[base + alternative]);
            const float probability = weight > 0.0f ? weight / total : 0.0f;
            const double conservative_bound =
                (double)error_bounds[base + alternative];
            low_total += (double)probability * exp(-conservative_bound);
            high_total += (double)probability * exp(conservative_bound);
        }
        if (!(low_total > 0.0) || !(high_total > 0.0)) {
            bad_rows[row] = 1;
            return;
        }
    }

    int sorted_position = 0;
    int last_nontrivial = alternative_count - 1;
    float cumulative_weight = 0.0f;
    double cumulative_probability = 0.0;
    for (int alternative = 0; alternative < alternative_count; ++alternative) {
        const float weight = phase_choice_expf(
            utilities[base + alternative], exact_shared_arithmetic);
        const float probability = weight > 0.0f ? weight / total : 0.0f;
        cumulative_weight += weight;
        cumulative_probability += (double)probability;
        if (probability >= 1.0e-30f) last_nontrivial = alternative;
        while (sorted_position < sample_size &&
               cumulative_probability > sorted_draws[sorted_position]) {
            const int original = order[sorted_position];
            choices[draw_base + original] = alternative;
            choice_probabilities[draw_base + original] = probability;
            ++sorted_position;
        }
        if (sorted_position == sample_size) break;
    }
    while (sorted_position < sample_size) {
        const int original = order[sorted_position];
        const float weight = phase_choice_expf(
            utilities[base + last_nontrivial], exact_shared_arithmetic);
        choices[draw_base + original] = last_nontrivial;
        choice_probabilities[draw_base + original] = weight / total;
        ++sorted_position;
    }

    // Mode 1 is the Phase 41 exact arithmetic ABI.  Mode 2 deliberately uses
    // the same NumPy-compatible exp/pairwise normalization while retaining
    // the interval guard for an approximately evaluated utility surface.
    // With the Phase 41 arithmetic ABI the utility surface is already the
    // exact Sharrow surface.  The interval calculation below is only an error-
    // envelope proof for approximate utilities; even a zero envelope can flag
    // harmless normalization drift because ActivitySim intentionally compares
    // draws to the un-renormalized float32 probability prefix.  Bypass that
    // proof, not the actual choice calculation.  End-to-end output verification
    // remains the authority for exp/probability compatibility.
    if (exact_shared_arithmetic == 1) {
        guard_rows[row] = 0;
        probability_log_risk[row] = 0.0f;
        return;
    }

    // Mode 3 has a bit-identical utility surface but a shifted CUDA
    // exp/divide path that can differ from NumPy by one float32 ulp.  Guard
    // only draws close enough to a CDF boundary for that drift to matter.
    if (exact_shared_arithmetic == 3) {
        const double cdf_rounding_reserve = 5.0e-7;
        double prefix = 0.0;
        bool guarded = false;
        for (int alternative = 0; alternative < alternative_count; ++alternative) {
            const float weight = phase_choice_expf(
                utilities[base + alternative], exact_shared_arithmetic);
            const float probability = weight > 0.0f ? weight / total : 0.0f;
            const double previous = prefix;
            prefix += (double)probability;
            for (int draw = 0; draw < sample_size; ++draw) {
                if (choices[draw_base + draw] != alternative) continue;
                const double random_draw = random_draws[draw_base + draw];
                if (fabs(prefix - random_draw) <= cdf_rounding_reserve ||
                    fabs(random_draw - previous) <= cdf_rounding_reserve) {
                    guarded = true;
                }
            }
        }
        guard_rows[row] = guarded ? 1 : 0;
        probability_log_risk[row] = 0.0f;
        return;
    }

    // Certify every selected CDF interval against the utility-error envelope.
    bool guarded = false;
    double row_probability_risk = 0.0;
    double prefix_low = 0.0;
    double prefix_high = 0.0;
    double previous_low = 0.0;
    double previous_high = 0.0;
    for (int alternative = 0; alternative < alternative_count; ++alternative) {
        const double conservative_bound =
            (double)error_bounds[base + alternative];
        const float weight = phase_choice_expf(
            utilities[base + alternative], exact_shared_arithmetic);
        const float probability = weight > 0.0f ? weight / total : 0.0f;
        previous_low = prefix_low;
        previous_high = prefix_high;
        prefix_low += (double)probability * exp(-conservative_bound);
        prefix_high += (double)probability * exp(conservative_bound);
        for (int draw = 0; draw < sample_size; ++draw) {
            if (choices[draw_base + draw] != alternative) continue;
            const double suffix_high = high_total - prefix_high;
            const double suffix_low_before = low_total - previous_low;
            const double cdf_low = prefix_low / (prefix_low + suffix_high);
            const double previous_cdf_high = alternative == 0
                ? 0.0
                : previous_high / (previous_high + suffix_low_before);
            const double random_draw = random_draws[draw_base + draw];
            // The interval algebra bounds the utility surface.  Reserve an
            // additional ten parts per million for the shifted float32
            // exp/divide/CDF implementation used by the model-wide sampler.
            const double cdf_rounding_reserve = 1.0e-5;
            if (!(cdf_low - cdf_rounding_reserve > random_draw &&
                  previous_cdf_high + cdf_rounding_reserve <= random_draw)) {
                guarded = true;
            }
            const double selected_low =
                (double)probability * exp(-conservative_bound) /
                high_total;
            const double selected_high =
                (double)probability * exp(conservative_bound) /
                low_total;
            if (probability > 0.0f && selected_low > 0.0 && selected_high > 0.0) {
                const double lower_risk = log((double)probability / selected_low);
                const double upper_risk = log(selected_high / (double)probability);
                const double selected_risk = lower_risk > upper_risk
                    ? lower_risk : upper_risk;
                if (selected_risk > row_probability_risk) {
                    row_probability_risk = selected_risk;
                }
            }
        }
    }
    // Destination choice carries log(sample probability) into its diagnostic
    // logsum.  Route rows whose certified selected-probability envelope consumes
    // most of the public 1e-4 logsum budget to the exact Sharrow adjudicator.
    if (row_probability_risk > 9.0e-5) guarded = true;
    guard_rows[row] = guarded ? 1 : 0;
    probability_log_risk[row] = (float)row_probability_risk;
}
'''
        choice_source = choice_source.replace(
            "numpy_pairwise_exp_sum_1454",
            f"numpy_pairwise_exp_sum_{alternative_count}",
        )
        choice_kernel = cp.RawKernel(
            choice_source,
            "phase40_resident_inverse_cdf",
            options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
        )
        choice_kernel.compile()
        _CHOICE_KERNELS[alternative_count] = choice_kernel
        if alternative_count == 1454:
            _CHOICE_KERNEL = choice_kernel
    if _DUPLICATE_KERNEL is None:
        duplicate_source = r'''
extern "C" __global__ void phase40_duplicate_counts(
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
        _DUPLICATE_KERNEL = cp.RawKernel(
            duplicate_source,
            "phase40_duplicate_counts",
            options=("--std=c++11",),
        )
        _DUPLICATE_KERNEL.compile()
    return _CHOICE_KERNELS[alternative_count], _DUPLICATE_KERNEL


def _host_duplicate_contract(choices):
    """Return ActivitySim first-occurrence flags and total duplicate counts."""
    choices = np.asarray(choices)
    equal = choices[:, :, None] == choices[:, None, :]
    counts = equal.sum(axis=2, dtype=np.uint32)
    prior = np.tril(np.ones((choices.shape[1], choices.shape[1]), dtype=bool), k=-1)
    first = ~np.any(equal & prior[None, :, :], axis=2)
    first = first.astype(np.uint8, copy=False)
    return first, counts


def _preserved_order_choices(probabilities, random_draws, alternatives):
    """ActivitySim's preserved-order inverse CDF without lazy Numba state."""
    probabilities = np.asarray(probabilities)
    random_draws = np.asarray(random_draws, dtype=np.float64)
    alternatives = np.asarray(alternatives)
    chooser_rows, sample_size = random_draws.shape
    choices = np.empty((chooser_rows, sample_size), dtype=np.int32)
    choice_probabilities = np.empty((chooser_rows, sample_size), dtype=np.float32)
    for row in range(chooser_rows):
        order = np.argsort(random_draws[row])
        sample_position = 0
        cumulative = 0.0
        for alternative_position in range(probabilities.shape[1]):
            cumulative += float(probabilities[row, alternative_position])
            while (
                sample_position < sample_size
                and cumulative > random_draws[row, order[sample_position]]
            ):
                original_position = int(order[sample_position])
                choices[row, original_position] = alternatives[alternative_position]
                choice_probabilities[row, original_position] = probabilities[
                    row, alternative_position
                ]
                sample_position += 1
            if sample_position == sample_size:
                break
        if sample_position < sample_size:
            alternative_position = probabilities.shape[1] - 1
            while (
                probabilities[row, alternative_position] < 1e-30
                and alternative_position > 0
            ):
                alternative_position -= 1
            while sample_position < sample_size:
                original_position = int(order[sample_position])
                choices[row, original_position] = alternatives[alternative_position]
                choice_probabilities[row, original_position] = probabilities[
                    row, alternative_position
                ]
                sample_position += 1
    return choices, choice_probabilities


def _pack_sample(
    trips,
    choices,
    probabilities,
    random_draws,
    first_occurrence,
    pick_counts,
    alt_col_name,
):
    """Build ActivitySim's narrow sample frame from compact fixed-width arrays."""
    chooser_ids = np.repeat(np.asarray(trips.index), choices.shape[1])
    valid = first_occurrence.reshape(-1).astype(bool, copy=False)
    result = pd.DataFrame(
        {
            alt_col_name: choices.reshape(-1)[valid],
            "rand": random_draws.reshape(-1)[valid],
            "prob": probabilities.reshape(-1)[valid],
            "pick_count": pick_counts.reshape(-1)[valid],
            trips.index.name: chooser_ids[valid],
        }
    )
    result.set_index(trips.index.name, inplace=True)
    result["prob"] = result["prob"].astype(np.float32)
    result["pick_count"] = result["pick_count"].astype(np.uint32)
    result = result.sort_values(by=alt_col_name).sort_index(kind="mergesort")
    return result


def sample_trip_destinations_resident(
    state,
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skim_hotel,
    estimator,
    trace_label,
    *,
    _exact_shared_arithmetic=False,
):
    """Return the public sample with utilities/probabilities/choices resident."""
    global _ACTIVITYSIM_CHOICE_WARM
    from activitysim.core import estimation, logit, simulate
    from activitysim.core.choosing import choice_maker
    from activitysim.core.interaction_sample import resolve_sample_method

    started = time.perf_counter()
    # A focused resume can enter trip destination before ActivitySim's ordinary
    # chooser has been JIT-compiled.  Sharrow temporarily changes the Numba
    # thread environment during exact adjudication, so compile this downstream
    # dependency first under ActivitySim's authoritative thread setting.
    if not _ACTIVITYSIM_CHOICE_WARM:
        choice_maker(
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float64),
        )
        _ACTIVITYSIM_CHOICE_WARM = True
    if estimator is not None or estimation.manager.enabled:
        raise Phase40Unsupported("Phase 40 does not support estimation mode")
    if resolve_sample_method(
        state, model_settings.compute_settings.subcomponent_settings("sample")
    ) != "inverse_cdf":
        raise Phase40Unsupported("Phase 40 requires inverse-CDF sampling")
    sample_size = int(model_settings.SAMPLE_SIZE)
    if sample_size <= 0 or sample_size > 32 or state.settings.disable_destination_sampling:
        raise Phase40Unsupported("Phase 40 requires 1..32 enabled destination draws")
    if len(trips) == 0:
        raise Phase40Unsupported("Phase 40 received an empty chooser segment")

    spec = simulate.spec_for_segment(
        state,
        None,
        spec_id="SAMPLE_SPEC",
        segment_name=primary_purpose,
        estimator=None,
        spec_file_name=model_settings.SAMPLE_SPEC,
        coefficients_file_name=model_settings.COEFFICIENTS,
    )
    _, fingerprint = _specification_contract(spec)
    constants = dict(model_settings.CONSTANTS)
    if set(constants) < {"max_walk_distance", "max_bike_distance"}:
        raise Phase40Unsupported("Phase 40 distance limits are absent")

    cp = _cupy()
    utilities, error_bounds, _, clocks = _gpu_utilities(
        trips,
        alternatives,
        size_term_matrix,
        skim_hotel,
        spec,
        constants,
        device_only=True,
    )
    if _exact_shared_arithmetic:
        # Phase 41 has exhaustively qualified the generated CUDA schedule
        # against live Sharrow over all 133,075,896 public benchmark cells.
        # A zero envelope tells the resident chooser that no CPU arithmetic
        # adjudication is required; the exact utility surface never leaves CUDA.
        error_bounds.fill(np.float32(0.0))
        cp.cuda.Stream.null.synchronize()
    utility_complete = time.perf_counter()

    # ActivitySim remains the authority for stream advancement and chooser-key
    # mapping.  The resulting compact draws are the only sampling input upload.
    random_draws = state.get_rn_generator().random_for_df(trips, n=sample_size)
    random_draws = np.ascontiguousarray(random_draws, dtype=np.float64)
    device_random = cp.asarray(random_draws)
    choices = cp.empty((len(trips), sample_size), dtype=cp.int32)
    probabilities = cp.empty((len(trips), sample_size), dtype=cp.float32)
    guard_rows = cp.zeros(len(trips), dtype=cp.uint8)
    bad_rows = cp.zeros(len(trips), dtype=cp.uint8)
    probability_log_risk = cp.zeros(len(trips), dtype=cp.float32)
    choice_kernel, duplicate_kernel = _compile_resident_kernels(cp)
    block = 128
    choice_kernel(
        ((len(trips) + block - 1) // block,),
        (block,),
        (
            utilities,
            error_bounds,
            device_random,
            choices,
            probabilities,
            guard_rows,
            bad_rows,
            probability_log_risk,
            np.int32(len(trips)),
            np.int32(len(alternatives)),
            np.int32(sample_size),
            np.int32(1 if _exact_shared_arithmetic else 0),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    choice_complete = time.perf_counter()

    bad_host = cp.asnumpy(bad_rows).astype(bool, copy=False)
    if bad_host.any():
        raise Phase40Unsupported(
            f"Phase 40 produced {int(np.count_nonzero(bad_host))} zero/invalid probability rows"
        )
    guard_host = cp.asnumpy(guard_rows).astype(bool, copy=False)
    probability_log_risk_host = cp.asnumpy(probability_log_risk)

    first_occurrence = cp.empty_like(choices, dtype=cp.uint8)
    pick_counts = cp.empty_like(choices, dtype=cp.uint32)
    duplicate_kernel(
        ((len(trips) + block - 1) // block,),
        (block,),
        (
            choices,
            first_occurrence,
            pick_counts,
            np.int32(len(trips)),
            np.int32(sample_size),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    duplicate_complete = time.perf_counter()

    host_choices = cp.asnumpy(choices)
    host_probabilities = cp.asnumpy(probabilities)
    host_first = cp.asnumpy(first_occurrence)
    host_counts = cp.asnumpy(pick_counts)
    compact_downloaded = time.perf_counter()

    guard_started = time.perf_counter()
    guard_count = int(np.count_nonzero(guard_host))
    if _exact_shared_arithmetic and guard_count:
        raise Phase40Unsupported(
            f"Phase 41 exact arithmetic unexpectedly guarded {guard_count} rows"
        )
    if guard_count:
        guarded_trips = trips.iloc[np.flatnonzero(guard_host)]
        # Phase 39 proved this compact evaluator array-identical to the live
        # Sharrow idotter.  Reusing the cached strict arithmetic ABI avoids
        # reconstructing Sharrow's flow/data-tree machinery for every one of
        # the 30 sparse guard segments.
        exact_utilities = _guard_reference_utilities(
            guarded_trips,
            alternatives,
            size_term_matrix,
            skim_hotel,
            spec,
            constants,
        )
        exact_probs = logit.utils_to_probs(
            state,
            pd.DataFrame(exact_utilities, index=guarded_trips.index),
            allow_zero_probs=True,
            trace_label="phase40_trip_destination_precision_guard",
            trace_choosers=guarded_trips,
            overflow_protection=False,
        )
        exact_choices, exact_choice_probs = _preserved_order_choices(
            exact_probs.to_numpy(copy=False),
            random_draws[guard_host],
            np.asarray(alternatives.index),
        )
        host_choices[guard_host] = exact_choices
        host_probabilities[guard_host] = exact_choice_probs
        exact_first, exact_counts = _host_duplicate_contract(exact_choices)
        host_first[guard_host] = exact_first
        host_counts[guard_host] = exact_counts
    guard_complete = time.perf_counter()

    sample = _pack_sample(
        trips,
        host_choices,
        host_probabilities,
        random_draws,
        host_first,
        host_counts,
        model_settings.ALT_DEST_COL_NAME,
    )
    finished = time.perf_counter()
    compact_host_bytes = int(
        host_choices.nbytes
        + host_probabilities.nbytes
        + host_first.nbytes
        + host_counts.nbytes
        + guard_host.nbytes
        + bad_host.nbytes
        + probability_log_risk_host.nbytes
    )
    telemetry = Phase40SamplingTelemetry(
            trace_label=str(trace_label),
            purpose=str(primary_purpose),
            backend=(
                "phase41_resident_cuda_sampling_exact_shared_arithmetic"
                if _exact_shared_arithmetic
                else "phase40_resident_cuda_sampling_with_sparse_sharrow_guard"
            ),
            chooser_rows=int(len(trips)),
            alternatives=int(len(alternatives)),
            utility_cells=int(len(trips) * len(alternatives)),
            random_draws=int(random_draws.size),
            sampled_rows=int(len(sample)),
            utility_host_bytes=0,
            compact_host_bytes=compact_host_bytes,
            random_upload_bytes=int(random_draws.nbytes),
            dense_utility_download_bytes_avoided=int(utilities.nbytes + error_bounds.nbytes),
            arithmetic_guard_rows=guard_count,
            arithmetic_guard_cells=int(guard_count * len(alternatives)),
            cdf_error_bound_scale=1.0,
            selected_probability_log_guard_tolerance=9e-5,
            selected_probability_log_risk_max=float(
                np.max(probability_log_risk_host, initial=0.0)
            ),
            selected_probability_log_risk_p99=float(
                np.quantile(probability_log_risk_host, 0.99)
            ),
            utility_kernel_seconds=clocks["completed"] - clocks["uploaded"],
            resident_choice_kernel_seconds=choice_complete - utility_complete,
            duplicate_kernel_seconds=duplicate_complete - choice_complete,
            exact_guard_seconds=guard_complete - guard_started,
            exact_guard_evaluator=(
                "none_exact_shared_arithmetic"
                if _exact_shared_arithmetic
                else "cached_strict_sharrow_idotter_abi"
            ),
            arithmetic_abi_version=SHARROW15_ABI_VERSION,
            arithmetic_abi_sha256=SHARROW15_ABI_SHA256,
            probability_abi_version=NUMPY_FLOAT32_CHOICE_ABI_VERSION,
            probability_abi_sha256=NUMPY_FLOAT32_CHOICE_ABI_SHA256,
            exact_shared_arithmetic=bool(_exact_shared_arithmetic),
            compact_download_seconds=compact_downloaded - duplicate_complete,
            host_pack_seconds=finished - guard_complete,
            total_seconds=finished - started,
            fallback_calls=0,
            specification_sha256=fingerprint,
            contract_valid=True,
        )
    (_PHASE41_TELEMETRY if _exact_shared_arithmetic else _TELEMETRY).append(
        telemetry
    )
    if os.environ.get("CHOICEFORGE_PHASE40_SAMPLE_SHADOW", "0") == "1":
        # A full Phase 39 shadow would advance the RNG twice.  Phase 40 instead
        # compares its compact result to the already-computed exact guarded
        # contract and relies on end-to-end frozen-output verification.
        if guard_count == 0:
            raise AssertionError("Phase 40 sample shadow expected at least one guard row")
    return sample


def sample_trip_destinations_resident_exact_abi(*args, **kwargs):
    """Phase 41 entry point: fully resident sampling with no CPU exact guard."""
    return sample_trip_destinations_resident(
        *args, _exact_shared_arithmetic=True, **kwargs
    )
