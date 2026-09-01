"""Phase 48 resident sampled-destination probability and choice backend.

The public interface mirrors Sharrow's fail-closed compiled-flow idea: a
versioned backend advertises reviewed programs, prewarms them, and evaluates a
replaceable compact state.  ActivitySim remains the orchestration and random-
ledger authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .arithmetic_abi import numpy_float32_pairwise_sum_cuda_helpers
from .modelwide_sampling import _compile_phase46_choice, numpy_preserved_order_choices


_CHOICE_KERNELS = {}
_EXP_CORRECTION_KERNEL = None
_EXP_CORRECTION_DEVICE = None
_TELEMETRY = []

# Exhaustive uint32-domain scan of NumPy 2.4.6's Windows x86-v3 float32 exp
# against numpy_avx2_expf.  The public destination contract is fail-closed
# outside [-80, 80] (except its -999 padding), so only the 73 differences in
# that complete domain belong in the resident correction table.
_EXP_CORRECTIONS = (
    (1065686418, 1077216500), (1075528661, 1093993716),
    (1081343201, 1110770932), (1086097722, 1135936758),
    (1087551356, 1144325361), (1089004992, 1152713974),
    (1090458626, 1161102577), (1091215651, 1169491190),
    (1092669286, 1186268406), (1094122921, 1203045622),
    (1094849738, 1211434225), (1095576556, 1219822838),
    (1096303373, 1228211441), (1097030191, 1236600054),
    (1097757008, 1244988657), (1098483826, 1253377270),
    (1100149372, 1286931702), (1101603007, 1320486134),
    (1109900763, 1572144386), (1110082467, 1580532979),
    (1110627581, 1605698840), (1110809284, 1614087387),
    (1111172693, 1630864615), (1111354398, 1639253250),
    (1111536102, 1647641843), (1112081216, 1672807705),
    (1112626328, 1697973479), (1112808033, 1706362114),
    (1112989737, 1714750707), (1113534851, 1739916569),
    (1114079963, 1765082343), (1114261668, 1773470978),
    (1114443372, 1781859571), (1114988486, 1807025433),
    (1115533598, 1832191208), (1116426901, 1907688708),
    (1116517753, 1916077299), (1116790310, 1941243161),
    (1117062866, 1966408936), (1117335423, 1991574798),
    (1117426275, 1999963390), (3213170066, 1052050674),
    (3223012309, 1035273458), (3228826849, 1018496242),
    (3247269611, 850724085), (3247633020, 842335472),
    (3248723246, 817169653), (3249086655, 808781040),
    (3249813473, 792003813), (3250176881, 783615221),
    (3251267108, 758449381), (3251630516, 750060789),
    (3252720743, 724894949), (3253084151, 716506357),
    (3254174378, 691340517), (3254537786, 682951925),
    (3255930776, 624231653), (3256112480, 615843061),
    (3257384411, 557122788), (3258111229, 523568333),
    (3258292932, 515179786), (3258838046, 490013924),
    (3259564864, 456459469), (3260473385, 414516467),
    (3261018499, 389350605), (3261927020, 347407603),
    (3262472134, 322241740), (3263365436, 271910104),
    (3263910549, 221578467), (3264001401, 213189875),
    (3264273958, 188024012), (3264546514, 162858239),
    (3264819071, 137692375),
)
_EXP_CORRECTION_SHA256 = "7d381f55dfc0a244bda39418af15e79d6f81980c194138b43d24dce9e1affe69"


@dataclass(frozen=True)
class DestinationBackendContract:
    """Portable declaration of the reviewed Phase 48 backend boundary."""

    name: str = "choiceforge.cuda.destination_graph"
    version: int = 1
    execution_modes: tuple[str, ...] = ("require", "test")
    utility_dtype: str = "float32"
    random_dtype: str = "float64-mt19937"
    widths: tuple[int, ...] = (21, 25, 29, 30)
    unknown_program_policy: str = "fail_closed"
    logsum_policy: str = "gpu_pairwise_sum_then_numpy_scalar_log"
    exponential_policy: str = "numpy246-x86v3-exhaustive-domain-correction-v1"
    exponential_domain: tuple[float, float] = (-80.0, 80.0)
    padding_utility: float = -999.0
    exponential_correction_sha256: str = _EXP_CORRECTION_SHA256

    def document(self, programs) -> dict:
        payload = {
            **self.__dict__,
            "execution_modes": list(self.execution_modes),
            "widths": list(self.widths),
            "programs": [list(program) for program in programs],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["abi_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload


CONTRACT = DestinationBackendContract()


def reset_phase48_telemetry():
    _TELEMETRY.clear()


def phase48_telemetry():
    return list(_TELEMETRY)


def _compile_exp_correction(cp):
    """Compile the exhaustive-domain NumPy exp correction and cache its LUT."""
    global _EXP_CORRECTION_KERNEL, _EXP_CORRECTION_DEVICE
    if _EXP_CORRECTION_KERNEL is not None:
        return _EXP_CORRECTION_KERNEL, _EXP_CORRECTION_DEVICE, True
    source = r'''
extern "C" __global__ void phase48_numpy_exp_correction(
    const float* utilities,
    const float* row_maxima,
    float* weights,
    unsigned char* bad_rows,
    const unsigned int* input_bits,
    const unsigned int* output_bits,
    long long cells,
    int width,
    int correction_count)
{
    const long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    const int row = (int)(cell / width);
    const float value = __fsub_rn(utilities[cell], row_maxima[row]);
    if ((value < -80.0f || value > 80.0f) && value != -999.0f) {
        bad_rows[row] = 1;
        return;
    }
    const unsigned int bits = __float_as_uint(value);
    int low = 0;
    int high = correction_count;
    while (low < high) {
        const int middle = low + ((high - low) >> 1);
        if (input_bits[middle] < bits) low = middle + 1;
        else high = middle;
    }
    if (low < correction_count && input_bits[low] == bits) {
        weights[cell] = __uint_as_float(output_bits[low]);
    }
}
'''
    kernel = cp.RawKernel(
        source,
        "phase48_numpy_exp_correction",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    kernel.compile()
    pairs = np.asarray(_EXP_CORRECTIONS, dtype=np.uint32)
    _EXP_CORRECTION_DEVICE = (cp.asarray(pairs[:, 0]), cp.asarray(pairs[:, 1]))
    _EXP_CORRECTION_KERNEL = kernel
    return kernel, _EXP_CORRECTION_DEVICE, False


def _compile_resident_choice(cp, width: int):
    """Compile one-draw exact-probability selection with an exported row sum."""
    width = int(width)
    if width not in CONTRACT.widths:
        raise ValueError(f"Phase 48 unsupported compact width: {width}")
    if width in _CHOICE_KERNELS:
        return _CHOICE_KERNELS[width], True
    helpers = numpy_float32_pairwise_sum_cuda_helpers(width)
    source = helpers + r'''
extern "C" __global__ void phase48_resident_final_choice(
    const float* weights,
    const double* random_draws,
    int* choices,
    float* selected_probabilities,
    float* row_totals,
    unsigned char* guard_rows,
    unsigned char* bad_rows,
    int chooser_rows,
    int width)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= chooser_rows) return;
    const long long base = (long long)row * width;
    const float total = PAIRWISE_SUM(weights + base);
    row_totals[row] = total;
    if (!(total > 0.0f) || !isfinite(total)) {
        bad_rows[row] = 1;
        return;
    }
    const double random_draw = random_draws[row];
    const double reserve = 5.0e-7;
    double prefix = 0.0;
    int last_nontrivial = width - 1;
    bool selected = false;
    for (int alternative = 0; alternative < width; ++alternative) {
        const float weight = weights[base + alternative];
        const float probability = weight > 0.0f ? weight / total : 0.0f;
        const double previous = prefix;
        prefix += (double)probability;
        if (probability >= 1.0e-30f) last_nontrivial = alternative;
        if (!selected && prefix > random_draw) {
            choices[row] = alternative;
            selected_probabilities[row] = probability;
            guard_rows[row] = (
                fabs(prefix - random_draw) <= reserve ||
                fabs(random_draw - previous) <= reserve) ? 1 : 0;
            selected = true;
        }
    }
    if (!selected) {
        choices[row] = last_nontrivial;
        selected_probabilities[row] = weights[base + last_nontrivial] / total;
        guard_rows[row] = 1;
    }
}
'''.replace("PAIRWISE_SUM", f"numpy_pairwise_sum_{width}")
    kernel = cp.RawKernel(
        source,
        "phase48_resident_final_choice",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    kernel.compile()
    _CHOICE_KERNELS[width] = kernel
    return kernel, False


def prewarm_phase48_public_runtime(cp=None) -> dict:
    from .cuda_backend import _cupy
    from .modelwide_final import PUBLIC_FINAL_PROGRAMS

    cp = cp or _cupy()
    started = time.perf_counter()
    compiled = 0
    _, _, correction_hit = _compile_exp_correction(cp)
    for width in CONTRACT.widths:
        _, hit = _compile_resident_choice(cp, width)
        compiled += int(not hit)
    cp.cuda.Stream.null.synchronize()
    return {
        "widths": list(CONTRACT.widths),
        "new_choice_programs_compiled": compiled,
        "new_exp_correction_programs_compiled": int(not correction_hit),
        "exp_correction_entries": len(_EXP_CORRECTIONS),
        "contract": CONTRACT.document(PUBLIC_FINAL_PROGRAMS),
        "seconds": time.perf_counter() - started,
    }


def finish_resident_final_choice(
    *,
    state,
    choosers,
    alternatives,
    choice_column,
    allow_zero_probs,
    zero_prob_choice_val,
    want_logsums,
    trace_label,
    telemetry,
    component,
    service,
    workspace,
    padded,
    offsets,
    width,
    expressions,
    program_cache_hit,
    started,
    prepared,
    utility_complete,
    shadow_reference=None,
    resident_mode_logsum_device=None,
):
    """Finish Phase 47 utility through a resident exact probability graph."""
    from activitysim.core import logit

    if allow_zero_probs:
        raise ValueError("Phase 48 public final graph does not allow zero probabilities")
    cp = service.cp
    rows = len(choosers)
    row_maxima = workspace["row_maxima"]
    weights = workspace["weights"]
    # ActivitySim disables overflow shifting when skip_failed_choices is active,
    # even when its caller requested protection.  Preserve that operational
    # rule exactly; algebraically equivalent shifting can move float32 CDF bits.
    shifts_applied = not state.settings.skip_failed_choices
    if shifts_applied:
        cp.max(padded, axis=1, out=row_maxima)
    else:
        row_maxima.fill(np.float32(0.0))
    weight_kernel, _, _ = _compile_phase46_choice(cp, width)
    cells = padded.size
    weight_kernel(
        ((cells + 255) // 256,),
        (256,),
        (
            padded,
            row_maxima,
            weights,
            np.int64(cells),
            np.int32(width),
        ),
    )
    correction_kernel, correction_device, correction_cache_hit = (
        _compile_exp_correction(cp)
    )
    correction_kernel(
        ((cells + 255) // 256,),
        (256,),
        (
            padded,
            row_maxima,
            weights,
            workspace["bad"],
            correction_device[0],
            correction_device[1],
            np.int64(cells),
            np.int32(width),
            np.int32(len(_EXP_CORRECTIONS)),
        ),
    )
    _, device_draws = service.random_for_df(
        state, choosers, 1, device_only=True
    )
    choice_kernel, choice_cache_hit = _compile_resident_choice(cp, width)
    choice_kernel(
        ((rows + 127) // 128,),
        (128,),
        (
            weights,
            device_draws,
            workspace["positions"],
            workspace["selected_probabilities"],
            workspace["row_totals"],
            workspace["guard"],
            workspace["bad"],
            np.int32(rows),
            np.int32(width),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    probability_complete = time.perf_counter()
    if int(cp.count_nonzero(workspace["bad"]).get()):
        raise ValueError("Phase 48 produced zero or invalid final probabilities")

    gpu_positions = cp.asnumpy(workspace["positions"])
    guard = cp.asnumpy(workspace["guard"]).astype(bool, copy=False)
    guard_rows = np.flatnonzero(guard)
    transfer_bytes = rows * (4 + 1)
    selected_positions = gpu_positions
    pre_guard_mismatches = 0
    if len(guard_rows):
        guard_utilities = cp.asnumpy(padded[cp.asarray(guard_rows)])
        guard_draws = cp.asnumpy(device_draws[cp.asarray(guard_rows)])
        transfer_bytes += guard_utilities.nbytes + guard_draws.nbytes
        exact_probs = logit.utils_to_probs(
            state,
            pd.DataFrame(guard_utilities, index=choosers.index[guard_rows]),
            allow_zero_probs=False,
            trace_label=str(trace_label) + ".phase48_exact_guard",
            trace_choosers=choosers.iloc[guard_rows],
            overflow_protection=True,
        ).to_numpy(copy=False)
        exact_positions, _ = numpy_preserved_order_choices(
            exact_probs,
            guard_draws,
            np.arange(width, dtype=np.int32),
        )
        exact_positions = exact_positions[:, 0]
        pre_guard_mismatches = int(np.count_nonzero(
            selected_positions[guard_rows] != exact_positions
        ))
        selected_positions[guard_rows] = exact_positions

    logsums = None
    if want_logsums:
        totals_host = cp.asnumpy(workspace["row_totals"])
        maxima_host = cp.asnumpy(row_maxima)
        transfer_bytes += totals_host.nbytes + maxima_host.nbytes
        with np.errstate(divide="warn"):
            logsum_values = np.log(totals_host)
        logsum_values += maxima_host
        logsums = pd.Series(logsum_values, index=choosers.index)
    transfer_complete = time.perf_counter()

    shadow = os.environ.get("CHOICEFORGE_PHASE48_SHADOW", "0") == "1"
    utility_bit_mismatches = 0
    weight_bit_mismatches = 0
    total_bit_mismatches = 0
    probability_bit_mismatches = 0
    choice_mismatches = 0
    logsum_bit_mismatches = 0
    shadow_utility_min = None
    shadow_utility_max = None
    if shadow:
        utility_host = cp.asnumpy(padded)
        reference_utility = np.asarray(shadow_reference(), dtype=np.float32)
        actual_utility = np.concatenate([
            utility_host[row, : int(offsets[row + 1] - offsets[row])]
            for row in range(rows)
        ])
        shadow_utility_min = float(actual_utility.min())
        shadow_utility_max = float(actual_utility.max())
        utility_bit_mismatches = int(np.count_nonzero(
            actual_utility.view(np.uint32) != reference_utility.view(np.uint32)
        ))
        shifted = (
            utility_host - utility_host.max(axis=1, keepdims=True)
            if shifts_applied
            else utility_host
        )
        reference_weights = np.exp(shifted)
        np.putmask(reference_weights, reference_weights <= logit.EXP_UTIL_MIN, 0)
        reference_totals = reference_weights.sum(axis=1)
        device_weights = cp.asnumpy(weights)
        device_totals = cp.asnumpy(workspace["row_totals"])
        weight_bit_mismatches = int(np.count_nonzero(
            device_weights.view(np.uint32) != reference_weights.view(np.uint32)
        ))
        total_bit_mismatches = int(np.count_nonzero(
            device_totals.view(np.uint32) != reference_totals.view(np.uint32)
        ))
        device_probabilities = cp.asnumpy(
            weights / workspace["row_totals"][:, None]
        )
        reference_probabilities = reference_weights / reference_totals[:, None]
        probability_bit_mismatches = int(np.count_nonzero(
            device_probabilities.view(np.uint32)
            != reference_probabilities.view(np.uint32)
        ))
        all_draws = cp.asnumpy(device_draws)
        exact_all, _ = numpy_preserved_order_choices(
            reference_probabilities,
            all_draws,
            np.arange(width, dtype=np.int32),
        )
        choice_mismatches = int(np.count_nonzero(
            selected_positions != exact_all[:, 0]
        ))
        if want_logsums:
            reference_logsums = np.log(reference_totals)
            if shifts_applied:
                reference_logsums += utility_host.max(axis=1)
            logsum_bit_mismatches = int(np.count_nonzero(
                np.asarray(logsums).view(np.uint32)
                != reference_logsums.view(np.uint32)
            ))
        proof = {
            "trace_label": str(trace_label),
            "utility_bits": utility_bit_mismatches,
            "weight_bits": weight_bit_mismatches,
            "total_bits": total_bit_mismatches,
            "probability_bits": probability_bit_mismatches,
            "choice_mismatches": choice_mismatches,
            "logsum_bits": logsum_bit_mismatches,
        }
        if weight_bit_mismatches:
            mismatch = np.argwhere(
                device_weights.view(np.uint32) != reference_weights.view(np.uint32)
            )[0]
            mismatch_row, mismatch_alt = (int(mismatch[0]), int(mismatch[1]))
            proof["first_weight_mismatch"] = {
                "row": mismatch_row,
                "alternative": mismatch_alt,
                "utility": float(shifted[mismatch_row, mismatch_alt]),
                "utility_bits": int(
                    shifted.view(np.uint32)[mismatch_row, mismatch_alt]
                ),
                "device_weight": float(device_weights[mismatch_row, mismatch_alt]),
                "device_weight_bits": int(
                    device_weights.view(np.uint32)[mismatch_row, mismatch_alt]
                ),
                "numpy_weight": float(reference_weights[mismatch_row, mismatch_alt]),
                "numpy_weight_bits": int(
                    reference_weights.view(np.uint32)[mismatch_row, mismatch_alt]
                ),
            }
        print("PHASE48_GRAPH_SHADOW " + repr(proof), flush=True)
        mismatch_keys = (
            "utility_bits", "weight_bits", "total_bits", "probability_bits",
            "choice_mismatches", "logsum_bits",
        )
        if any(proof[key] for key in mismatch_keys):
            raise RuntimeError("Phase 48 resident graph shadow failed: " + repr(proof))
    shadow_complete = time.perf_counter()

    selected_rows = offsets[:-1] + selected_positions.astype(np.int64)
    destination_bridge = getattr(service, "destination_supergraph_bridge", None)
    if destination_bridge is not None:
        if resident_mode_logsum_device is None:
            raise RuntimeError("Phase 49 resident logsum device vector is absent")
        destination_bridge.capture_selected(
            np.asarray(choosers.index, dtype=np.int64),
            selected_rows,
            component=str(component),
        )
    choices = pd.Series(
        alternatives[choice_column].to_numpy(copy=False)[selected_rows],
        index=choosers.index,
    )
    if zero_prob_choice_val is not None:
        # The public contract disallows zero-probability rows, so this is a no-op.
        pass
    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = logsums
    finished = time.perf_counter()
    event = {
        "component": str(component),
        "trace_label": str(trace_label),
        "chooser_rows": rows,
        "alternative_rows": len(alternatives),
        "max_alternatives": width,
        "program_terms": len(expressions),
        "utility_program_cache_hit": bool(program_cache_hit),
        "choice_program_cache_hit": bool(choice_cache_hit),
        "exp_correction_program_cache_hit": bool(correction_cache_hit),
        "exp_correction_entries": len(_EXP_CORRECTIONS),
        "exp_correction_sha256": _EXP_CORRECTION_SHA256,
        "prepare_seconds": prepared - started,
        "utility_seconds": utility_complete - prepared,
        "resident_probability_choice_seconds": probability_complete - utility_complete,
        "compact_transfer_guard_logsum_seconds": transfer_complete - probability_complete,
        "shadow_seconds": shadow_complete - transfer_complete,
        "pack_seconds": finished - shadow_complete,
        "total_seconds": finished - started,
        "exact_guard_rows": len(guard_rows),
        "pre_guard_mismatches": pre_guard_mismatches,
        "device_to_host_bytes": int(transfer_bytes),
        "dense_utility_download_bytes_avoided": int(padded.nbytes),
        "want_logsums": bool(want_logsums),
        "overflow_shifts_applied": shifts_applied,
        "utility_shadow_bit_mismatches": utility_bit_mismatches,
        "weight_shadow_bit_mismatches": weight_bit_mismatches,
        "total_shadow_bit_mismatches": total_bit_mismatches,
        "probability_shadow_bit_mismatches": probability_bit_mismatches,
        "choice_shadow_mismatches": choice_mismatches,
        "logsum_shadow_bit_mismatches": logsum_bit_mismatches,
        "shadow_utility_min": shadow_utility_min,
        "shadow_utility_max": shadow_utility_max,
        "runtime": "phase48_resident_destination_graph",
    }
    _TELEMETRY.append(event)
    if telemetry is not None:
        telemetry.append(event)
    return choices


def summarize_phase48_telemetry(events=None) -> dict:
    events = list(_TELEMETRY if events is None else events)
    return {
        "calls": len(events),
        "chooser_rows": sum(item["chooser_rows"] for item in events),
        "alternative_rows": sum(item["alternative_rows"] for item in events),
        "guard_rows": sum(item["exact_guard_rows"] for item in events),
        "device_to_host_bytes": sum(item["device_to_host_bytes"] for item in events),
        "dense_utility_download_bytes_avoided": sum(
            item["dense_utility_download_bytes_avoided"] for item in events
        ),
        "seconds": sum(item["total_seconds"] for item in events),
        "events": events,
    }
