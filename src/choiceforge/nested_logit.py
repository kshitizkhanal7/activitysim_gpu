"""CUDA reduction for the canonical 21-alternative MTC mode-choice nest."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


MTC21_ALTERNATIVES = (
    "DRIVEALONEFREE", "DRIVEALONEPAY", "SHARED2FREE", "SHARED2PAY",
    "SHARED3FREE", "SHARED3PAY", "WALK", "BIKE", "WALK_LOC",
    "WALK_LRF", "WALK_EXP", "WALK_HVY", "WALK_COM", "DRIVE_LOC",
    "DRIVE_LRF", "DRIVE_EXP", "DRIVE_HVY", "DRIVE_COM", "TAXI",
    "TNC_SINGLE", "TNC_SHARED",
)

_SOURCE = r"""
extern "C" __device__ double lse(const double* x, int begin, int end, double scale) {
    double high = -1.7976931348623157e308;
    for (int j = begin; j < end; ++j) high = fmax(high, x[j] / scale);
    double total = 0.0;
    for (int j = begin; j < end; ++j) total += exp(x[j] / scale - high);
    return high + log(total);
}

extern "C" __global__ void mtc21_nested_logsum(
    const double* utility, double* result, long long rows,
    double auto_c, double auto_sub_c, double nm_c,
    double transit_c, double transit_sub_c, double ridehail_c) {
    long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    const double* u = utility + row * 21;

    double da = auto_sub_c * lse(u, 0, 2, auto_c * auto_sub_c);
    double sr2 = auto_sub_c * lse(u, 2, 4, auto_c * auto_sub_c);
    double sr3 = auto_sub_c * lse(u, 4, 6, auto_c * auto_sub_c);
    double auto_high = fmax(da, fmax(sr2, sr3));
    double auto_log = auto_c * (auto_high + log(
        exp(da-auto_high) + exp(sr2-auto_high) + exp(sr3-auto_high)));

    double nm_log = nm_c * lse(u, 6, 8, nm_c);
    double walk_access = transit_sub_c * lse(u, 8, 13, transit_c * transit_sub_c);
    double drive_access = transit_sub_c * lse(u, 13, 18, transit_c * transit_sub_c);
    double transit_high = fmax(walk_access, drive_access);
    double transit_log = transit_c * (transit_high + log(
        exp(walk_access-transit_high) + exp(drive_access-transit_high)));
    double ridehail_log = ridehail_c * lse(u, 18, 21, ridehail_c);

    double root_high = fmax(fmax(auto_log, nm_log), fmax(transit_log, ridehail_log));
    result[row] = root_high + log(exp(auto_log-root_high) + exp(nm_log-root_high)
        + exp(transit_log-root_high) + exp(ridehail_log-root_high));
}
"""

# Sharrow's generated nested-logit flow uses float32 storage and updates nests
# in a fixed bottom-up order.  This kernel deliberately mirrors that algorithm
# instead of algebraically regrouping log-sum-exp terms.  The distinction is
# observable for random draws very near a scheduling probability boundary.
_SHARROW_FLOAT32_SOURCE = r"""
extern "C" __device__ void sharrow_nest(
    float* utility, int up, int begin, int end, float mu) {
    float shifter = -3.402823466e38F;
    int shifter_position = -1;
    for (int child = begin; child < end; ++child) {
        if (utility[child] > -3.402823466e38F) {
            float z = utility[child] / mu;
            if (z > shifter) {
                shifter = z;
                shifter_position = child;
            }
        }
    }
    for (int child = begin; child < end; ++child) {
        if (utility[child] > -3.402823466e38F) {
            if (child == shifter_position) utility[up] += 1.0f;
            else utility[up] += expf(utility[child] / mu - shifter);
        }
    }
    utility[up] = (logf(utility[up]) + shifter) * mu;
}

extern "C" __device__ void sharrow_root(float* utility) {
    const int children[4] = {24, 25, 28, 29};
    float shifter = -3.402823466e38F;
    int shifter_position = -1;
    for (int n = 0; n < 4; ++n) {
        int child = children[n];
        float z = utility[child];
        if (z > shifter) {
            shifter = z;
            shifter_position = child;
        }
    }
    for (int n = 0; n < 4; ++n) {
        int child = children[n];
        if (child == shifter_position) utility[30] += 1.0f;
        else utility[30] += expf(utility[child] - shifter);
    }
    utility[30] = logf(utility[30]) + shifter;
}

extern "C" __global__ void mtc21_sharrow_nested_logsum(
    const float* input, float* result, long long rows,
    float auto_c, float auto_sub_c, float nm_c,
    float transit_c, float transit_sub_c, float ridehail_c) {
    long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    float utility[31] = {0.0f};
    for (int alternative = 0; alternative < 21; ++alternative)
        utility[alternative] = input[row * 21 + alternative];
    sharrow_nest(utility, 21, 0, 2, auto_sub_c);
    sharrow_nest(utility, 22, 2, 4, auto_sub_c);
    sharrow_nest(utility, 23, 4, 6, auto_sub_c);
    sharrow_nest(utility, 24, 21, 24, auto_c);
    sharrow_nest(utility, 25, 6, 8, nm_c);
    sharrow_nest(utility, 26, 8, 13, transit_sub_c);
    sharrow_nest(utility, 27, 13, 18, transit_sub_c);
    sharrow_nest(utility, 28, 26, 28, transit_c);
    sharrow_nest(utility, 29, 18, 21, ridehail_c);
    sharrow_root(utility);
    result[row] = utility[30];
}
"""

# ActivitySim's public-model logsum path uses Sharrow for float32 utilities,
# then its pandas reducer promotes those utilities to float64 before applying
# exp, ordered child sums, log, and exp for each nest.  Preserve those exact
# algebraic boundaries here; the stable LSE kernel above remains the default
# for callers that do not require legacy scheduling replay semantics.
_ACTIVITYSIM_PANDAS_SOURCE = r"""
extern "C" __device__ double pandas_nest(
    const double* utility, int begin, int end, double coefficient) {
    double total = 0.0;
    for (int child = begin; child < end; ++child) total += utility[child];
    return exp(coefficient * log(total));
}

extern "C" __global__ void mtc21_activitysim_pandas_logsum(
    const double* input, double* result, long long rows,
    double auto_c, double auto_sub_c, double nm_c,
    double transit_c, double transit_sub_c, double ridehail_c) {
    long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    double utility[31] = {0.0};
    const double* raw = input + row * 21;
    double auto_leaf = auto_c * auto_sub_c;
    double transit_leaf = transit_c * transit_sub_c;
    for (int alternative = 0; alternative < 6; ++alternative)
        utility[alternative] = exp(raw[alternative] / auto_leaf);
    for (int alternative = 6; alternative < 8; ++alternative)
        utility[alternative] = exp(raw[alternative] / nm_c);
    for (int alternative = 8; alternative < 18; ++alternative)
        utility[alternative] = exp(raw[alternative] / transit_leaf);
    for (int alternative = 18; alternative < 21; ++alternative)
        utility[alternative] = exp(raw[alternative] / ridehail_c);

    utility[21] = pandas_nest(utility, 0, 2, auto_sub_c);
    utility[22] = pandas_nest(utility, 2, 4, auto_sub_c);
    utility[23] = pandas_nest(utility, 4, 6, auto_sub_c);
    utility[24] = pandas_nest(utility, 21, 24, auto_c);
    utility[25] = pandas_nest(utility, 6, 8, nm_c);
    utility[26] = pandas_nest(utility, 8, 13, transit_sub_c);
    utility[27] = pandas_nest(utility, 13, 18, transit_sub_c);
    utility[28] = pandas_nest(utility, 26, 28, transit_c);
    utility[29] = pandas_nest(utility, 18, 21, ridehail_c);
    double root_total = utility[24] + utility[25] + utility[28] + utility[29];
    utility[30] = exp(log(root_total));
    result[row] = log(utility[30]);
}
"""

_KERNEL = None
_SHARROW_FLOAT32_KERNEL = None
_ACTIVITYSIM_PANDAS_KERNEL = None


@dataclass(frozen=True)
class NestedLogitTelemetry:
    rows: int
    input_bytes: int
    host_to_device_ms: float
    kernel_ms: float
    device_to_host_ms: float


def _children(node):
    return node["alternatives"] if isinstance(node, dict) else node.alternatives


def _coefficient(node):
    return float(node["coefficient"] if isinstance(node, dict) else node.coefficient)


def _name(node):
    return str(node["name"] if isinstance(node, dict) else node.name)


def mtc21_coefficients(nest_spec) -> tuple[float, ...]:
    """Validate the canonical topology and return its six coefficients."""
    root = {_name(node): node for node in _children(nest_spec)}
    expected = {"AUTO", "NONMOTORIZED", "TRANSIT", "RIDEHAIL"}
    if set(root) != expected or _coefficient(nest_spec) != 1.0:
        raise ValueError("not the canonical MTC 21-alternative root nest")
    auto = {_name(node): node for node in _children(root["AUTO"])}
    transit = {_name(node): node for node in _children(root["TRANSIT"])}
    if set(auto) != {"DRIVEALONE", "SHAREDRIDE2", "SHAREDRIDE3"}:
        raise ValueError("unsupported AUTO subnest")
    if set(transit) != {"WALKACCESS", "DRIVEACCESS"}:
        raise ValueError("unsupported TRANSIT subnest")
    auto_sub = {_coefficient(node) for node in auto.values()}
    transit_sub = {_coefficient(node) for node in transit.values()}
    if len(auto_sub) != 1 or len(transit_sub) != 1:
        raise ValueError("nonuniform MTC subnest coefficients")
    return (
        _coefficient(root["AUTO"]), auto_sub.pop(),
        _coefficient(root["NONMOTORIZED"]), _coefficient(root["TRANSIT"]),
        transit_sub.pop(), _coefficient(root["RIDEHAIL"]),
    )


def mtc21_nested_logsums_cuda(
    utilities,
    nest_spec,
    alternatives=MTC21_ALTERNATIVES,
    *,
    return_telemetry=False,
    return_device=False,
    numeric_policy="activitysim_float64",
):
    """Return transfer-inclusive logsums on the host under a named policy.

    ``return_telemetry`` exposes the unavoidable host/device transfers while
    ActivitySim/Sharrow still produces its utility matrix on the CPU.  This is
    intentionally measured separately from kernel execution so a future fused
    utility evaluator has a concrete before/after target. The ActivitySim
    policies return float64; ``sharrow_float32`` preserves float32 throughout.
    ``return_device`` keeps the modeled logsum vector on CUDA so a downstream
    GPU component can consume it without a host materialization.
    """
    if tuple(alternatives) != MTC21_ALTERNATIVES:
        raise ValueError("mode columns are not in canonical MTC order")
    from .cuda_backend import _cupy

    if numeric_policy not in {
        "activitysim_float64",
        "activitysim_pandas_float64",
        "sharrow_float32",
    }:
        raise ValueError(f"unknown nested-logit numeric policy {numeric_policy!r}")
    cp = _cupy()
    dtype = cp.float32 if numeric_policy == "sharrow_float32" else cp.float64
    device_input = hasattr(utilities, "__cuda_array_interface__")
    if device_input:
        device_values = cp.ascontiguousarray(utilities, dtype=dtype)
        if device_values.ndim != 2:
            raise ValueError("expected a row-by-21 utility matrix")
        rows, columns = device_values.shape
        input_bytes = int(device_values.nbytes)
    else:
        host_dtype = np.float32 if numeric_policy == "sharrow_float32" else np.float64
        values = np.ascontiguousarray(utilities, dtype=host_dtype)
        rows, columns = values.shape if values.ndim == 2 else (0, 0)
        input_bytes = int(values.nbytes)
    if columns != 21:
        raise ValueError("expected a row-by-21 utility matrix")
    coefficients = mtc21_coefficients(nest_spec)

    global _KERNEL, _SHARROW_FLOAT32_KERNEL, _ACTIVITYSIM_PANDAS_KERNEL
    if numeric_policy == "sharrow_float32":
        if _SHARROW_FLOAT32_KERNEL is None:
            _SHARROW_FLOAT32_KERNEL = cp.RawKernel(
                _SHARROW_FLOAT32_SOURCE, "mtc21_sharrow_nested_logsum"
            )
        kernel = _SHARROW_FLOAT32_KERNEL
    elif numeric_policy == "activitysim_pandas_float64":
        if _ACTIVITYSIM_PANDAS_KERNEL is None:
            _ACTIVITYSIM_PANDAS_KERNEL = cp.RawKernel(
                _ACTIVITYSIM_PANDAS_SOURCE,
                "mtc21_activitysim_pandas_logsum",
            )
        kernel = _ACTIVITYSIM_PANDAS_KERNEL
    else:
        if _KERNEL is None:
            _KERNEL = cp.RawKernel(_SOURCE, "mtc21_nested_logsum")
        kernel = _KERNEL
    start = time.perf_counter()
    if not device_input:
        device_values = cp.asarray(values)
        cp.cuda.Stream.null.synchronize()
        after_upload = time.perf_counter()
    else:
        # Device coercion occurred before the transfer timer; it is not a
        # host-to-device transfer and must not be charged as one.
        after_upload = start
    device_result = cp.empty(rows, dtype=dtype)
    threads = 256
    kernel_coefficients = (
        tuple(np.float32(x) for x in coefficients)
        if numeric_policy == "sharrow_float32"
        else coefficients
    )
    kernel(
        ((rows + threads - 1) // threads,),
        (threads,),
        (device_values, device_result, np.int64(rows), *kernel_coefficients),
    )
    cp.cuda.Stream.null.synchronize()
    after_kernel = time.perf_counter()
    result = device_result if return_device else cp.asnumpy(device_result)
    after_download = time.perf_counter()
    if not return_telemetry:
        return result
    return result, NestedLogitTelemetry(
        rows=rows,
        input_bytes=input_bytes,
        host_to_device_ms=(after_upload - start) * 1000,
        kernel_ms=(after_kernel - after_upload) * 1000,
        device_to_host_ms=(
            0.0 if return_device else (after_download - after_kernel) * 1000
        ),
    )
