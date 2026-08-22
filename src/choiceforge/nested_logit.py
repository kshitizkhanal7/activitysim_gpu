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

_KERNEL = None


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
    utilities, nest_spec, alternatives=MTC21_ALTERNATIVES, *, return_telemetry=False
):
    """Return transfer-inclusive float64 logsums on the host.

    ``return_telemetry`` exposes the unavoidable host/device transfers while
    ActivitySim/Sharrow still produces its utility matrix on the CPU.  This is
    intentionally measured separately from kernel execution so a future fused
    utility evaluator has a concrete before/after target.
    """
    if tuple(alternatives) != MTC21_ALTERNATIVES:
        raise ValueError("mode columns are not in canonical MTC order")
    from .cuda_backend import _cupy

    cp = _cupy()
    device_input = hasattr(utilities, "__cuda_array_interface__")
    if device_input:
        device_values = cp.ascontiguousarray(utilities, dtype=cp.float64)
        if device_values.ndim != 2:
            raise ValueError("expected a row-by-21 utility matrix")
        rows, columns = device_values.shape
        input_bytes = int(device_values.nbytes)
    else:
        values = np.ascontiguousarray(utilities, dtype=np.float64)
        rows, columns = values.shape if values.ndim == 2 else (0, 0)
        input_bytes = int(values.nbytes)
    if columns != 21:
        raise ValueError("expected a row-by-21 utility matrix")
    coefficients = mtc21_coefficients(nest_spec)

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SOURCE, "mtc21_nested_logsum")
    start = time.perf_counter()
    if not device_input:
        device_values = cp.asarray(values)
        cp.cuda.Stream.null.synchronize()
        after_upload = time.perf_counter()
    else:
        # Device coercion occurred before the transfer timer; it is not a
        # host-to-device transfer and must not be charged as one.
        after_upload = start
    device_result = cp.empty(rows, dtype=cp.float64)
    threads = 256
    _KERNEL(
        ((rows + threads - 1) // threads,),
        (threads,),
        (device_values, device_result, np.int64(rows), *coefficients),
    )
    cp.cuda.Stream.null.synchronize()
    after_kernel = time.perf_counter()
    result = cp.asnumpy(device_result)
    after_download = time.perf_counter()
    if not return_telemetry:
        return result
    return result, NestedLogitTelemetry(
        rows=rows,
        input_bytes=input_bytes,
        host_to_device_ms=(after_upload - start) * 1000,
        kernel_ms=(after_kernel - after_upload) * 1000,
        device_to_host_ms=(after_download - after_kernel) * 1000,
    )
