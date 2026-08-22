"""A strict, device-resident ABI for lowered destination utility models.

ActivitySim/Sharrow evaluates an MTC trip-mode specification as Python-like
expressions over chooser attributes and skims.  Moving that whole language to
the GPU in one step would be unsafe.  This module defines the smaller boundary
needed to move it incrementally: once expressions have been *lowered* to a
dense numeric feature matrix, calculate all alternative utilities on the
device, without copying them back before the nested-logsum kernel.

The API deliberately does not parse ActivitySim expressions.  Callers must
provide the feature order explicitly and validate it against the CPU reference
before enabling this backend for a production model.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


_ORDERED_F32_SOURCE = r'''
extern "C" __global__ void ordered_linear_f32(
    const float* features, const float* coefficients, const float* constants,
    float* result, long long rows, int feature_count, int alternative_count) {
    long long item = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = rows * alternative_count;
    if (item >= total) return;
    long long row = item / alternative_count;
    int alternative = (int)(item - row * alternative_count);
    float value = constants[alternative];
    for (int feature = 0; feature < feature_count; ++feature) {
        value = value + features[row * feature_count + feature] *
            coefficients[feature * alternative_count + alternative];
    }
    result[item] = value;
}
'''
_ORDERED_F32_KERNEL = None


@dataclass(frozen=True)
class DestinationUtilityTelemetry:
    """Transfer and GEMM timings for one utility evaluation."""

    rows: int
    features: int
    alternatives: int
    input_bytes: int
    host_to_device_ms: float
    kernel_ms: float
    device_to_host_ms: float


@dataclass(frozen=True)
class DestinationLogsumTelemetry:
    """Timing for a lowered-utility-to-MTC-logsum device pipeline."""

    utility: DestinationUtilityTelemetry
    nested_logsum: object


@dataclass(frozen=True)
class LoweredDestinationUtility:
    """Validated coefficient matrix for a fully lowered utility specification.

    ``coefficients[f, a]`` multiplies feature ``f`` for alternative ``a``;
    ``constants[a]`` is added once per row.  Feature and alternative names are
    part of the object so an upstream compiler cannot silently change a column
    order and obtain plausible but wrong choices.
    """

    feature_names: tuple[str, ...]
    alternative_names: tuple[str, ...]
    coefficients: np.ndarray
    constants: np.ndarray
    compute_dtype: str = "float64"

    def __post_init__(self):
        features = tuple(self.feature_names)
        alternatives = tuple(self.alternative_names)
        try:
            dtype = np.dtype(self.compute_dtype)
        except TypeError as err:
            raise ValueError("compute_dtype must be float32 or float64") from err
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("compute_dtype must be float32 or float64")
        coefficients = np.ascontiguousarray(self.coefficients, dtype=dtype)
        constants = np.ascontiguousarray(self.constants, dtype=dtype)
        if not features or len(set(features)) != len(features):
            raise ValueError("feature_names must be unique and nonempty")
        if not alternatives or len(set(alternatives)) != len(alternatives):
            raise ValueError("alternative_names must be unique and nonempty")
        if coefficients.shape != (len(features), len(alternatives)):
            raise ValueError("coefficients must have shape (features, alternatives)")
        if constants.shape != (len(alternatives),):
            raise ValueError("constants must have one value per alternative")
        if not np.isfinite(coefficients).all() or not np.isfinite(constants).all():
            raise ValueError("lowered utility coefficients must be finite")
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "alternative_names", alternatives)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "compute_dtype", dtype.name)

    def cpu_reference(self, features, *, ordered=False) -> np.ndarray:
        """Evaluate exactly the declared dense linear ABI on the CPU."""
        values = _host_features(features, len(self.feature_names), np.dtype(self.compute_dtype))
        if not ordered:
            return values @ self.coefficients + self.constants
        result = np.broadcast_to(self.constants, (len(values), len(self.alternative_names))).copy()
        for feature in range(len(self.feature_names)):
            result += values[:, feature:feature + 1] * self.coefficients[feature]
        return result

    def cuda(self, features, *, return_device=False, return_telemetry=False, ordered=False):
        """Evaluate utilities using float64 GEMM and retain them on the GPU.

        A CUDA-array input (for example CuPy) incurs no host-to-device copy.
        ``return_device=True`` is the intended route into
        :func:`choiceforge.nested_logit.mtc21_nested_logsums_cuda`.
        """
        from .cuda_backend import _cupy

        cp = _cupy()
        dtype = cp.float32 if self.compute_dtype == "float32" else cp.float64
        device_input = hasattr(features, "__cuda_array_interface__")
        if device_input:
            device_features = cp.ascontiguousarray(features, dtype=dtype)
            _validate_shape(device_features.shape, len(self.feature_names))
            input_bytes = int(device_features.nbytes)
        else:
            host_features = _host_features(features, len(self.feature_names), np.dtype(self.compute_dtype))
            input_bytes = int(host_features.nbytes)

        start = time.perf_counter()
        if not device_input:
            device_features = cp.asarray(host_features)
            cp.cuda.Stream.null.synchronize()
            after_upload = time.perf_counter()
        else:
            after_upload = start
        coefficients = cp.asarray(self.coefficients)
        constants = cp.asarray(self.constants)
        if ordered:
            if self.compute_dtype != "float32":
                raise ValueError("ordered CUDA utility mode requires float32")
            global _ORDERED_F32_KERNEL
            if _ORDERED_F32_KERNEL is None:
                _ORDERED_F32_KERNEL = cp.RawKernel(
                    _ORDERED_F32_SOURCE, "ordered_linear_f32", options=("--fmad=false",)
                )
            utilities = cp.empty(
                (device_features.shape[0], len(self.alternative_names)), dtype=cp.float32
            )
            total = int(device_features.shape[0]) * len(self.alternative_names)
            threads = 256
            _ORDERED_F32_KERNEL(
                ((total + threads - 1) // threads,), (threads,),
                (device_features, coefficients, constants, utilities,
                 np.int64(device_features.shape[0]), np.int32(len(self.feature_names)),
                 np.int32(len(self.alternative_names))),
            )
        else:
            utilities = device_features @ coefficients + constants
        cp.cuda.Stream.null.synchronize()
        after_kernel = time.perf_counter()
        result = utilities if return_device else cp.asnumpy(utilities)
        after_download = time.perf_counter()
        telemetry = DestinationUtilityTelemetry(
            rows=int(device_features.shape[0]),
            features=len(self.feature_names),
            alternatives=len(self.alternative_names),
            input_bytes=input_bytes,
            host_to_device_ms=(after_upload - start) * 1000,
            kernel_ms=(after_kernel - after_upload) * 1000,
            device_to_host_ms=0.0 if return_device else (after_download - after_kernel) * 1000,
        )
        return (result, telemetry) if return_telemetry else result


def _validate_shape(shape, features: int) -> None:
    if len(shape) != 2 or shape[1] != features:
        raise ValueError(f"expected a row-by-{features} feature matrix")


def _host_features(features, expected_features: int, dtype=np.float64) -> np.ndarray:
    values = np.ascontiguousarray(features, dtype=dtype)
    _validate_shape(values.shape, expected_features)
    if not np.isfinite(values).all():
        raise ValueError("lowered utility features must be finite")
    return values


def mtc21_logsums_from_lowered_cuda(model, features, nest_spec, *, return_telemetry=False):
    """Run a lowered MTC-21 utility and its nested reduction without an
    intermediate device-to-host round trip.

    This is the production-shaped Phase 12 boundary.  It is deliberately
    separate from ActivitySim until a model-specific expression lowerer has
    passed its CPU equivalence gate.
    """
    from .nested_logit import mtc21_nested_logsums_cuda

    utilities, utility_telemetry = model.cuda(
        features, return_device=True, return_telemetry=True
    )
    logsums, nested_telemetry = mtc21_nested_logsums_cuda(
        utilities, nest_spec, model.alternative_names, return_telemetry=True
    )
    if not return_telemetry:
        return logsums
    return logsums, DestinationLogsumTelemetry(utility_telemetry, nested_telemetry)
