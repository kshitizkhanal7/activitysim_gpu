"""CuPy RawKernel backend for fixed-alternative MNL choice.

The kernels use one CUDA block per chooser. They target the common ActivitySim
case with at most 1,024 fixed alternatives. The fused-linear kernel keeps the
utility row and exponential weights in shared memory; it never materializes an
``N x A`` utility matrix in device global memory.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .api import ChoiceResult

CUDA_SOURCE = r"""
extern "C" __global__
void choose_f32(
    const float* utilities,
    const unsigned char* availability,
    const float* uniforms,
    int n_rows,
    int n_alts,
    int use_availability,
    int* choices,
    float* logsums)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float shared[];
    float* values = shared;
    float* scratch = shared + blockDim.x;

    const float neg_inf = -__int_as_float(0x7f800000);
    float utility = neg_inf;
    if (lane < n_alts) {
        const int idx = row * n_alts + lane;
        const bool available = !use_availability || availability[idx] != 0;
        const float candidate = utilities[idx];
        if (available && isfinite(candidate)) utility = candidate;
    }
    values[lane] = utility;
    scratch[lane] = utility;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    // All warps must capture the reduced maximum before any lane reuses the
    // scratch buffer for weights. A single warp executes in lockstep and hid
    // this race in the original 32-alternative tests.
    __syncthreads();
    if (!isfinite(row_max)) {
        if (lane == 0) { choices[row] = -1; logsums[row] = neg_inf; }
        return;
    }

    const float weight = lane < n_alts && isfinite(values[lane])
        ? expf(values[lane] - row_max) : 0.0f;
    values[lane] = weight;
    scratch[lane] = weight;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }

    if (lane == 0) {
        const float total = scratch[0];
        const float threshold = uniforms[row] * total;
        float cumulative = 0.0f;
        int selected = -1;
        int last_positive = -1;
        for (int alt = 0; alt < n_alts; ++alt) {
            const float w = values[alt];
            if (w > 0.0f) last_positive = alt;
            cumulative += w;
            if (selected < 0 && cumulative >= threshold) selected = alt;
        }
        if (selected < 0) {
            float max_weight = -1.0f;
            for (int alt = 0; alt < n_alts; ++alt) {
                if (values[alt] > max_weight) { max_weight = values[alt]; selected = alt; }
            }
        }
        choices[row] = selected;
        logsums[row] = row_max + logf(total);
    }
}

extern "C" __global__
void linear_choice_f32(
    const float* chooser_features,
    const float* coefficients,
    const float* constants,
    const unsigned char* availability,
    const float* uniforms,
    int n_rows,
    int n_alts,
    int n_features,
    int use_availability,
    int* choices,
    float* logsums)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float shared[];
    float* values = shared;
    float* scratch = shared + blockDim.x;

    const float neg_inf = -__int_as_float(0x7f800000);
    float utility = neg_inf;
    if (lane < n_alts) {
        const int availability_idx = row * n_alts + lane;
        const bool available = !use_availability || availability[availability_idx] != 0;
        if (available) {
            float acc = constants[lane];
            const int x_offset = row * n_features;
            const int beta_offset = lane * n_features;
            for (int feature = 0; feature < n_features; ++feature) {
                acc = fmaf(
                    chooser_features[x_offset + feature],
                    coefficients[beta_offset + feature],
                    acc);
            }
            if (isfinite(acc)) utility = acc;
        }
    }
    values[lane] = utility;
    scratch[lane] = utility;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    // Prevent lanes in one warp from overwriting scratch[0] before lanes in
    // another warp have loaded the reduced maximum.
    __syncthreads();
    if (!isfinite(row_max)) {
        if (lane == 0) { choices[row] = -1; logsums[row] = neg_inf; }
        return;
    }

    const float weight = lane < n_alts && isfinite(values[lane])
        ? expf(values[lane] - row_max) : 0.0f;
    values[lane] = weight;
    scratch[lane] = weight;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }

    if (lane == 0) {
        const float total = scratch[0];
        const float threshold = uniforms[row] * total;
        float cumulative = 0.0f;
        int selected = -1;
        int last_positive = -1;
        for (int alt = 0; alt < n_alts; ++alt) {
            const float w = values[alt];
            if (w > 0.0f) last_positive = alt;
            cumulative += w;
            if (selected < 0 && cumulative >= threshold) selected = alt;
        }
        if (selected < 0) {
            float max_weight = -1.0f;
            for (int alt = 0; alt < n_alts; ++alt) {
                if (values[alt] > max_weight) { max_weight = values[alt]; selected = alt; }
            }
        }
        choices[row] = selected;
        logsums[row] = row_max + logf(total);
    }
}

// ActivitySim's choice_maker equivalent. One thread owns a chooser so floating-
// point subtraction and alternative traversal have exactly the same order as
// the Numba CPU routine. Probabilities and random draws remain float64.
extern "C" __global__
void choose_probabilities_f64(
    const double* probabilities,
    const double* uniforms,
    int n_rows,
    int n_alts,
    int* choices)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    double z = uniforms[row];
    int selected = -1;
    for (int alt = 0; alt < n_alts; ++alt) {
        z -= probabilities[row * n_alts + alt];
        if (z <= 0.0) { selected = alt; break; }
    }
    if (selected < 0) {
        double max_probability = 0.0;
        for (int alt = 0; alt < n_alts; ++alt) {
            const double p = probabilities[row * n_alts + alt];
            if (p > max_probability) { max_probability = p; selected = alt; }
        }
    }
    choices[row] = selected;
}
"""

_DLL_DIRECTORY_HANDLES: list[Any] = []


def _register_pip_cuda_dlls() -> None:
    """Register CUDA DLL directories installed by NVIDIA's pip packages.

    CuPy 14 discovers these packages itself. ActivitySim's current dependency
    constraints resolve to CuPy 13, which predates that Windows behavior, so we
    register the isolated environment's ``nvidia/*/bin`` directories explicitly.
    """

    if os.name != "nt" or _DLL_DIRECTORY_HANDLES:
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        bin_dirs = []
        for bin_dir in sorted(nvidia_root.glob("*/bin")):
            bin_dirs.append(str(bin_dir))
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(bin_dir)))
        os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])
        # CuPy 13 asks NVRTC to locate its builtins relative to CUDA_PATH. The
        # pip package uses a component root instead of a monolithic toolkit.
        nvrtc_root = nvidia_root / "cuda_nvrtc"
        if nvrtc_root.is_dir():
            os.environ.setdefault("CUDA_PATH", str(nvrtc_root))


def _cupy() -> Any:
    _register_pip_cuda_dlls()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "The CUDA backend requires CuPy. Install ChoiceForge with the 'gpu' extra."
        ) from exc
    return cp


def cuda_available() -> bool:
    """Return whether CuPy can see at least one CUDA device."""

    try:
        cp = _cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _kernels() -> tuple[Any, Any, Any]:
    cp = _cupy()
    module = cp.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++11",),
        name_expressions=("choose_f32", "linear_choice_f32", "choose_probabilities_f64"),
    )
    return (
        module.get_function("choose_f32"),
        module.get_function("linear_choice_f32"),
        module.get_function("choose_probabilities_f64"),
    )


def _threads_for(n_alts: int) -> int:
    if not 1 <= n_alts <= 1024:
        raise ValueError("the fixed-alternative CUDA kernel supports 1 to 1,024 alternatives")
    return max(32, 1 << (n_alts - 1).bit_length())


def _availability(cp: Any, availability: Any | None, shape: tuple[int, int]) -> tuple[Any, np.int32]:
    if availability is None:
        return cp.empty(1, dtype=cp.uint8), np.int32(0)
    arr = cp.ascontiguousarray(cp.asarray(availability, dtype=cp.uint8))
    if arr.shape != shape:
        raise ValueError(f"availability must have shape {shape}, got {arr.shape}")
    return arr, np.int32(1)


class CudaChoiceBackend:
    """Compiled CUDA implementation with explicit host/device result control."""

    def __init__(self) -> None:
        if not cuda_available():
            raise RuntimeError("no CUDA device is available through CuPy")

    def choose_from_utilities(
        self,
        utilities: Any,
        uniforms: Any,
        availability: Any | None = None,
        *,
        return_device: bool = False,
    ) -> ChoiceResult[Any]:
        cp = _cupy()
        u = cp.ascontiguousarray(cp.asarray(utilities, dtype=cp.float32))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float32))
        if u.ndim != 2:
            raise ValueError("utilities must be two-dimensional")
        n_rows, n_alts = map(int, u.shape)
        if draws.shape != (n_rows,):
            raise ValueError(f"uniforms must have shape {(n_rows,)}, got {draws.shape}")
        if bool(cp.any((draws < 0) | (draws >= 1) | ~cp.isfinite(draws)).item()):
            raise ValueError("uniforms must be finite values in [0, 1)")
        avail, use_avail = _availability(cp, availability, (n_rows, n_alts))
        choices = cp.empty(n_rows, dtype=cp.int32)
        logsums = cp.empty(n_rows, dtype=cp.float32)
        threads = _threads_for(n_alts)
        choose_kernel, _, _ = _kernels()
        choose_kernel(
            (n_rows,),
            (threads,),
            (u, avail, draws, np.int32(n_rows), np.int32(n_alts), use_avail, choices, logsums),
            shared_mem=2 * threads * np.dtype(np.float32).itemsize,
        )
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))

    def linear_choice(
        self,
        chooser_features: Any,
        coefficients: Any,
        constants: Any,
        uniforms: Any,
        availability: Any | None = None,
        *,
        return_device: bool = False,
    ) -> ChoiceResult[Any]:
        cp = _cupy()
        x = cp.ascontiguousarray(cp.asarray(chooser_features, dtype=cp.float32))
        beta = cp.ascontiguousarray(cp.asarray(coefficients, dtype=cp.float32))
        asc = cp.ascontiguousarray(cp.asarray(constants, dtype=cp.float32))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float32))
        if x.ndim != 2 or beta.ndim != 2 or x.shape[1] != beta.shape[1]:
            raise ValueError("chooser_features and coefficients require matching feature dimensions")
        n_rows, n_features = map(int, x.shape)
        n_alts = int(beta.shape[0])
        if asc.shape != (n_alts,) or draws.shape != (n_rows,):
            raise ValueError("constants or uniforms have incompatible shapes")
        if bool(cp.any((draws < 0) | (draws >= 1) | ~cp.isfinite(draws)).item()):
            raise ValueError("uniforms must be finite values in [0, 1)")
        avail, use_avail = _availability(cp, availability, (n_rows, n_alts))
        choices = cp.empty(n_rows, dtype=cp.int32)
        logsums = cp.empty(n_rows, dtype=cp.float32)
        threads = _threads_for(n_alts)
        _, linear_kernel, _ = _kernels()
        linear_kernel(
            (n_rows,),
            (threads,),
            (
                x,
                beta,
                asc,
                avail,
                draws,
                np.int32(n_rows),
                np.int32(n_alts),
                np.int32(n_features),
                use_avail,
                choices,
                logsums,
            ),
            shared_mem=2 * threads * np.dtype(np.float32).itemsize,
        )
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))

    def choose_from_probabilities(
        self,
        probabilities: Any,
        uniforms: Any,
        *,
        return_device: bool = False,
    ) -> Any:
        """Match ActivitySim ``choice_maker`` on float64 probability rows."""

        cp = _cupy()
        probs = cp.ascontiguousarray(cp.asarray(probabilities, dtype=cp.float64))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float64).reshape(-1))
        if probs.ndim != 2 or draws.shape != (probs.shape[0],):
            raise ValueError("probabilities must be (N,A) and uniforms must contain N values")
        n_rows, n_alts = map(int, probs.shape)
        choices = cp.empty(n_rows, dtype=cp.int32)
        _, _, kernel = _kernels()
        threads = 256
        blocks = (n_rows + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (probs, draws, np.int32(n_rows), np.int32(n_alts), choices),
        )
        return choices if return_device else cp.asnumpy(choices)
