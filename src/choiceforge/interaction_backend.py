"""Fused ragged interaction-choice kernels for ActivitySim replay data."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .api import ChoiceResult
from .cuda_backend import _cupy, _register_pip_cuda_dlls

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


CUDA_SOURCE = r"""
extern "C" __global__
void interaction_terms_choice_f32(
    const float* terms, const float* coefficients, const long long* offsets,
    const float* uniforms, int n_choosers, int n_terms,
    int* choices, float* logsums)
{
    const int chooser = blockIdx.x;
    const int lane = threadIdx.x;
    if (chooser >= n_choosers) return;
    const long long begin = offsets[chooser];
    const int count = (int)(offsets[chooser + 1] - begin);
    extern __shared__ float shared[];
    float* values = shared;
    float* scratch = shared + blockDim.x;
    const float neg_inf = -__int_as_float(0x7f800000);
    float utility = neg_inf;
    if (lane < count) {
        const long long base = (begin + lane) * n_terms;
        float acc = 0.0f;
        for (int term = 0; term < n_terms; ++term)
            acc = fmaf(terms[base + term], coefficients[term], acc);
        utility = acc;
    }
    values[lane] = utility;
    scratch[lane] = utility;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();
    const float weight = lane < count ? expf(values[lane] - row_max) : 0.0f;
    values[lane] = weight;
    scratch[lane] = weight;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        const float total = scratch[0];
        const float threshold = uniforms[chooser] * total;
        float cumulative = 0.0f;
        int selected = -1;
        for (int alt = 0; alt < count; ++alt) {
            cumulative += values[alt];
            if (selected < 0 && cumulative >= threshold) selected = alt;
        }
        if (selected < 0) selected = count - 1;
        choices[chooser] = selected;
        logsums[chooser] = row_max + logf(total);
    }
}

extern "C" __global__
void interaction_batched_terms_choice_f32(
    const float* terms, const float* coefficients, const long long* offsets,
    const int* chooser_batches, const float* uniforms,
    int n_choosers, int n_terms, int n_batches,
    int* choices, float* logsums)
{
    const int chooser = blockIdx.x;
    const int lane = threadIdx.x;
    if (chooser >= n_choosers) return;
    const long long begin = offsets[chooser];
    const int count = (int)(offsets[chooser + 1] - begin);
    const int batch = chooser_batches[chooser];
    if (batch < 0 || batch >= n_batches) return;
    extern __shared__ float shared[];
    float* values = shared;
    float* scratch = shared + blockDim.x;
    const float neg_inf = -__int_as_float(0x7f800000);
    float utility = neg_inf;
    if (lane < count) {
        const long long base = (begin + lane) * n_terms;
        const int beta_base = batch * n_terms;
        float acc = 0.0f;
        for (int term = 0; term < n_terms; ++term)
            acc = fmaf(terms[base + term], coefficients[beta_base + term], acc);
        utility = acc;
    }
    values[lane] = utility;
    scratch[lane] = utility;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();
    const float weight = lane < count ? expf(values[lane] - row_max) : 0.0f;
    values[lane] = weight;
    scratch[lane] = weight;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        const float total = scratch[0];
        const float threshold = uniforms[chooser] * total;
        float cumulative = 0.0f;
        int selected = -1;
        for (int alt = 0; alt < count; ++alt) {
            cumulative += values[alt];
            if (selected < 0 && cumulative >= threshold) selected = alt;
        }
        if (selected < 0) selected = count - 1;
        choices[chooser] = selected;
        logsums[chooser] = row_max + logf(total);
    }
}
"""


def offsets_from_ids(chooser_ids: Any) -> np.ndarray:
    """Return CSR-style offsets for contiguous ActivitySim interaction rows."""
    ids = np.asarray(chooser_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("chooser_ids must be a non-empty one-dimensional array")
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return np.ascontiguousarray(np.r_[starts, ids.size], dtype=np.int64)


@lru_cache(maxsize=1)
def _kernel():
    cp = _cupy()
    return cp.RawKernel(CUDA_SOURCE, "interaction_terms_choice_f32", options=("--std=c++11",))


@lru_cache(maxsize=1)
def _batched_kernel():
    cp = _cupy()
    return cp.RawKernel(
        CUDA_SOURCE,
        "interaction_batched_terms_choice_f32",
        options=("--std=c++11",),
    )


class CudaInteractionBackend:
    """CUDA backend for a lowered term matrix with ragged alternative sets."""

    def __init__(self) -> None:
        _register_pip_cuda_dlls()
        cp = _cupy()
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device is available through CuPy")

    def choose_from_terms(
        self,
        terms: Any,
        coefficients: Any,
        offsets: Any,
        uniforms: Any,
        *,
        return_device: bool = False,
    ) -> ChoiceResult[Any]:
        cp = _cupy()
        x = cp.ascontiguousarray(cp.asarray(terms, dtype=cp.float32))
        beta = cp.ascontiguousarray(cp.asarray(coefficients, dtype=cp.float32))
        ptr = cp.ascontiguousarray(cp.asarray(offsets, dtype=cp.int64))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float32))
        if x.ndim != 2 or beta.shape != (x.shape[1],):
            raise ValueError("terms must be 2D and coefficients must match its columns")
        n = int(ptr.size - 1)
        if draws.shape != (n,):
            raise ValueError("uniforms length must match the number of choosers")
        counts = ptr[1:] - ptr[:-1]
        max_count = int(cp.max(counts).item())
        if max_count > 1024 or int(ptr[0].item()) != 0 or int(ptr[-1].item()) != x.shape[0]:
            raise ValueError("invalid offsets or more than 1,024 alternatives")
        threads = max(32, 1 << (max_count - 1).bit_length())
        choices = cp.empty(n, dtype=cp.int32)
        logsums = cp.empty(n, dtype=cp.float32)
        _kernel()(
            (n,), (threads,),
            (x, beta, ptr, draws, np.int32(n), np.int32(x.shape[1]), choices, logsums),
            shared_mem=2 * threads * np.dtype(np.float32).itemsize,
        )
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))

    def choose_from_batched_terms(
        self,
        terms: Any,
        coefficients: Any,
        offsets: Any,
        chooser_batches: Any,
        uniforms: Any,
        *,
        return_device: bool = False,
    ) -> ChoiceResult[Any]:
        """Evaluate many coefficient segments in one transfer and CUDA launch."""
        cp = _cupy()
        x = cp.ascontiguousarray(cp.asarray(terms, dtype=cp.float32))
        beta = cp.ascontiguousarray(cp.asarray(coefficients, dtype=cp.float32))
        ptr = cp.ascontiguousarray(cp.asarray(offsets, dtype=cp.int64))
        segments = cp.ascontiguousarray(cp.asarray(chooser_batches, dtype=cp.int32))
        draws = cp.ascontiguousarray(cp.asarray(uniforms, dtype=cp.float32))
        if x.ndim != 2 or beta.ndim != 2 or beta.shape[1] != x.shape[1]:
            raise ValueError("terms and coefficients must be compatible 2D arrays")
        n = int(ptr.size - 1)
        if segments.shape != (n,) or draws.shape != (n,):
            raise ValueError("segment and uniform lengths must match choosers")
        counts = ptr[1:] - ptr[:-1]
        max_count = int(cp.max(counts).item())
        if max_count > 1024 or int(ptr[0].item()) != 0 or int(ptr[-1].item()) != x.shape[0]:
            raise ValueError("invalid offsets or more than 1,024 alternatives")
        if int(cp.min(segments).item()) < 0 or int(cp.max(segments).item()) >= beta.shape[0]:
            raise ValueError("chooser segment is outside the coefficient table")
        threads = max(32, 1 << (max_count - 1).bit_length())
        choices = cp.empty(n, dtype=cp.int32)
        logsums = cp.empty(n, dtype=cp.float32)
        _batched_kernel()(
            (n,),
            (threads,),
            (
                x,
                beta,
                ptr,
                segments,
                draws,
                np.int32(n),
                np.int32(x.shape[1]),
                np.int32(beta.shape[0]),
                choices,
                logsums,
            ),
            shared_mem=2 * threads * np.dtype(np.float32).itemsize,
        )
        if return_device:
            return ChoiceResult(choices, logsums)
        return ChoiceResult(cp.asnumpy(choices), cp.asnumpy(logsums))


if njit is not None:

    @njit(cache=True)
    def _choose_ragged_utilities(utilities, offsets, draws):
        n = offsets.size - 1
        choices = np.empty(n, dtype=np.int32)
        logsums = np.empty(n, dtype=np.float64)
        for chooser in range(n):
            begin = offsets[chooser]
            end = offsets[chooser + 1]
            row_max = -np.inf
            for row in range(begin, end):
                row_max = max(row_max, utilities[row])
            total = 0.0
            for row in range(begin, end):
                total += np.exp(utilities[row] - row_max)
            threshold = draws[chooser] * total
            cumulative = 0.0
            selected = end - begin - 1
            for row in range(begin, end):
                cumulative += np.exp(utilities[row] - row_max)
                if cumulative >= threshold:
                    selected = row - begin
                    break
            choices[chooser] = selected
            logsums[chooser] = row_max + np.log(total)
        return choices, logsums

    @njit(cache=True)
    def _choose_batched_terms(terms, coefficients, offsets, segments, draws):
        n = offsets.size - 1
        choices = np.empty(n, dtype=np.int32)
        logsums = np.empty(n, dtype=np.float64)
        for chooser in range(n):
            begin = offsets[chooser]
            end = offsets[chooser + 1]
            segment = segments[chooser]
            row_max = -np.inf
            for row in range(begin, end):
                utility = 0.0
                for term in range(terms.shape[1]):
                    utility += terms[row, term] * coefficients[segment, term]
                row_max = max(row_max, utility)
            total = 0.0
            for row in range(begin, end):
                utility = 0.0
                for term in range(terms.shape[1]):
                    utility += terms[row, term] * coefficients[segment, term]
                total += np.exp(utility - row_max)
            threshold = draws[chooser] * total
            cumulative = 0.0
            selected = end - begin - 1
            for row in range(begin, end):
                utility = 0.0
                for term in range(terms.shape[1]):
                    utility += terms[row, term] * coefficients[segment, term]
                cumulative += np.exp(utility - row_max)
                if cumulative >= threshold:
                    selected = row - begin
                    break
            choices[chooser] = selected
            logsums[chooser] = row_max + np.log(total)
        return choices, logsums


def choose_terms_numpy(terms, coefficients, offsets, uniforms) -> ChoiceResult[np.ndarray]:
    """Strong CPU baseline: one BLAS aggregation plus a compiled ragged choice."""
    if njit is None:
        raise RuntimeError("the CPU interaction baseline requires Numba")
    x = np.ascontiguousarray(terms, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float32)
    ptr = np.ascontiguousarray(offsets, dtype=np.int64)
    draws = np.ascontiguousarray(uniforms, dtype=np.float64)
    utilities = np.asarray(x @ beta, dtype=np.float64)
    choices, logsums = _choose_ragged_utilities(utilities, ptr, draws)
    return ChoiceResult(choices, logsums)


def choose_batched_terms_numpy(
    terms, coefficients, offsets, chooser_batches, uniforms
) -> ChoiceResult[np.ndarray]:
    """Compiled CPU comparator for the single-launch segmented CUDA path."""
    if njit is None:
        raise RuntimeError("the CPU interaction baseline requires Numba")
    x = np.ascontiguousarray(terms, dtype=np.float32)
    beta = np.ascontiguousarray(coefficients, dtype=np.float32)
    ptr = np.ascontiguousarray(offsets, dtype=np.int64)
    segments = np.ascontiguousarray(chooser_batches, dtype=np.int32)
    draws = np.ascontiguousarray(uniforms, dtype=np.float64)
    choices, logsums = _choose_batched_terms(x, beta, ptr, segments, draws)
    return ChoiceResult(choices, logsums)
