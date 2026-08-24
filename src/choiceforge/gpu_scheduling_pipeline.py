"""GPU-resident preparation for sequential mandatory-tour scheduling.

ActivitySim schedules mandatory tours in six ordered batches.  For each batch
it expands every tour against the 190 departure/duration alternatives, removes
alternatives that collide with the person's timetable, evaluates seven
timetable expressions, and then performs the choice.  Phase 20 began after
those preparation steps.  This module moves them onto the device.

The one deliberately compact upstream input is ``mode_logsum_cache``.  MTC's
190 hourly TDD alternatives map to only fifteen valid pairs of five network
skim periods.  The cache is stored as a dense 5-by-5 array per chooser; invalid
pairs are unused.  It is the stable hand-off from the separately qualified
GPU mode-choice/logsum engine, not a 15-million-row scheduling replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .cuda_backend import _cupy
from .gpu_native import GpuOnlyViolation, _is_cuda_array

try:
    import numba
except ImportError:  # pragma: no cover - optional CPU benchmark dependency
    numba = None


I_EMPTY = np.int8(0)
I_START = np.int8(2)
I_END = np.int8(4)
I_START_END = np.int8(6)
I_MIDDLE = np.int8(7)
N_SKIM_PERIODS = 5


def skim_period_code(hours: Any) -> np.ndarray:
    """Return Prototype MTC's five skim-period codes for integer model hours."""

    value = np.asarray(hours)
    return np.where(
        value <= 5,
        0,
        np.where(value <= 9, 1, np.where(value <= 14, 2, np.where(value <= 18, 3, 4))),
    ).astype(np.int8)


def mode_logsum_slots(alternative_values: Any, alternative_ids: Any) -> np.ndarray:
    """Map TDD ids to their ordered 5-by-5 outbound/inbound skim slot."""

    alternatives = np.asarray(alternative_values)
    ids = np.asarray(alternative_ids, dtype=np.int64)
    outbound = skim_period_code(alternatives[ids, 0])
    inbound = skim_period_code(alternatives[ids, 1])
    return (outbound * N_SKIM_PERIODS + inbound).astype(np.int16)


def compress_mode_choice_logsums(
    offsets: Any,
    alternative_ids: Any,
    alternative_values: Any,
    row_values: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor repeated row logsums into exact per-chooser skim-period caches.

    Every repeated value is checked bit-for-bit.  A capture that violates the
    public model's 5-by-5 factorization fails closed instead of silently taking
    the first value.
    """

    ptr = np.asarray(offsets, dtype=np.int64)
    ids = np.asarray(alternative_ids, dtype=np.int16)
    rows = np.asarray(row_values, dtype=np.float32)
    if ptr.ndim != 1 or ptr.size < 2 or ptr[0] != 0 or ptr[-1] != ids.size:
        raise ValueError("invalid scheduling CSR arrays")
    if rows.ndim != 2 or rows.shape[0] != ids.size or rows.shape[1] < 1:
        raise ValueError("row_values must contain mode_choice_logsum in column zero")

    slots = mode_logsum_slots(alternative_values, ids)
    owners = np.repeat(np.arange(ptr.size - 1, dtype=np.int64), np.diff(ptr))
    cache = np.zeros((ptr.size - 1, N_SKIM_PERIODS**2), dtype=np.float32)
    present = np.zeros(cache.shape, dtype=np.bool_)
    logsums = rows[:, 0]
    for row in range(ids.size):
        owner = owners[row]
        slot = slots[row]
        if present[owner, slot]:
            if cache[owner, slot].view(np.uint32) != logsums[row].view(np.uint32):
                raise ValueError(
                    f"mode-choice logsum is not constant within skim slot {int(slot)}"
                )
        else:
            cache[owner, slot] = logsums[row]
            present[owner, slot] = True
    return cache, present


def build_tdd_footprints(alternative_values: Any) -> tuple[np.ndarray, int]:
    """Build ActivitySim-compatible timetable state footprints for every TDD."""

    alternatives = np.asarray(alternative_values)
    if alternatives.ndim != 2 or alternatives.shape[1] < 2:
        raise ValueError("alternative_values must have start and end columns")
    starts = alternatives[:, 0].astype(np.int32)
    ends = alternatives[:, 1].astype(np.int32)
    if np.any(ends < starts):
        raise ValueError("TDD end cannot precede start")
    minimum = int(starts.min())
    maximum = int(ends.max())
    first_period = minimum - 1
    footprints = np.zeros((len(alternatives), maximum - minimum + 3), dtype=np.int8)
    for tdd, (start, end) in enumerate(zip(starts, ends)):
        start_column = int(start) - first_period
        end_column = int(end) - first_period
        if start_column == end_column:
            footprints[tdd, start_column] = I_START_END
        else:
            footprints[tdd, start_column] = I_START
            footprints[tdd, end_column] = I_END
            footprints[tdd, start_column + 1 : end_column] = I_MIDDLE
    return footprints, first_period


def _collides(candidate: np.ndarray, window: np.ndarray) -> bool:
    return bool(
        np.any(
            ((candidate == I_START) & ((window == I_START) | (window == I_MIDDLE)))
            | ((candidate == I_END) & ((window == I_END) | (window == I_MIDDLE)))
            | (
                (candidate == I_MIDDLE)
                & (
                    (window == I_START)
                    | (window == I_END)
                    | (window == I_START_END)
                    | (window == I_MIDDLE)
                )
            )
            | ((candidate == I_START_END) & (window == I_MIDDLE))
        )
    )


@dataclass(frozen=True)
class PreparedSchedulingBatch:
    chooser_values: Any
    row_values: Any
    alternative_ids: Any
    offsets: Any
    row_owners: Any

    @property
    def interaction_rows(self) -> int:
        return int(self.alternative_ids.shape[0])


class CpuSchedulingPreparer:
    """Independent, readable ActivitySim timetable/preparation reference."""

    def __init__(self, person_count: int, alternative_values: Any):
        alternatives = np.asarray(alternative_values, dtype=np.float32)
        self.alternative_values = alternatives
        self.footprints, self.first_period = build_tdd_footprints(alternatives)
        self.windows = np.zeros((person_count, self.footprints.shape[1]), dtype=np.int8)
        self.previous_tdd = np.zeros(person_count, dtype=np.int16)

    def reset(self) -> None:
        self.windows.fill(I_EMPTY)
        self.previous_tdd.fill(0)

    def prepare(
        self,
        person_rows: Any,
        chooser_values: Any,
        mode_logsum_cache: Any,
        *,
        end_previous_column: int,
        tour_count_column: int,
        tour_num_column: int,
    ) -> PreparedSchedulingBatch:
        people = np.asarray(person_rows, dtype=np.int32)
        chooser = np.ascontiguousarray(chooser_values, dtype=np.float32).copy()
        cache = np.asarray(mode_logsum_cache, dtype=np.float32)
        if chooser.shape[0] != people.size or cache.shape != (people.size, 25):
            raise ValueError("CPU scheduling preparation inputs have incompatible shapes")

        chooser[:, end_previous_column] = self.alternative_values[
            self.previous_tdd[people], 1
        ]
        counts = np.empty(people.size, dtype=np.int64)
        feasible: list[np.ndarray] = []
        for row, person in enumerate(people):
            ids = np.fromiter(
                (
                    tdd
                    for tdd, footprint in enumerate(self.footprints)
                    if not _collides(footprint, self.windows[person])
                ),
                dtype=np.int16,
            )
            feasible.append(ids)
            counts[row] = ids.size
        offsets = np.empty(people.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        alternative_ids = np.concatenate(feasible)
        owners = np.repeat(np.arange(people.size, dtype=np.int32), counts)
        row_values = np.empty((alternative_ids.size, 8), dtype=np.float32)

        for row, (owner, tdd) in enumerate(zip(owners, alternative_ids)):
            person = people[owner]
            start = int(self.alternative_values[tdd, 0])
            end = int(self.alternative_values[tdd, 1])
            start_column = start - self.first_period
            end_column = end - self.first_period
            window = self.windows[person]
            slot = int(skim_period_code([start])[0]) * 5 + int(
                skim_period_code([end])[0]
            )
            adjacent_before = start_column > 1 and window[start_column - 1] != I_MIDDLE
            adjacent_after = (
                end_column < window.size - 2 and window[end_column + 1] != I_MIDDLE
            )
            tour_count = chooser[owner, tour_count_column]
            tour_num = chooser[owner, tour_num_column]
            available_count = int(np.count_nonzero(window[1:-1] != I_MIDDLE))
            remaining = available_count - max(end - start - 1, 0)
            row_values[row] = (
                cache[owner, slot],
                window[start_column] in (I_END, I_START_END),
                window[end_column] in (I_START, I_START_END),
                (tour_count > 1) and (tour_num == 1) and adjacent_before,
                (tour_count > 1) and (tour_num == 1) and adjacent_after,
                (tour_num > 1) and adjacent_before,
                (tour_num > 1) and adjacent_after,
                (np.float32(1.0) / np.float32(remaining))
                if (tour_count > 1 and tour_num == 1)
                else np.float32(0.0),
            )
        return PreparedSchedulingBatch(chooser, row_values, alternative_ids, offsets, owners)

    def assign(self, person_rows: Any, selected_tdds: Any) -> None:
        people = np.asarray(person_rows, dtype=np.int32)
        selected = np.asarray(selected_tdds, dtype=np.int16)
        if people.shape != selected.shape:
            raise ValueError("person rows and selected TDDs must have equal shape")
        self.windows[people] |= self.footprints[selected]
        self.previous_tdd[people] = selected


if numba is not None:
    @numba.njit(cache=True)
    def _collision_numba(footprint, window):
        for period in range(footprint.size):
            candidate = footprint[period]
            existing = window[period]
            if (
                (candidate == 2 and (existing == 2 or existing == 7))
                or (candidate == 4 and (existing == 4 or existing == 7))
                or (
                    candidate == 7
                    and (existing == 2 or existing == 4 or existing == 6 or existing == 7)
                )
                or (candidate == 6 and existing == 7)
            ):
                return True
        return False

    @numba.njit(parallel=True, cache=True)
    def _prepare_compiled_cpu(
        windows,
        previous_tdd,
        people,
        chooser_input,
        alternatives,
        footprints,
        cache,
        first_period,
        end_previous_column,
        tour_count_column,
        tour_num_column,
    ):
        n = people.size
        n_alternatives = alternatives.shape[0]
        chooser = chooser_input.copy()
        counts = np.zeros(n, dtype=np.int64)
        for owner in numba.prange(n):
            person = people[owner]
            chooser[owner, end_previous_column] = alternatives[previous_tdd[person], 1]
            for tdd in range(n_alternatives):
                if not _collision_numba(footprints[tdd], windows[person]):
                    counts[owner] += 1
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        for owner in range(n):
            offsets[owner + 1] = offsets[owner] + counts[owner]
        alternative_ids = np.empty(offsets[-1], dtype=np.int16)
        owners = np.empty(offsets[-1], dtype=np.int32)
        row_values = np.empty((offsets[-1], 8), dtype=np.float32)
        for owner in numba.prange(n):
            person = people[owner]
            window = windows[person]
            available_count = 0
            for period in range(1, window.size - 1):
                available_count += window[period] != 7
            output = offsets[owner]
            for tdd in range(n_alternatives):
                if _collision_numba(footprints[tdd], window):
                    continue
                alternative_ids[output] = tdd
                owners[output] = owner
                start = int(alternatives[tdd, 0])
                end = int(alternatives[tdd, 1])
                start_column = start - first_period
                end_column = end - first_period
                out_period = 0 if start <= 5 else (1 if start <= 9 else (2 if start <= 14 else (3 if start <= 18 else 4)))
                in_period = 0 if end <= 5 else (1 if end <= 9 else (2 if end <= 14 else (3 if end <= 18 else 4)))
                adjacent_before = start_column > 1 and window[start_column - 1] != 7
                adjacent_after = end_column < window.size - 2 and window[end_column + 1] != 7
                tour_count = chooser[owner, tour_count_column]
                tour_num = chooser[owner, tour_num_column]
                remaining = available_count - max(end - start - 1, 0)
                row_values[output, 0] = cache[owner, out_period * 5 + in_period]
                row_values[output, 1] = window[start_column] == 4 or window[start_column] == 6
                row_values[output, 2] = window[end_column] == 2 or window[end_column] == 6
                row_values[output, 3] = tour_count > 1 and tour_num == 1 and adjacent_before
                row_values[output, 4] = tour_count > 1 and tour_num == 1 and adjacent_after
                row_values[output, 5] = tour_num > 1 and adjacent_before
                row_values[output, 6] = tour_num > 1 and adjacent_after
                row_values[output, 7] = (
                    np.float32(1.0) / np.float32(remaining)
                    if tour_count > 1 and tour_num == 1
                    else np.float32(0.0)
                )
                output += 1
        return chooser, row_values, alternative_ids, offsets, owners

    @numba.njit(cache=True)
    def _assign_compiled_cpu(windows, previous_tdd, footprints, people, selected):
        for owner in range(people.size):
            person = people[owner]
            tdd = selected[owner]
            for period in range(windows.shape[1]):
                windows[person, period] |= footprints[tdd, period]
            previous_tdd[person] = tdd


class CompiledCpuSchedulingPreparer(CpuSchedulingPreparer):
    """Numba CPU baseline with the same primitive-input boundary as the GPU."""

    def __init__(self, person_count: int, alternative_values: Any):
        if numba is None:
            raise RuntimeError("Numba is required for the compiled CPU preparer")
        super().__init__(person_count, alternative_values)

    def prepare(
        self,
        person_rows: Any,
        chooser_values: Any,
        mode_logsum_cache: Any,
        *,
        end_previous_column: int,
        tour_count_column: int,
        tour_num_column: int,
    ) -> PreparedSchedulingBatch:
        values = _prepare_compiled_cpu(
            self.windows,
            self.previous_tdd,
            np.ascontiguousarray(person_rows, dtype=np.int32),
            np.ascontiguousarray(chooser_values, dtype=np.float32),
            self.alternative_values,
            self.footprints,
            np.ascontiguousarray(mode_logsum_cache, dtype=np.float32),
            self.first_period,
            end_previous_column,
            tour_count_column,
            tour_num_column,
        )
        return PreparedSchedulingBatch(*values)

    def assign(self, person_rows: Any, selected_tdds: Any) -> None:
        people = np.ascontiguousarray(person_rows, dtype=np.int32)
        selected = np.ascontiguousarray(selected_tdds, dtype=np.int16)
        if people.shape != selected.shape:
            raise ValueError("person rows and selected TDDs must have equal shape")
        _assign_compiled_cpu(
            self.windows, self.previous_tdd, self.footprints, people, selected
        )


_CUDA_SOURCE = r'''
extern "C" __global__ void feasible_mask(
 const signed char* windows, const signed char* footprints,
 const int* person_rows, int n_choosers, int n_alternatives, int n_periods,
 unsigned char* mask)
{
 const int flat = blockIdx.x * blockDim.x + threadIdx.x;
 const int total = n_choosers * n_alternatives;
 if (flat >= total) return;
 const int chooser = flat / n_alternatives;
 const int tdd = flat - chooser * n_alternatives;
 const signed char* window = windows + person_rows[chooser] * n_periods;
 const signed char* footprint = footprints + tdd * n_periods;
 bool available = true;
 for (int p=0; p<n_periods; ++p) {
   const int c = (int)footprint[p]; const int w = (int)window[p];
   const bool collision =
     (c == 2 && (w == 2 || w == 7)) ||
     (c == 4 && (w == 4 || w == 7)) ||
     (c == 7 && (w == 2 || w == 4 || w == 6 || w == 7)) ||
     (c == 6 && w == 7);
   if (collision) { available = false; break; }
 }
 mask[flat] = available ? 1 : 0;
}

extern "C" __global__ void prepare_rows(
 const signed char* windows, const short* previous_tdd,
 const int* person_rows, float* chooser_values, int chooser_width,
 const float* alternatives, const float* logsum_cache,
 const short* alternative_ids, const int* owners, long long n_rows,
 int n_periods, int first_period, int end_previous_column,
 int tour_count_column, int tour_num_column, float* row_values)
{
 const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= n_rows) return;
 const int owner = owners[row]; const int person = person_rows[owner];
 const int tdd = (int)alternative_ids[row];
 const int start = (int)alternatives[tdd * 3];
 const int end = (int)alternatives[tdd * 3 + 1];
 const int start_col = start - first_period; const int end_col = end - first_period;
 const signed char* window = windows + person * n_periods;
 const int out_period = start <= 5 ? 0 : (start <= 9 ? 1 : (start <= 14 ? 2 : (start <= 18 ? 3 : 4)));
 const int in_period = end <= 5 ? 0 : (end <= 9 ? 1 : (end <= 14 ? 2 : (end <= 18 ? 3 : 4)));
 const float tour_count = chooser_values[owner * chooser_width + tour_count_column];
 const float tour_num = chooser_values[owner * chooser_width + tour_num_column];
 const bool adjacent_before = start_col > 1 && window[start_col - 1] != 7;
 const bool adjacent_after = end_col < n_periods - 2 && window[end_col + 1] != 7;
 int available_count = 0;
 for (int p=1; p<n_periods-1; ++p) available_count += window[p] != 7;
 const int remaining = available_count - max(end - start - 1, 0);
 float* out = row_values + row * 8;
 out[0] = logsum_cache[owner * 25 + out_period * 5 + in_period];
 out[1] = window[start_col] == 4 || window[start_col] == 6;
 out[2] = window[end_col] == 2 || window[end_col] == 6;
 out[3] = tour_count > 1.0f && tour_num == 1.0f && adjacent_before;
 out[4] = tour_count > 1.0f && tour_num == 1.0f && adjacent_after;
 out[5] = tour_num > 1.0f && adjacent_before;
 out[6] = tour_num > 1.0f && adjacent_after;
 out[7] = tour_count > 1.0f && tour_num == 1.0f ? 1.0f / (float)remaining : 0.0f;
 if (row == 0 || owners[row - 1] != owner) {
   chooser_values[owner * chooser_width + end_previous_column] =
     alternatives[(int)previous_tdd[person] * 3 + 1];
 }
}

extern "C" __global__ void assign_windows(
 signed char* windows, short* previous_tdd, const signed char* footprints,
 const int* person_rows, const short* selected_tdds, int n, int n_periods)
{
 const int chooser = blockIdx.x * blockDim.x + threadIdx.x;
 if (chooser >= n) return;
 const int person = person_rows[chooser]; const int tdd = (int)selected_tdds[chooser];
 signed char* window = windows + person * n_periods;
 const signed char* footprint = footprints + tdd * n_periods;
 for (int p=0; p<n_periods; ++p) window[p] = window[p] | footprint[p];
 previous_tdd[person] = (short)tdd;
}
'''


@lru_cache(maxsize=1)
def _gpu_kernels():
    cp = _cupy()
    module = cp.RawModule(code=_CUDA_SOURCE, options=("--std=c++11",))
    return (
        module.get_function("feasible_mask"),
        module.get_function("prepare_rows"),
        module.get_function("assign_windows"),
    )


class GpuSchedulingPreparer:
    """Persistent device timetable and exact feasible-row generator."""

    def __init__(self, person_count: int, alternative_values: Any):
        if not _is_cuda_array(alternative_values):
            raise GpuOnlyViolation("scheduling alternatives must reside on the GPU")
        cp = _cupy()
        self.alternative_values = cp.ascontiguousarray(alternative_values, dtype=cp.float32)
        starts = self.alternative_values[:, 0].astype(cp.int32)
        ends = self.alternative_values[:, 1].astype(cp.int32)
        if bool(cp.any(ends < starts).item()):
            raise ValueError("TDD end cannot precede start")
        minimum = int(cp.min(starts).item())
        maximum = int(cp.max(ends).item())
        self.first_period = minimum - 1
        periods = cp.arange(self.first_period, maximum + 2, dtype=cp.int32)[None, :]
        start_grid = starts[:, None]
        end_grid = ends[:, None]
        self.footprints = cp.where(
            (start_grid == end_grid) & (periods == start_grid),
            I_START_END,
            cp.where(
                periods == start_grid,
                I_START,
                cp.where(
                    periods == end_grid,
                    I_END,
                    cp.where(
                        (periods > start_grid) & (periods < end_grid), I_MIDDLE, I_EMPTY
                    ),
                ),
            ),
        ).astype(cp.int8)
        self.windows = cp.zeros((person_count, self.footprints.shape[1]), dtype=cp.int8)
        self.previous_tdd = cp.zeros(person_count, dtype=cp.int16)
        self._feasible, self._prepare, self._assign = _gpu_kernels()

    def reset(self) -> None:
        self.windows.fill(I_EMPTY)
        self.previous_tdd.fill(0)

    def prepare(
        self,
        person_rows: Any,
        chooser_values: Any,
        mode_logsum_cache: Any,
        *,
        end_previous_column: int,
        tour_count_column: int,
        tour_num_column: int,
    ) -> PreparedSchedulingBatch:
        inputs = (person_rows, chooser_values, mode_logsum_cache)
        if any(not _is_cuda_array(value) for value in inputs):
            raise GpuOnlyViolation("all scheduling preparation inputs must reside on the GPU")
        cp = _cupy()
        people = cp.ascontiguousarray(person_rows, dtype=cp.int32)
        chooser = cp.ascontiguousarray(chooser_values, dtype=cp.float32).copy()
        cache = cp.ascontiguousarray(mode_logsum_cache, dtype=cp.float32)
        n = int(people.size)
        n_alternatives = int(self.alternative_values.shape[0])
        if chooser.shape[0] != n or cache.shape != (n, 25):
            raise ValueError("GPU scheduling preparation inputs have incompatible shapes")

        mask = cp.empty((n, n_alternatives), dtype=cp.uint8)
        total = n * n_alternatives
        threads = 256
        self._feasible(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                self.windows,
                self.footprints,
                people,
                np.int32(n),
                np.int32(n_alternatives),
                np.int32(self.windows.shape[1]),
                mask,
            ),
        )
        counts = mask.sum(axis=1, dtype=cp.int64)
        offsets = cp.empty(n + 1, dtype=cp.int64)
        offsets[0] = 0
        cp.cumsum(counts, out=offsets[1:])
        flat = cp.flatnonzero(mask.reshape(-1))
        alternative_ids = cp.ascontiguousarray(flat % n_alternatives, dtype=cp.int16)
        owners = cp.ascontiguousarray(flat // n_alternatives, dtype=cp.int32)
        row_values = cp.empty((flat.size, 8), dtype=cp.float32)
        if flat.size:
            self._prepare(
                ((int(flat.size) + threads - 1) // threads,),
                (threads,),
                (
                    self.windows,
                    self.previous_tdd,
                    people,
                    chooser,
                    np.int32(chooser.shape[1]),
                    self.alternative_values,
                    cache,
                    alternative_ids,
                    owners,
                    np.int64(flat.size),
                    np.int32(self.windows.shape[1]),
                    np.int32(self.first_period),
                    np.int32(end_previous_column),
                    np.int32(tour_count_column),
                    np.int32(tour_num_column),
                    row_values,
                ),
            )
        return PreparedSchedulingBatch(chooser, row_values, alternative_ids, offsets, owners)

    def assign(self, person_rows: Any, selected_tdds: Any) -> None:
        if not _is_cuda_array(person_rows) or not _is_cuda_array(selected_tdds):
            raise GpuOnlyViolation("timetable assignments must reside on the GPU")
        cp = _cupy()
        people = cp.ascontiguousarray(person_rows, dtype=cp.int32)
        selected = cp.ascontiguousarray(selected_tdds, dtype=cp.int16)
        if people.shape != selected.shape:
            raise ValueError("person rows and selected TDDs must have equal shape")
        threads = 256
        self._assign(
            ((int(people.size) + threads - 1) // threads,),
            (threads,),
            (
                self.windows,
                self.previous_tdd,
                self.footprints,
                people,
                selected,
                np.int32(people.size),
                np.int32(self.windows.shape[1]),
            ),
        )
