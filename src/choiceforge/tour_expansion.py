"""GPU-native mandatory-tour row construction for Prototype MTC Extended.

ActivitySim's mandatory-tour-frequency model chooses one of five alternatives,
then expands each person into zero, one, or two ordered tour rows.  The
functions here reproduce that variable-length postprocessor with numeric,
device-resident columns.  String/categorical decoding is intentionally an
egress concern; modeled work uses the stable codes documented below.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from .cuda_backend import _cupy
from .gpu_native import DeviceTable, GpuOnlyViolation, _is_cuda_array


# mandatory_tour_frequency.csv alternative order:
# work1, work2, school1, school2, work_and_school
WORK_COUNTS = np.asarray([1, 2, 0, 0, 1], dtype=np.int8)
SCHOOL_COUNTS = np.asarray([0, 0, 1, 2, 1], dtype=np.int8)

# canonical_ids.canonical_tours() for the checked public configuration contains
# 41 possible tour labels.  Alphabetical positions are school1=31, school2=32,
# work1=39, and work2=40.  The benchmark independently verifies these constants
# against every public tour_id rather than trusting the constants by assertion.
POSSIBLE_TOURS_COUNT = 41
CANONICAL_OFFSETS = np.asarray([39, 40, 31, 32], dtype=np.int64)

TOUR_TYPE_WORK = np.int8(0)
TOUR_TYPE_SCHOOL = np.int8(1)
TOUR_CATEGORY_MANDATORY = np.int8(0)

TOUR_COLUMNS = (
    "tour_id",
    "person_id",
    "tour_type",
    "tour_type_count",
    "tour_type_num",
    "tour_num",
    "tour_count",
    "tour_category",
    "number_of_participants",
    "destination",
    "origin",
    "household_id",
)


MANDATORY_TOUR_CUDA = r"""
extern "C" __global__ void expand_mandatory_tours(
    const long long* person_id, const long long* household_id,
    const signed char* choice, const unsigned char* is_worker,
    const long long* workplace, const long long* school,
    const long long* home, const long long* offsets, int n,
    long long* tour_id, long long* out_person, signed char* tour_type,
    signed char* type_count, signed char* type_num, signed char* tour_num,
    signed char* tour_count, signed char* category,
    signed char* participants, long long* destination,
    long long* origin, long long* out_household)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int selected = (int)choice[row];
    int work_count = selected == 1 ? 2 : ((selected == 0 || selected == 4) ? 1 : 0);
    int school_count = selected == 3 ? 2 : ((selected == 2 || selected == 4) ? 1 : 0);
    int total = work_count + school_count;
    long long output = offsets[row];
    for (int position = 0; position < total; ++position) {
        bool work_tour = position < work_count;
        int nth = work_tour ? position + 1 : position - work_count + 1;
        int sequence = position + 1;
        if (selected == 4 && !is_worker[row]) sequence = 3 - sequence;
        long long canonical = work_tour ? 38 + nth : 30 + nth;
        tour_id[output] = person_id[row] * 41LL + canonical;
        out_person[output] = person_id[row];
        tour_type[output] = work_tour ? 0 : 1;
        type_count[output] = (signed char)(work_tour ? work_count : school_count);
        type_num[output] = (signed char)nth;
        tour_num[output] = (signed char)sequence;
        tour_count[output] = (signed char)total;
        category[output] = 0;
        participants[output] = 1;
        destination[output] = work_tour ? workplace[row] : school[row];
        origin[output] = home[row];
        out_household[output] = household_id[row];
        ++output;
    }
}
"""


@lru_cache(maxsize=1)
def _mandatory_tour_kernel():
    return _cupy().RawKernel(
        MANDATORY_TOUR_CUDA, "expand_mandatory_tours", options=("--std=c++11",)
    )


def _mandatory_tours(
    person_id: Any,
    household_id: Any,
    mtf_choice: Any,
    is_worker: Any,
    workplace_zone_id: Any,
    school_zone_id: Any,
    home_zone_id: Any,
    xp: Any,
) -> Mapping[str, Any]:
    """Shared array program; ``xp`` is NumPy or CuPy."""

    ids = xp.asarray(person_id, dtype=xp.int64)
    households = xp.asarray(household_id, dtype=xp.int64)
    choices = xp.asarray(mtf_choice, dtype=xp.int8)
    workers = xp.asarray(is_worker, dtype=xp.bool_)
    work_zones = xp.asarray(workplace_zone_id, dtype=xp.int64)
    school_zones = xp.asarray(school_zone_id, dtype=xp.int64)
    home_zones = xp.asarray(home_zone_id, dtype=xp.int64)
    shape = ids.shape
    if ids.ndim != 1 or any(
        value.shape != shape
        for value in (households, choices, workers, work_zones, school_zones, home_zones)
    ):
        raise ValueError("mandatory-tour chooser inputs must be equal-length vectors")

    work_counts = xp.where(
        choices == 1, 2, xp.where((choices == 0) | (choices == 4), 1, 0)
    ).astype(xp.int8)
    school_counts = xp.where(
        choices == 3, 2, xp.where((choices == 2) | (choices == 4), 1, 0)
    ).astype(xp.int8)
    total_counts = work_counts + school_counts

    # Four fixed candidate slots per person make the subsequent Boolean
    # compaction equivalent to ActivitySim stack/repeat ordering:
    # work1, work2, school1, school2.
    slot = xp.arange(4, dtype=xp.int8)[None, :]
    active = xp.column_stack(
        (work_counts >= 1, work_counts >= 2, school_counts >= 1, school_counts >= 2)
    )
    owner = xp.repeat(xp.arange(ids.size, dtype=xp.int64), 4)[active.reshape(-1)]
    active_slot = xp.broadcast_to(slot, active.shape)[active]

    output_person = ids[owner]
    is_work = active_slot < 2
    type_num = xp.where(is_work, active_slot + 1, active_slot - 1).astype(xp.int8)
    type_count = xp.where(is_work, work_counts[owner], school_counts[owner]).astype(xp.int8)
    tour_count = total_counts[owner].astype(xp.int8)

    # Position inside each person's active four-slot row. A prefix sum provides
    # the same one-based cumcount as pandas groupby without leaving the device.
    prefix = xp.cumsum(active.astype(xp.int8), axis=1)
    tour_num = prefix[active].astype(xp.int8)
    swap = (choices[owner] == 4) & ~workers[owner]
    tour_num = xp.where(swap, 3 - tour_num, tour_num).astype(xp.int8)

    offsets = xp.where(
        active_slot == 0,
        39,
        xp.where(active_slot == 1, 40, xp.where(active_slot == 2, 31, 32)),
    ).astype(xp.int64)
    tour_id = output_person * POSSIBLE_TOURS_COUNT + offsets
    destination = xp.where(is_work, work_zones[owner], school_zones[owner])
    return {
        "tour_id": xp.ascontiguousarray(tour_id, dtype=xp.int64),
        "person_id": xp.ascontiguousarray(output_person, dtype=xp.int64),
        "tour_type": xp.ascontiguousarray(
            xp.where(is_work, TOUR_TYPE_WORK, TOUR_TYPE_SCHOOL), dtype=xp.int8
        ),
        "tour_type_count": xp.ascontiguousarray(type_count, dtype=xp.int8),
        "tour_type_num": xp.ascontiguousarray(type_num, dtype=xp.int8),
        "tour_num": xp.ascontiguousarray(tour_num, dtype=xp.int8),
        "tour_count": xp.ascontiguousarray(tour_count, dtype=xp.int8),
        "tour_category": xp.full(tour_id.shape, TOUR_CATEGORY_MANDATORY, dtype=xp.int8),
        "number_of_participants": xp.ones(tour_id.shape, dtype=xp.int8),
        "destination": xp.ascontiguousarray(destination, dtype=xp.int64),
        "origin": xp.ascontiguousarray(home_zones[owner], dtype=xp.int64),
        "household_id": xp.ascontiguousarray(households[owner], dtype=xp.int64),
    }


def mandatory_tours_cpu(
    person_id: Any,
    household_id: Any,
    mtf_choice: Any,
    is_worker: Any,
    workplace_zone_id: Any,
    school_zone_id: Any,
    home_zone_id: Any,
) -> dict[str, np.ndarray]:
    """Independent vectorized CPU reference for mandatory-tour expansion."""

    return dict(
        _mandatory_tours(
            person_id,
            household_id,
            mtf_choice,
            is_worker,
            workplace_zone_id,
            school_zone_id,
            home_zone_id,
            np,
        )
    )


def mandatory_tours_gpu(
    person_id: Any,
    household_id: Any,
    mtf_choice: Any,
    is_worker: Any,
    workplace_zone_id: Any,
    school_zone_id: Any,
    home_zone_id: Any,
) -> DeviceTable:
    """Expand mandatory tours on-device and reject host modeled inputs."""

    inputs = (
        person_id,
        household_id,
        mtf_choice,
        is_worker,
        workplace_zone_id,
        school_zone_id,
        home_zone_id,
    )
    if any(not _is_cuda_array(value) for value in inputs):
        raise GpuOnlyViolation("mandatory-tour expansion inputs must reside on the GPU")
    cp = _cupy()
    ids = cp.ascontiguousarray(person_id, dtype=cp.int64)
    households = cp.ascontiguousarray(household_id, dtype=cp.int64)
    choices = cp.ascontiguousarray(mtf_choice, dtype=cp.int8)
    workers = cp.ascontiguousarray(is_worker, dtype=cp.uint8)
    work_zones = cp.ascontiguousarray(workplace_zone_id, dtype=cp.int64)
    school_zones = cp.ascontiguousarray(school_zone_id, dtype=cp.int64)
    home_zones = cp.ascontiguousarray(home_zone_id, dtype=cp.int64)
    if ids.ndim != 1 or any(
        value.shape != ids.shape
        for value in (households, choices, workers, work_zones, school_zones, home_zones)
    ):
        raise ValueError("mandatory-tour chooser inputs must be equal-length vectors")
    if ids.size and bool(cp.any((choices < 0) | (choices > 4)).item()):
        raise ValueError("mandatory-tour choices must be in [0, 4]")

    counts = (1 + ((choices == 1) | (choices == 3) | (choices == 4))).astype(cp.int64)
    offsets = cp.empty(ids.size + 1, dtype=cp.int64)
    offsets[0] = 0
    cp.cumsum(counts, out=offsets[1:])
    n_tours = int(offsets[-1].item())
    output = {
        "tour_id": cp.empty(n_tours, dtype=cp.int64),
        "person_id": cp.empty(n_tours, dtype=cp.int64),
        "tour_type": cp.empty(n_tours, dtype=cp.int8),
        "tour_type_count": cp.empty(n_tours, dtype=cp.int8),
        "tour_type_num": cp.empty(n_tours, dtype=cp.int8),
        "tour_num": cp.empty(n_tours, dtype=cp.int8),
        "tour_count": cp.empty(n_tours, dtype=cp.int8),
        "tour_category": cp.empty(n_tours, dtype=cp.int8),
        "number_of_participants": cp.empty(n_tours, dtype=cp.int8),
        "destination": cp.empty(n_tours, dtype=cp.int64),
        "origin": cp.empty(n_tours, dtype=cp.int64),
        "household_id": cp.empty(n_tours, dtype=cp.int64),
    }
    if ids.size:
        threads = 256
        blocks = (int(ids.size) + threads - 1) // threads
        _mandatory_tour_kernel()(
            (blocks,),
            (threads,),
            (
                ids,
                households,
                choices,
                workers,
                work_zones,
                school_zones,
                home_zones,
                offsets,
                np.int32(ids.size),
                *(output[name] for name in TOUR_COLUMNS),
            ),
        )
    return DeviceTable(output)
