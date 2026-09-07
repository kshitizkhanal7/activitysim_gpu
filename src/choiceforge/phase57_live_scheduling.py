"""Build representative logsum rows from the current timetable on CUDA.

No saved logsum or final-output values are consumed. The reduction preserves
ActivitySim's first-available-alternative order for each tour and period pair.
"""
from __future__ import annotations

from functools import lru_cache
import time
import numpy as np
import pandas as pd

SOURCE = r'''
extern "C" __global__ void feasible_period_pairs(
 const signed char* windows, const signed char* footprints,
 const int* slots, int n, int a, int periods, int* first, int* counts) {
 const int flat = blockIdx.x * blockDim.x + threadIdx.x;
 if (flat >= n*a) return;
 const int owner = flat/a, alt = flat%a;
 for (int p=0; p<periods; ++p) {
   const int c=footprints[alt*periods+p], w=windows[owner*periods+p];
   if ((c==2 && (w==2 || w==7)) || (c==4 && (w==4 || w==7)) ||
       (c==7 && (w==2 || w==4 || w==6 || w==7)) || (c==6 && w==7)) return;
 }
 atomicMin(first+owner*25+slots[alt], alt);
 atomicAdd(counts+owner, 1);
}
'''


@lru_cache(maxsize=1)
def _kernel():
    from choiceforge.cuda_backend import _cupy
    return _cupy().RawKernel(SOURCE, "feasible_period_pairs", options=("--std=c++11",))


def period_pairs_cuda(windows, footprints, slots):
    """Return first feasible TDD for each pair and feasible row counts."""
    from choiceforge.cuda_backend import _cupy
    cp = _cupy()
    windows = np.ascontiguousarray(windows, dtype=np.int8)
    footprints = np.ascontiguousarray(footprints, dtype=np.int8)
    slots = np.ascontiguousarray(slots, dtype=np.int32)
    if windows.ndim != 2 or footprints.ndim != 2 or windows.shape[1] != footprints.shape[1]:
        raise ValueError("timetable and footprint shapes differ")
    n, periods = windows.shape
    a = len(footprints)
    if a == 0 or slots.shape != (a,) or np.any((slots < 0) | (slots >= 25)):
        raise ValueError("unsupported period-pair layout")
    if n*a > np.iinfo(np.int32).max:
        raise ValueError("period-pair launch exceeds int32 capacity")
    first = cp.full((n, 25), a, dtype=cp.int32)
    counts = cp.zeros(n, dtype=cp.int32)
    if n:
        _kernel()(((n*a+255)//256,), (256,), (
            cp.asarray(windows), cp.asarray(footprints), cp.asarray(slots),
            np.int32(n), np.int32(a), np.int32(periods), first, counts,
        ))
    return cp.asnumpy(first), cp.asnumpy(counts)


def representative_rows(state, tours, alts, timetable, window_id_col):
    started = time.perf_counter()
    network = state.get_injectable("network_los")
    out = network.skim_time_period_label(alts["start"], as_cat=True)
    inc = network.skim_time_period_label(alts["end"], as_cat=True)
    if not isinstance(out.dtype, pd.CategoricalDtype) or out.dtype != inc.dtype or len(out.cat.categories) != 5:
        raise ValueError("Phase 57 live scheduling requires five matching categorical skim periods")
    if not np.array_equal(alts.index, np.arange(len(alts))):
        raise ValueError("Phase 57 live scheduling requires positional TDD identities")
    slots = out.cat.codes.to_numpy(dtype=np.int32)*5 + inc.cat.codes.to_numpy(dtype=np.int32)
    rows = timetable.window_row_ix.apply_to(np.asarray(tours[window_id_col]))
    first, counts = period_pairs_cuda(timetable.windows[rows], timetable.tdd_footprints, slots)
    if np.any(counts == 0):
        raise ValueError("Phase 57 found a tour without feasible time alternatives")
    ordered = np.argsort(first, axis=1, kind="stable")
    present = np.take_along_axis(first, ordered, axis=1) < len(alts)
    owners, positions = np.nonzero(present)
    selected_slots = ordered[owners, positions]
    # _compute_logsums only needs representative period labels and times.
    result = pd.DataFrame(index=tours.index.take(owners))
    result["out_period"] = pd.Categorical.from_codes(selected_slots//5, dtype=out.dtype)
    result["in_period"] = pd.Categorical.from_codes(selected_slots%5, dtype=inc.dtype)
    return result, {
        "tours": len(tours), "full_interaction_rows_avoided": int(counts.sum()),
        "representative_rows": len(result), "pair_download_bytes": first.nbytes+counts.nbytes,
        "seconds": time.perf_counter()-started,
    }


def representative_times(state, rows, purpose):
    segments = state.get_injectable("tdd_alt_segments", None)
    if segments is None:
        raise ValueError("Phase 57 requires configured representative TDD segments")
    segments = segments.copy()
    if "tour_purpose" in segments:
        match = segments.tour_purpose == purpose
        if not match.any():
            match = segments.tour_purpose.isnull()
        segments = segments.loc[match]
    if segments.time_period.duplicated().any() or len(segments) != 5:
        raise ValueError("Phase 57 requires exactly one representative per skim period")
    lookup = segments.set_index("time_period")
    result = rows.copy()
    result["start"] = lookup.start.reindex(result.out_period.to_numpy()).to_numpy()
    result["end"] = lookup.end.reindex(result.in_period.to_numpy()).to_numpy()
    if result[["start", "end"]].isna().any().any():
        raise ValueError("Phase 57 representative time lookup is incomplete")
    result["duration"] = result["end"]-result["start"]
    return result
