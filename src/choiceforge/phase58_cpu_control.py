"""Compiled CPU control for the identical compact scheduling boundary."""
import numba
import numpy as np


@numba.njit(cache=True, parallel=True)
def cpu_pairs(windows, footprints, slots):
    n, periods = windows.shape
    a = len(footprints)
    first = np.full((n, 25), a, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    for owner in numba.prange(n):
        for alt in range(a):
            available = True
            for p in range(periods):
                c, w = footprints[alt, p], windows[owner, p]
                if ((c == 2 and (w == 2 or w == 7))
                    or (c == 4 and (w == 4 or w == 7))
                    or (c == 7 and (w == 2 or w == 4 or w == 6 or w == 7))
                    or (c == 6 and w == 7)):
                    available = False
                    break
            if available:
                first[owner, slots[alt]] = min(first[owner, slots[alt]], alt)
                counts[owner] += 1
    return first, counts


def period_pairs_cpu(windows, footprints, slots, threads=48):
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        return cpu_pairs(np.ascontiguousarray(windows, dtype=np.int8),
                         np.ascontiguousarray(footprints, dtype=np.int8),
                         np.ascontiguousarray(slots, dtype=np.int32))
    finally:
        numba.set_num_threads(previous)
