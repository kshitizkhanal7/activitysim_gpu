"""Diagnostic only: test whether compiled CPU normals are byte-identical."""
import json
import time
import numba
import numpy as np


@numba.njit(parallel=True, fastmath=False)
def normals(seeds,offsets,size):
    out = np.empty((len(seeds),size),np.float64)
    discarded = np.empty(len(seeds),np.float64)
    for row in numba.prange(len(seeds)):
        np.random.seed(seeds[row])
        checksum = 0.
        for j in range(offsets[row]):
            checksum += np.random.random()
        discarded[row] = checksum
        for j in range(size):
            out[row,j] = np.random.standard_normal()
    return out, discarded


if __name__ == "__main__":
    rng = np.random.default_rng(61991)
    seeds = rng.integers(0,2**32,10000,dtype=np.uint32)
    offsets = rng.integers(0,40,len(seeds),dtype=np.int64)
    normals(seeds[:1],offsets[:1],6)
    started = time.perf_counter()
    actual, discarded = normals(seeds,offsets,6)
    elapsed = time.perf_counter()-started
    expected = np.empty_like(actual)
    for row in range(len(seeds)):
        state = np.random.RandomState(int(seeds[row]))
        skipped = state.random_sample(int(offsets[row]))
        assert discarded[row] == sum(skipped)
        expected[row] = state.standard_normal(6)
    print(json.dumps({"draws":actual.size,"different_float64_bits":int(np.count_nonzero(
        actual.view(np.uint64)!=expected.view(np.uint64))),"max_absolute_error":float(np.max(np.abs(actual-expected))),
        "compiled_seconds":elapsed,"policy":"diagnostic only; any bit mismatch rejects substitution"}))
