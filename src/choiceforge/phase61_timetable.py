"""Exact 32-period collision masks, shared by CPU and CUDA consumers."""
from functools import lru_cache
import time
import numba
import numpy as np
from .cuda_backend import _cupy
from .phase59_entity_store import EntityStore


@numba.njit(cache=True)
def encode(values):
    masks = np.zeros((len(values),4), np.uint32)
    for row in range(len(values)):
        for period in range(values.shape[1]):
            value = values[row,period]
            slot = 0 if value == 2 else 1 if value == 4 else 2 if value == 6 else 3 if value == 7 else -1
            if slot >= 0:
                masks[row,slot] |= np.uint32(1) << np.uint32(period)
    return masks


@numba.njit(cache=True, parallel=True)
def available_cpu(windows, footprints, owners, alternatives):
    out = np.empty(len(owners), np.bool_)
    for row in numba.prange(len(owners)):
        w, a = windows[owners[row]], footprints[alternatives[row]]
        out[row] = ((a[0] & (w[0]|w[3])) | (a[1] & (w[1]|w[3])) |
                    (a[2] & w[3]) | (a[3] & (w[0]|w[1]|w[2]|w[3]))) == 0
    return out


SOURCE = r'''
extern "C" __global__ void available(
 const unsigned int* w2, const unsigned int* w4, const unsigned int* w6,
 const unsigned int* w7, const unsigned int* masks, const int* owners,
 const int* alts, int n, unsigned char* out) {
 int r=blockDim.x*blockIdx.x+threadIdx.x; if(r>=n)return;
 int w=owners[r], a=alts[r]*4;
 out[r]=!((masks[a]&(w2[w]|w7[w])) | (masks[a+1]&(w4[w]|w7[w])) |
          (masks[a+2]&w7[w]) | (masks[a+3]&(w2[w]|w4[w]|w6[w]|w7[w])));
}
'''


@lru_cache(maxsize=1)
def kernel():
    return _cupy().RawKernel(SOURCE, "available")


class Availability:
    def __init__(self, *, backend="cuda", capture=None):
        self.store = EntityStore()
        self.backend, self.capture = backend, capture
        self.tables, self.events = {}, []

    def __call__(self, table, ids, alts):
        started = time.perf_counter()
        windows, footprints = np.asarray(table.windows), np.asarray(table.tdd_footprints)
        if (windows.ndim != 2 or footprints.ndim != 2 or windows.shape[1] != footprints.shape[1]
                or not 0 < windows.shape[1] <= 32 or not np.isin(windows,[0,2,4,6,7]).all()
                or not np.isin(footprints,[0,2,4,6,7]).all()):
            raise ValueError("Phase 61 timetable outside 32-period collision vocabulary")
        owners = table.windows_df.index.get_indexer(np.asarray(ids,dtype=np.int64)).astype(np.int32)
        alternatives = np.asarray(alts,dtype=np.int32)
        if (owners.shape != alternatives.shape or owners.ndim != 1 or (owners<0).any()
                or (alternatives<0).any() or (alternatives>=len(footprints)).any()):
            raise ValueError("Phase 61 timetable identities or alternative positions invalid")
        name = str(id(table))
        prior = self.tables.get(name)
        # Retain the table object to prevent identity reuse. Read live authoritative
        # windows each call, including rollback/replacement and joint-tour windows.
        if prior is None or not np.array_equal(prior["footprints"],footprints):
            masks = encode(footprints)
            prior = {"table":table,"footprints":footprints.copy(),"masks":masks,
                     "device_masks":_cupy().asarray(masks) if self.backend == "cuda" else None}
            self.tables[name] = prior
        packed = encode(windows)
        columns = tuple("state"+str(i) for i in range(4))
        lease = None
        if self.capture is not None:
            self.capture(windows, footprints, packed, prior["masks"], owners, alternatives)
        if self.backend == "cuda":
            cp = _cupy()
            lease = self.store.publish(name, table.windows_df.index,
                                       {k:packed[:,i] for i,k in enumerate(columns)})
            result = cp.empty(len(owners),cp.uint8)
            if len(owners):
                kernel()(((len(owners)+255)//256,), (256,), (*lease.arrays(),prior["device_masks"],
                    cp.asarray(owners),cp.asarray(alternatives),np.int32(len(owners)),result))
            output = cp.asnumpy(result).astype(bool)
        elif self.backend == "cpu":
            previous = numba.get_num_threads()
            try:
                numba.set_num_threads(min(24,numba.config.NUMBA_NUM_THREADS))
                output = available_cpu(packed,prior["masks"],owners,alternatives)
            finally:
                numba.set_num_threads(previous)
        else:
            raise ValueError("Phase 61 unknown timetable backend")
        self.events.append({"rows":len(owners),"table":name,"backend":self.backend,
            "cpu_threads":min(24,numba.config.NUMBA_NUM_THREADS) if self.backend=="cpu" else None,
            "generation":lease.generation if lease is not None else None,"seconds":time.perf_counter()-started,
            "query_upload_bytes":owners.nbytes+alternatives.nbytes if self.backend=="cuda" else 0,
            "result_download_bytes":len(output) if self.backend=="cuda" else 0})
        return output
