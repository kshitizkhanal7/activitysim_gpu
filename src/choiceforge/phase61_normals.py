"""Batched CPU standard normals with the authoritative ActivitySim row ledger.

Broadcast/scaling/lognormal transforms remain in ActivitySim. Non-unit direct
requests retain its RandomState implementation. The skipped-uniform checksum
is observable so the compiler cannot eliminate advancing each seeded stream.
"""
import time
import numba
import numpy as np


@numba.njit(cache=True, parallel=True, fastmath=False)
def standard_normals(seeds, offsets, size):
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
    return out,discarded


def normal(original, events, channel, frame, step, mu, sigma, lognormal=False, size=None, *, capture=None):
    supported = (not lognormal and np.isscalar(mu) and mu == 0 and np.isscalar(sigma) and sigma == 1
                 and (size is None or isinstance(size,(int,np.integer)) and 1 <= size <= 16)
                 and len(frame) > 0 and frame.index.is_unique)
    if not supported:
        return original(channel,frame,step,mu,sigma,lognormal=lognormal,size=size)
    started = time.perf_counter()
    if not channel.step_name or channel.step_name != step:
        raise ValueError("Phase 61 standard normals outside the current row-ledger epoch")
    ledger = channel.row_states.loc[frame.index,["row_seed","offset"]].copy()
    seeds,offsets = ledger.row_seed.to_numpy(),ledger.offset.to_numpy()
    if (seeds.dtype.kind not in "iu" or offsets.dtype.kind not in "iu" or
            (seeds<0).any() or (seeds>=2**32).any() or (offsets<0).any()):
        raise ValueError("Phase 61 invalid seeded normal stream")
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(min(48,numba.config.NUMBA_NUM_THREADS))
        values,discarded = standard_normals(seeds.astype(np.uint32),offsets.astype(np.int64),1 if size is None else size)
    finally:
        numba.set_num_threads(previous)
    values = values + float(mu)  # Match RandomState.normal's loc addition, including signed zero.
    if not np.isfinite(discarded).all() or not np.isfinite(values).all():
        raise ValueError("Phase 61 nonfinite standard normals or skipped-draw checksum")
    if not ledger.equals(channel.row_states.loc[frame.index,["row_seed","offset"]]):
        raise ValueError("Phase 61 normal row ledger changed before commit")
    count = 1 if size is None else int(size)
    if capture is not None:
        capture(seeds, offsets, values, frame.index.to_numpy(), step)
    channel.row_states.loc[frame.index,"offset"] += count
    events.append({"step":step,"rows":len(frame),"draws_per_row":count,
                   "discarded_uniform_checksum":float(discarded.sum()),"seconds":time.perf_counter()-started,
                   "backend":"compiled_cpu_standard_normal","numba_threads":min(48,numba.config.NUMBA_NUM_THREADS)})
    return values[:,0] if size is None else values
