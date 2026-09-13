"""Balanced equivalent-CPU controls on captured inputs, never cached model answers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import os
# Prevent completed CPU trials' OpenMP workers from busy-spinning while the
# following GPU trial runs. Set before importing/initializing Numba/OpenMP.
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
import numba
import numpy as np
from choiceforge.cuda_backend import _cupy
from choiceforge.phase61_timetable import encode, available_cpu, kernel
from choiceforge.phase61_normals import standard_normals
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from choiceforge.phase59_cpu_algorithms import modes_cpu
from choiceforge.nested_logit import MTC21_ALTERNATIVES


def measure(functions, repetitions):
    for function in functions.values(): function()
    samples = {name:[] for name in functions}
    for trial in range(repetitions):
        names = list(functions) if trial%2 == 0 else list(reversed(functions))
        for name in names:
            started = time.perf_counter()
            functions[name]()
            samples[name].append(time.perf_counter()-started)
    return {"samples_seconds":samples,"medians_seconds":{k:statistics.median(v) for k,v in samples.items()}}


def load(directory):
    return [dict(np.load(p,allow_pickle=False)) for p in sorted(directory.glob("batch-*.npz"))]


def timetable(batches, repetitions):
    from activitysim.core.timetable import COLLISION_ARRAY
    cp = _cupy()
    # Check the original collision vocabulary in bounded chunks, independently
    # from the newly introduced packed equation shared by both fast backends.
    for b in batches:
        expected = available_cpu(b["packed"],b["masks"],b["owners"],b["alternatives"])
        for first in range(0,len(expected),100000):
            sl = slice(first,first+100000)
            raw = b["footprints"][b["alternatives"][sl]] + (b["windows"][b["owners"][sl]] << 3)
            np.testing.assert_array_equal(expected[sl],~np.isin(raw,COLLISION_ARRAY).any(axis=1))
        b["expected"] = expected
        b["device"] = ([cp.asarray(b["packed"][:,i]) for i in range(4)],
                       cp.asarray(b["masks"]),cp.asarray(b["owners"]),cp.asarray(b["alternatives"]))
    def gpu(transfer=False):
        results = []
        for b in batches:
            if transfer:
                packed = encode(b["windows"])
                arrays = ([cp.asarray(packed[:,i]) for i in range(4)],cp.asarray(b["masks"]),
                          cp.asarray(b["owners"]),cp.asarray(b["alternatives"]))
            else: arrays = b["device"]
            n = len(b["owners"])
            result = cp.empty(n,cp.uint8)
            kernel()(((n+255)//256,),(256,),(*arrays[0],*arrays[1:],np.int32(n),result))
            results.append(cp.asnumpy(result).astype(bool) if transfer else result)
        cp.cuda.Stream.null.synchronize()
        return results
    def cpu(pack=False):
        return [available_cpu(encode(b["windows"]) if pack else b["packed"],b["masks"],b["owners"],b["alternatives"]) for b in batches]
    for b,result in zip(batches,gpu(True)):
        np.testing.assert_array_equal(b["expected"],result)
    sweep = []
    for threads in (1,4,12,24,48):
        if threads > numba.config.NUMBA_NUM_THREADS: continue
        numba.set_num_threads(threads)
        result = measure({"cpu_packed":cpu,"gpu_resident":gpu,
                          "cpu_encode_compute":lambda:cpu(True),"gpu_encode_transfer_compute_download":lambda:gpu(True)},repetitions)
        sweep.append({"threads":threads,**result})
    best = min(sweep,key=lambda x:x["medians_seconds"]["cpu_encode_compute"])
    m = best["medians_seconds"]
    return {"scope":"all live timetable queries, authoritative collision vocabulary exact; immutable footprint encoding cached on both; GPU transfer scope re-encodes live windows and uploads all masks and query arrays and downloads booleans; excludes identity lookup, host validation and EntityStore bookkeeping on both",
            "rows":sum(len(b["owners"]) for b in batches),"batches":len(batches),"exact":True,"sweep":sweep,
            "best_cpu_threads":best["threads"],"transfer_inclusive_cpu_over_gpu":m["cpu_encode_compute"]/m["gpu_encode_transfer_compute_download"]}


def modes(batches,repetitions):
    cp = _cupy()
    for b in batches:
        b["nest"] = json.loads(str(b["nest_json"]))
        b["mus"] = validate(b["nest"],MTC21_ALTERNATIVES)
        b["draws"] = b["draws"].reshape(-1)
        b["device"] = (cp.asarray(b["utilities"]),cp.asarray(b["draws"]))
    def cpu():return [modes_cpu(b["utilities"],b["draws"],b["mus"]) for b in batches]
    def gpu(transfer=False):
        results = []
        for b in batches:
            arrays = (cp.asarray(b["utilities"]),cp.asarray(b["draws"])) if transfer else b["device"]
            result = reduce_modes(*arrays,b["nest"])
            results.append(tuple(cp.asnumpy(a) for a in result) if transfer else result)
        cp.cuda.Stream.null.synchronize()
        return results
    checks = []
    for expected,actual in zip(cpu(),gpu(True)):
        guarded = (expected[2]!=0)|(actual[2]!=0)
        np.testing.assert_array_equal(expected[0][~guarded],actual[0][~guarded])
        np.testing.assert_allclose(expected[1],actual[1],rtol=1e-12,atol=1e-12)
        checks.append({"rows":len(guarded),"guarded":int(guarded.sum()),"off_boundary_choices_exact":True,
                       "max_logsum_abs":float(np.max(np.abs(expected[1]-actual[1])))})
    sweep = []
    for threads in (1,4,12,24,48):
        if threads > numba.config.NUMBA_NUM_THREADS:continue
        numba.set_num_threads(threads)
        sweep.append({"threads":threads,**measure({"cpu":cpu,"gpu_resident":gpu,"gpu_transfer_inclusive":lambda:gpu(True)},repetitions)})
    best = min(sweep,key=lambda x:x["medians_seconds"]["cpu"])
    m = best["medians_seconds"]
    return {"scope":"actual live tour utilities, draws and nests; equivalent compiled nested reducers; allocations included; transfer-inclusive uploads utilities/draws and downloads reducer outputs; excludes expression evaluation, RNG and live boundary resolution on both; guarded rows remain subject to production CPU checks",
            "checks":checks,"sweep":sweep,"best_cpu_threads":best["threads"],
            "resident_cpu_over_gpu":m["cpu"]/m["gpu_resident"],"transfer_inclusive_cpu_over_gpu":m["cpu"]/m["gpu_transfer_inclusive"]}


def normals(batches,repetitions):
    def original():
        results = []
        rng = np.random.RandomState()
        for b in batches:
            values = np.empty_like(b["values"])
            for row in range(len(values)):
                rng.seed(int(b["seeds"][row]))
                rng.rand(int(b["offsets"][row]))
                values[row] = rng.normal(0.,1.,size=values.shape[1])
            results.append(values)
        return results
    def compiled():
        return [standard_normals(b["seeds"].astype(np.uint32),b["offsets"].astype(np.int64),b["values"].shape[1]) for b in batches]
    for b,expected,actual in zip(batches,original(),compiled()):
        np.testing.assert_array_equal(expected.view(np.uint64),b["values"].view(np.uint64))
        np.testing.assert_array_equal(expected.view(np.uint64),(actual[0]+0.).view(np.uint64))
        if not np.isfinite(actual[1]).all():raise ValueError("invalid discarded checksum")
    original_timing = measure({"original_numpy":original},repetitions)
    sweep=[]
    for threads in (1,4,12,24,48):
        if threads > numba.config.NUMBA_NUM_THREADS:continue
        numba.set_num_threads(threads)
        sweep.append({"threads":threads,**measure({"compiled_cpu":compiled},repetitions)})
    return {"scope":"all live standard-normal streams; same seeds, offsets, draw counts; original NumPy row loop versus compiled CPU batch; seed/offset conversion and allocations included; excludes dataframe ledger gather/commit on both; this is NOT a GPU gain",
            "rows":sum(len(b["seeds"]) for b in batches),"batches":len(batches),"bit_exact":True,
            "original":original_timing,"sweep":sweep}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=7)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    results={}
    for kind,function in (("timetable",timetable),("tour_modes",modes),("normals",normals)):
        batches=load(args.inputs/kind)
        if not batches:raise ValueError(f"Missing live {kind} inputs")
        results[kind]=function(batches,args.repetitions)
        print(kind+" controls passed",flush=True)
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/"src/choiceforge/phase61_timetable.py",root/"src/choiceforge/phase61_normals.py",
           root/"src/choiceforge/phase58_mode_reduction.py",root/"src/choiceforge/phase59_cpu_algorithms.py"]
    results["source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results["input_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.inputs.rglob("*.npz"))}
    results["not_full_model_performance"]=True
    results["cpu_worker_wait_policy"]="PASSIVE; prevents idle CPU workers contending with following GPU trials"
    args.output.write_text(json.dumps(results,indent=2)+"\n")
