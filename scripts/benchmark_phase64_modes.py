"""Same-input nested-reducer CPU/GPU controls, not complete-model timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ["OMP_WAIT_POLICY"]="PASSIVE"
import numba
import numpy as np
import pandas as pd
from activitysim.core import simulate, logit
from types import SimpleNamespace
from choiceforge.cuda_backend import _cupy
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from choiceforge.phase59_cpu_algorithms import modes_cpu


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9)
    args=parser.parse_args()
    if args.output.exists() or args.repetitions<3:
        raise ValueError("New output and at least three repetitions required")
    files=sorted(p for d in args.inputs for p in d.glob("batch-*.npz"))
    if not files:
        raise ValueError("No real captured inputs")
    cp=_cupy()
    preparation=time.perf_counter()
    batches=[]
    checks=[]
    state=SimpleNamespace(settings=SimpleNamespace(skip_failed_choices=False))
    for path in files:
        with np.load(path,allow_pickle=False) as raw:
            values=raw["utilities"].copy()
            draws=raw["draws"].copy().reshape(-1)
            nest=json.loads(str(raw["nest_json"]))
        coefficients=validate(nest,MTC21_ALTERNATIVES)
        device=(cp.asarray(values),cp.asarray(draws))
        cpu=modes_cpu(values,draws,coefficients)
        gpu=tuple(cp.asnumpy(x) for x in reduce_modes(*device,nest)[:3])
        frame=pd.DataFrame(values,columns=MTC21_ALTERNATIVES)
        nested=simulate.compute_nested_exp_utilities(frame,nest)
        conditional=simulate.compute_nested_probabilities(state,nested,nest,"phase64_control")
        probabilities=simulate.compute_base_probabilities(conditional,nest,frame).to_numpy()
        expected=logit.choice_maker(probabilities,draws.reshape(-1,1))
        expected_logs=np.log(nested.root.to_numpy())
        guarded=(cpu[2]!=0)|(gpu[2]!=0)
        for result in (cpu,gpu):
            np.testing.assert_array_equal(result[0][~guarded],expected[~guarded])
            np.testing.assert_allclose(result[1],expected_logs,atol=1e-12,rtol=1e-12)
        checks.append(dict(input=str(path),rows=len(values),guarded_rows=int(guarded.sum()),
            off_boundary_choices_exact=True,logsums_rtol=1e-12,logsums_atol=1e-12,
            cpu_max_logsum_error=float(np.max(np.abs(cpu[1]-expected_logs))),
            gpu_max_logsum_error=float(np.max(np.abs(gpu[1]-expected_logs)))))
        batches.append((values,draws,nest,coefficients,device))
    def cpu():
        return [modes_cpu(v,d,c) for v,d,n,c,dev in batches]
    def gpu(transfer=False):
        results=[]
        for v,d,n,c,dev in batches:
            arrays=(cp.asarray(v),cp.asarray(d)) if transfer else dev
            result=reduce_modes(*arrays,n)[:3]
            results.append(tuple(cp.asnumpy(x) for x in result) if transfer else result)
        cp.cuda.Stream.null.synchronize()
        return results
    prep_seconds=time.perf_counter()-preparation
    sweep=[]
    for threads in (1,4,12,24,48):
        if threads>numba.config.NUMBA_NUM_THREADS:
            continue
        numba.set_num_threads(threads)
        functions={"cpu":cpu,"gpu_transfer_inclusive":lambda:gpu(True),"gpu_resident":gpu}
        for f in functions.values():
            f()
        samples={name:[] for name in functions}
        for trial in range(args.repetitions):
            names=list(functions)
            for name in names if trial%2==0 else names[::-1]:
                start=time.perf_counter()
                result=functions[name]()
                samples[name].append(time.perf_counter()-start)
                del result
        sweep.append(dict(threads=threads,samples_seconds=samples,
                          medians_seconds={name:statistics.median(s) for name,s in samples.items()}))
    best=min(sweep,key=lambda r:r["medians_seconds"]["cpu"])
    medians=best["medians_seconds"]
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/"src/choiceforge/phase58_mode_reduction.py",root/"src/choiceforge/phase59_cpu_algorithms.py"]
    result=dict(complete=True,checks=checks,batches=len(batches),rows=sum(c["rows"] for c in checks),
        preparation_validation_seconds=prep_seconds,sweep=sweep,best_cpu_threads=best["threads"],
        cpu_over_gpu_transfer_inclusive=medians["cpu"]/medians["gpu_transfer_inclusive"],
        cpu_over_gpu_resident=medians["cpu"]/medians["gpu_resident"],
        scope="Nested reduction only; same float32 utilities, float64 draws/arithmetic, no fastmath. Host-start controls include GPU uploads/downloads; resident control starts and ends on GPU. Allocations included. Excludes utility evaluation, RNG and live boundary adjudication equally.",
        qualified_boundary_policy="Union of CPU/GPU 1e-9 guards excluded from exact primitive choices; production still adjudicates boundaries live. Both checked against independent upstream nested probabilities.",
        not_full_model_speedup=True,wait_policy="PASSIVE",
        evidence_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files,*sources]})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("complete","batches","rows","best_cpu_threads","cpu_over_gpu_transfer_inclusive","cpu_over_gpu_resident")}))


if __name__=="__main__":
    main()
