"""Reject incomplete, unfair, stale, or output-replayed durability evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_phase58_comparison import ROOT, RESULTS
from run_phase63_sequence import SEQUENCE

CASES = [("candidate","fresh"),("candidate","batch"),("regular","fresh"),("regular","batch")]
# Predeclared finite-run retention tolerances, not an infinite-service guarantee.
HOST_GROWTH_LIMIT = 512*1024**2
DEVICE_GROWTH_LIMIT = 128*1024**2


def require(value,message):
    if not value:
        raise ValueError(message)


def qualify(documents):
    require(len(documents)==8,"Need two reversed-order trials of all four strategies")
    ordered = sorted(documents,key=lambda d:(d["series_trial"],d["series_position"]))
    for trial in (1,2):
        group = [d for d in ordered if d["series_trial"]==trial]
        require([(d["mode"],d["execution"]) for d in group]==(CASES if trial==1 else CASES[::-1]),"Unbalanced strategy order")
        require([d["series_position"] for d in group]==list(range(4)),"Invalid series positions")
    require(all(a["started_at_ns"]<b["started_at_ns"] for a,b in zip(ordered,ordered[1:])),"Timestamps contradict execution order")
    first = ordered[0]
    for field in ("source_sha256","data_sha256","configuration_sha256","reference_output_sha256"):
        require(bool(first[field]),f"Missing {field}")
        require(all(d[field]==first[field] for d in documents),f"Different {field}")
    outputs,grouped,memory = set(),{},[]
    references = {}
    for doc in documents:
        require(not doc.get("diagnostic_not_performance",False),"Instrumented diagnostic is not timing evidence")
        require(doc["complete"] and doc["setup_included"] and doc["verification_outside_timed_region"],"Incomplete timing contract")
        require(doc["sequence"]==SEQUENCE.split(",") and len(doc["runs"])==10,"Incomplete ten-scenario sequence")
        require(doc["features"]=="plans,files,rss","Unequal preparation options")
        require(math.isfinite(doc["batch_process_wall_seconds"]) and doc["batch_process_wall_seconds"]>0,"Invalid clock")
        require(len(doc["workers"])==(1 if doc["execution"]=="batch" else 10),"Incorrect process strategy")
        require(len(doc["process_wall_seconds"])==len(doc["workers"]),"Missing process clocks")
        require(abs(sum(doc["process_wall_seconds"])-doc["batch_process_wall_seconds"])<1e-6,"Total excludes a process")
        for key,run in zip(doc["sequence"],doc["runs"]):
            require(run["scenario"]==key and run["output"] not in outputs,"Scenario reorder or output replay")
            outputs.add(run["output"])
            references.setdefault(key,run["reference"])
            require(run["reference"]==references[key],"Reference changed between repetitions")
            require(run["output"]!=run["reference"],"Candidate is its own reference")
            require(run["exact"]["decision_columns_exact"] and run["matrices"]["exact"] and run["summary_reports"]["exact"],"Output mismatch")
            require(len(run["components"])==34,"Missing complete-model components")
            require(run["environment"]["NUMBA_NUM_THREADS"]=="48" and run["environment"]["OMP_WAIT_POLICY"]=="PASSIVE","Unequal thread controls")
            require(run["memory_before"]["uss_bytes"]>0 and run["memory_after"]["uss_bytes"]>0,"Missing actual USS boundary samples")
            require(run["rss_sampling"]["observed_peak_rss_bytes"]>0 and run["rss_sampling"]["samples"]>0,
                    "Missing sampled RSS high water")
            if doc["mode"]=="candidate":
                require(run["memory_before"]["gpu_pool_used_bytes"]==0,"Previous invocation still owns active GPU arrays")
                proof = json.loads(Path(run["report"]).read_text())
                require(all(proof["proof_gates"].values()) and proof["phase63_durable_execution"]["enabled"],"Missing independent live GPU proof")
                require(set(proof["phase63_durable_execution"]["features"])=={"plans","files","rss"},"Candidate feature mismatch")
            else:
                require(run["cpu_preparation"]["enabled"] and set(run["cpu_preparation"]["features"])=={"plans","files","rss"},"CPU was denied equivalent preparation")
        for worker in doc["workers"]:
            require(worker["complete"] and worker["generated_program_sha256"],"Missing generated program provenance")
            tables = worker["input_tables"]
            require(tables["returns_private_copies"] and 0<tables["retained_bytes"]<=tables["limit_bytes"]<=1024**3,"Unbounded or shared input snapshots")
            if doc["mode"]=="candidate":
                require(worker["memory_after_final_reset"]["gpu_pool_used_bytes"]==0,"Final reset retained GPU arrays")
        if doc["execution"]=="batch":
            require(all(r["reset_modules"]>0 and r["reused_cpu_programs"]>0 for r in doc["runs"][1:]),"No reset or CPU program reuse")
            if doc["mode"]=="candidate":
                require(all(r["reused_source_keyed_cuda_programs"]>0 for r in doc["runs"][1:]),"No source-keyed GPU reuse")
            a,b = doc["runs"][4]["memory_after"],doc["runs"][9]["memory_after"]
            uss_growth = b["uss_bytes"]-a["uss_bytes"]
            gpu_growth = b.get("gpu_pool_used_bytes",0)-a.get("gpu_pool_used_bytes",0)
            memory.append(dict(mode=doc["mode"],trial=doc["series_trial"],uss_growth_bytes=uss_growth,
                               gpu_active_growth_bytes=gpu_growth,within_retention_limits=
                               uss_growth<=HOST_GROWTH_LIMIT and gpu_growth<=DEVICE_GROWTH_LIMIT))
        grouped.setdefault(doc["mode"]+"_"+doc["execution"],[]).append(doc["batch_process_wall_seconds"])
    require(len(set(references.values()))==4,"Changed scenarios share an old reference")
    medians = {k:statistics.median(v) for k,v in grouped.items()}
    return dict(status="outputs_qualified" if all(m["within_retention_limits"] for m in memory) else "outputs_qualified_memory_growth_requires_investigation",
                all_output_gates_pass=True,model_runs=len(outputs),medians=medians,memory=memory,
                memory_limits=dict(uss_growth_bytes=HOST_GROWTH_LIMIT,gpu_active_growth_bytes=DEVICE_GROWTH_LIMIT),
                persistent_cpu_over_hybrid=medians["regular_batch"]/medians["candidate_batch"],
                fresh_cpu_over_hybrid=medians["regular_fresh"]/medians["candidate_fresh"],
                scope="Two reversed-order trials; ten complete scenarios per strategy; same fixed public network. Finite retention checks are not proof of no leaks for arbitrary future scenarios.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",default="formal")
    args = parser.parse_args()
    paths = [RESULTS/f"phase63-{args.tag}-{m}-{e}-{trial}.json" for trial in (1,2) for m,e in CASES]
    docs = [json.loads(p.read_text()) for p in paths]
    result = qualify(docs)
    evidence = paths+[Path(__file__)]+[Path(r["report"]) for d in docs if d["mode"]=="candidate" for r in d["runs"]]
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    for field in ("source_sha256","data_sha256","configuration_sha256","reference_output_sha256"):
        result[field] = docs[0][field]
    target = RESULTS/"phase63-sequence-qualification.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","model_runs","medians")}))


if __name__=="__main__":
    main()
