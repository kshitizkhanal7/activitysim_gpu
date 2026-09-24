"""Fail-closed output, fairness and finite-memory gate for the 42-model campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_phase58_comparison import ROOT, RESULTS, source_fingerprint
from run_phase63_sequence import SEQUENCE
from qualify_phase63_sequences import require, HOST_GROWTH_LIMIT, DEVICE_GROWTH_LIMIT


def pipeline_check(preparation, features, hybrid=True):
    require(preparation["enabled"] and set(preparation["features"])==features,"Pipeline options differ")
    require(not preparation["saved_answers_read"],"Saved answers used")
    require(0<preparation["code_entries"]<=preparation["code_capacity"]<=8192,"Expression cache not bounded")
    require(preparation["code_hits"]>0,"Expression candidate unused")
    inputs=preparation["inputs"]
    require(inputs is not None and inputs["hits"]==3 and inputs["misses"]==0,"Raw artifacts not used equally")
    require(not inputs["saved_answers_read"],"Input cache contains answers")
    require(all(e["private_copy"] and e["source_sha256"] and e["artifact_sha256"] for e in inputs["events"]),"Input provenance/private copy missing")
    boundary=preparation["location_boundary"]
    require(boundary is not None and boundary["contexts_outstanding"]==0 and not boundary["saved_answers_read"],"Boundary context retained/answers used")
    if hybrid:
        require(bool(boundary["events"]),"Destination safeguard unused")
        require(all(e["complete"] and e["rng_unchanged"] and not e["saved_answers_read"] for e in boundary["events"]),"Destination safeguard incomplete or RNG changed")


def qualify(fresh, cpu, matched, memory, source, load_report):
    require(fresh["complete"] and cpu["complete"],"Incomplete fresh series")
    require(len(fresh["runs"])==12 and len(cpu["runs"])==2,"Need six fresh pairs and two ordinary CPU runs")
    require(len(matched)==4 and len(memory)==2,"Need reversed matched series and two memory sequences")
    features=set(matched[0]["phase64_features"].split(","))
    require(features=={"expressions","inputs","location_boundary"},"Invalid production feature selection")
    outputs=set()
    references={}
    configuration=fresh["runs"][0]["configuration_sha256"]
    reference=fresh["runs"][0]["reference"]
    def check(run):
        require(run["output"] not in outputs and run["output"]!=run["reference"],"Output replay/self-reference")
        outputs.add(run["output"])
        require(run["exact"]["decision_columns_exact"] and run["matrices"]["exact"] and run["summary_reports"]["exact"],"Output mismatch")
        require(len(run["components"])==34,"Incomplete model")
        if "scenario" in run:
            references.setdefault(run["scenario"],run["reference"])
            require(references[run["scenario"]]==run["reference"],"Scenario reference changed")
    for run in fresh["runs"]+cpu["runs"]:
        check(run)
        require(run["source_sha256"]==source and run["configuration_sha256"]==configuration,"Fresh provenance differs")
        require(run["reference"]==reference,"Fresh reference differs")
        require(not run["profiled_not_performance_evidence"],"Instrumented speed evidence")
        require(run["thread_environment"].get("NUMBA_NUM_THREADS")=="48" and run["thread_environment"].get("OMP_WAIT_POLICY")=="PASSIVE","Unmatched thread controls")
        require(math.isfinite(run["process_wall_seconds"]) and run["process_wall_seconds"]>0,"Bad clock")
        if run["mode"]!="regular":
            command=run["command"]
            proof=load_report(command[command.index("--report")+1])
            require(bool(proof["proof_gates"]) and all(proof["proof_gates"].values()),"Fresh live proof failed")
            if run["mode"]=="candidate":
                pipeline_check(proof["phase64_pipeline"],features)
    require([r["mode"] for r in cpu["runs"]]==["regular","regular"],"Not ordinary CPU")
    pairs=[]
    for trial in range(1,7):
        runs=[r for r in fresh["runs"] if r["trial"]==trial]
        require([r["mode"] for r in runs]==(["gpu","candidate"] if trial%2 else ["candidate","gpu"]),"Unbalanced fresh order")
        old,new=[next(r for r in runs if r["mode"]==m) for m in ("gpu","candidate")]
        pairs.append(dict(trial=trial,old_seconds=old["process_wall_seconds"],new_seconds=new["process_wall_seconds"],
                          old_charged=old["charged_total_seconds"],new_charged=new["charged_total_seconds"]))
    ordered=sorted(matched,key=lambda d:(d["series_trial"],d["series_position"]))
    require([(d["series_trial"],d["series_position"],d["mode"]) for d in ordered]==
            [(1,0,"regular"),(1,1,"candidate"),(2,0,"candidate"),(2,1,"regular")],"Matched order not reversed")
    require(all(a["started_at_ns"]<b["started_at_ns"] for a,b in zip(ordered,ordered[1:])),"Matched timestamps reversed")
    selected={"regular":[],"candidate":[]}
    retention=[]
    first=matched[0]
    for doc in matched+memory:
        require(doc["complete"] and doc["setup_included"] and doc["verification_outside_timed_region"],"Incomplete worker contract")
        require(not doc["diagnostic_not_performance"],"Diagnostic worker included")
        require(doc["features"]=="plans,files,rss" and set(doc["phase64_features"].split(","))==features,"Unequal preparation")
        for field in ("source_sha256","data_sha256","configuration_sha256","reference_output_sha256"):
            require(bool(doc[field]) and doc[field]==first[field],f"Worker {field} differs")
        require(all(doc["source_sha256"].get(k)==v for k,v in source.items()),"Worker source not current")
        batch=doc["execution"]=="batch"
        require(doc["sequence"]==(SEQUENCE.split(",") if batch else ["A","A"]),"Incomplete scenario sequence")
        require(len(doc["runs"])==len(doc["sequence"]),"Missing scenario")
        require(len(doc["workers"])==(1 if batch else 2),"Wrong worker strategy")
        require(all(math.isfinite(t) and t>0 for t in doc["process_wall_seconds"]),"Bad worker clock")
        require(len(doc["process_wall_seconds"])==len(doc["workers"]) and
                abs(sum(doc["process_wall_seconds"])-doc["batch_process_wall_seconds"])<1e-6,"Uncharged worker time")
        for index,run in enumerate(doc["runs"]):
            check(run)
            require(run["scenario"]==doc["sequence"][index],"Reordered scenarios")
            require(run["environment"]["NUMBA_NUM_THREADS"]=="48" and run["environment"]["OMP_WAIT_POLICY"]=="PASSIVE","Unmatched worker threads")
            require(run["memory_before"]["uss_bytes"]>0 and run["memory_after"]["uss_bytes"]>0,"USS samples absent")
            require(run["rss_sampling"]["samples"]>0 and run["rss_sampling"]["observed_peak_rss_bytes"]>0,"RSS samples absent")
            if doc["mode"]=="candidate":
                require(run["memory_before"]["gpu_pool_used_bytes"]==0,"Active GPU arrays survived reset")
                proof=load_report(run["report"])
                require(bool(proof["proof_gates"]) and all(proof["proof_gates"].values()),"Live proof failed")
                pipeline_check(proof["phase64_pipeline"],features)
            else:
                require(run["cpu_preparation"]["enabled"] and set(run["cpu_preparation"]["features"])=={"plans","files","rss"},"CPU lacks previous preparation")
                pipeline_check(run["cpu_pipeline_preparation"],features,hybrid=False)
            if not batch:
                require(run["reference"]==reference,"Matched reference differs")
                selected[doc["mode"]].append(doc["process_wall_seconds"][index])
        for worker in doc["workers"]:
            require(worker["complete"] and worker["generated_program_sha256"],"Generated provenance missing")
            tables=worker["input_tables"]
            # Zero is correct when the verified artifact bypasses the old CSV cache.
            require(tables["returns_private_copies"] and 0<=tables["retained_bytes"]<=tables["limit_bytes"]<=1024**3,"Unbounded/shared raw tables")
            if doc["mode"]=="candidate":
                require(worker["memory_after_final_reset"]["gpu_pool_used_bytes"]==0,"Final reset retained GPU arrays")
        if batch:
            require(doc["mode"]=="candidate","Only hybrid durability is qualified by this campaign")
            require(all(r["reset_modules"]>0 and r["reused_cpu_programs"]>0 and r["reused_source_keyed_cuda_programs"]>0 for r in doc["runs"][1:]),"Reset/reuse absent")
            a,b=[doc["runs"][i]["memory_after"] for i in (4,9)]
            host=b["uss_bytes"]-a["uss_bytes"]
            gpu=b["gpu_pool_used_bytes"]-a["gpu_pool_used_bytes"]
            retention.append(dict(trial=doc["series_trial"],uss_growth_bytes=host,gpu_growth_bytes=gpu,
                                  passed=host<=HOST_GROWTH_LIMIT and gpu<=DEVICE_GROWTH_LIMIT))
    require({d["series_trial"] for d in memory}=={1,2},"Memory trial repeated")
    require(len(references)==4 and len(set(references.values()))==4,"Scenario references not independent")
    require(all(len(v)==4 for v in selected.values()),"Need four matched fresh processes per engine")
    groups={"phase63":[r for r in fresh["runs"] if r["mode"]=="gpu"],
            "phase64":[r for r in fresh["runs"] if r["mode"]=="candidate"],"ordinary_cpu":cpu["runs"]}
    medians={name:statistics.median(r["process_wall_seconds"] for r in runs) for name,runs in groups.items()}
    matched_medians={k:statistics.median(v) for k,v in selected.items()}
    return dict(status="outputs_and_memory_qualified" if all(r["passed"] for r in retention) else "memory_investigation_required",
        model_runs=len(outputs),features=sorted(features),medians_seconds=medians,pairs=pairs,
        all_pairs_faster=all(p["new_seconds"]<p["old_seconds"] and p["new_charged"]<p["old_charged"] for p in pairs),
        under_70=medians["phase64"]<70,under_65=medians["phase64"]<65,
        ordinary_cpu_over_hybrid=medians["ordinary_cpu"]/medians["phase64"],
        previous_over_latest=medians["phase63"]/medians["phase64"],
        matched_fresh=dict(samples_seconds=selected,medians_seconds=matched_medians,
                           cpu_over_hybrid=matched_medians["regular"]/matched_medians["candidate"]),
        memory=retention,components={name:{step:statistics.median(r["components"][step] for r in runs)
                    for step in runs[0]["components"]} for name,runs in groups.items()},
        scope="42 warm-installed full models. Four matched-preparation fresh runs per engine; two ten-scenario hybrid-only memory sequences. No new persistent CPU comparison, clean install or cross-machine replication claim. GPU active arrays must be zero after resets; finite host/device growth bounds are not a proof of no future leaks.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--cpu-control",type=Path,help="Explicit replacement receipt; all provenance and thread gates still apply")
    args=parser.parse_args()
    target=RESULTS/f"phase64-{args.tag}-qualification.json"
    require(not target.exists(),"Preserve existing receipts")
    paths=[RESULTS/f"phase58-{args.tag}-{suffix}-summary.json" for suffix in ("fresh","cpu48")]
    if args.cpu_control:
        paths[1]=args.cpu_control
    paths += [RESULTS/f"phase64-{args.tag}-matched-{mode}-{trial}.json" for trial in (1,2) for mode in ("regular","candidate")]
    paths += [RESULTS/f"phase64-{args.tag}-memory-{trial}.json" for trial in (1,2)]
    docs=[json.loads(p.read_text()) for p in paths]
    reports=[]
    def read_report(name):
        path=Path(name)
        reports.append(path)
        return json.loads(path.read_text())
    result=qualify(docs[0],docs[1],docs[2:6],docs[6:],source_fingerprint(),read_report)
    result["source_sha256"]=source_fingerprint()
    result["evidence_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+reports+[Path(__file__)]}
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","model_runs","medians_seconds","under_70")}))


if __name__=="__main__":
    main()
