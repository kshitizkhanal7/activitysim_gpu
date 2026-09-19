"""Qualify two reversed-order repetitions of complete A-B-A scenario batches."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from run_phase58_comparison import ROOT, RESULTS
from qualify_phase59 import require


def qualify(documents, source):
    require(len(documents)==8,"Eight strategy runs required")
    cases = [("candidate","fresh"),("candidate","batch"),("regular","fresh"),("regular","batch")]
    ordered = sorted(documents,key=lambda d:(d["series_trial"],d["series_position"]))
    for trial in (1,2):
        group = [d for d in ordered if d["series_trial"]==trial]
        expected = cases if trial==1 else cases[::-1]
        require([(d["mode"],d["execution"]) for d in group]==expected
                and [d["series_position"] for d in group]==list(range(4)),"Unbalanced strategy order")
    require(all(a["started_at_ns"]<b["started_at_ns"] for a,b in zip(ordered,ordered[1:])),"Series timestamps contradict order")
    grouped = {}
    data = config = None
    outputs = set()
    for document in documents:
        require(document["complete"] and document["sequence"]=="A-B-A" and document["setup_included"],"Incomplete batch")
        require(document["skim_cache"]=="none","Unselected skim cache in timed series")
        require(math.isfinite(document["batch_process_wall_seconds"]) and document["batch_process_wall_seconds"]>0,"Invalid batch time")
        require(document["source_sha256"]==source,"Batch source differs from fresh qualification")
        if data is None:
            data,config = document["data_sha256"],document["configuration_sha256"]
        require(document["data_sha256"]==data and document["configuration_sha256"]==config,"Batch inputs differ")
        require([r["name"] for r in document["runs"]]==["A1","B","A2"],"Invalid isolation sequence")
        require(document["runs"][0]["reference"]==document["runs"][2]["reference"] != document["runs"][1]["reference"],"Wrong independent references")
        for run in document["runs"]:
            require(run["output"] not in outputs,"Output reused across independently executed scenarios")
            outputs.add(run["output"])
            require(run["exact"]["decision_columns_exact"] and run["matrices"]["exact"]
                    and run["summary_reports"]["exact"] and len(run["components"])==34,"Outputs failed")
            require(run["environment"]["NUMBA_NUM_THREADS"]=="48" and run["environment"]["OMP_WAIT_POLICY"]=="PASSIVE","Unequal batch thread policy")
            if document["mode"]=="candidate":
                proof = json.loads(Path(run["report"]).read_text())
                require(all(proof["proof_gates"].values()) and proof["phase62_reusable_execution"]["enabled"],"Missing live candidate proof")
        if document["execution"]=="batch":
            worker = document["worker"]
            require(worker["complete"] and worker["generated_program_sha256"],"No program provenance")
            require(worker["input_tables"]["hits"]>0 and worker["input_tables"]["returns_private_copies"],"No isolated raw-input reuse")
            require(0<worker["input_tables"]["retained_bytes"]<=worker["input_tables"]["limit_bytes"]<=1024**3,"Unbounded input retention")
            require(all(r["reset_modules"]>0 and r["reused_cpu_programs"]>0 for r in worker["runs"][1:]),"No state reset or CPU plan reuse")
            if document["mode"]=="candidate":
                require(all(r["reused_source_keyed_cuda_programs"]>0 for r in worker["runs"][1:]),"No CUDA program reuse")
        grouped.setdefault((document["mode"],document["execution"]),[]).append(document)
    require(set(grouped)=={(m,e) for m in ("candidate","regular") for e in ("fresh","batch")},"Missing control strategy")
    require(all(len(v)==2 for v in grouped.values()),"Two repetitions per strategy required")
    medians = {f"{m}_{e}":statistics.median(d["batch_process_wall_seconds"] for d in group)
               for (m,e),group in grouped.items()}
    ratios = {m:medians[f"{m}_fresh"]/medians[f"{m}_batch"] for m in ("candidate","regular")}
    return {"status":"qualified","sequence":"A-B-A","medians":medians,
            "fresh_over_persistent":ratios,"persistent_cpu_over_hybrid":medians["regular_batch"]/medians["candidate_batch"],
            "all_output_gates_pass":True,"source_sha256":source,"data_sha256":data,
            "scope":"Two repetitions per strategy, four-strategy order reversed in repetition two; full A-B-A outputs independently audited. All setup, fresh private input copies and scenario input hashing included. One machine, fixed public network and two seeds, not a universal isolation or performance guarantee."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",default="p62formal")
    args = parser.parse_args()
    paths = [RESULTS/f"phase62-{args.tag}-{m}-{e}-{trial}.json" for trial in (1,2)
             for m,e in (("candidate","fresh"),("candidate","batch"),("regular","fresh"),("regular","batch"))]
    documents = [json.loads(p.read_text()) for p in paths]
    qualification = RESULTS/"phase62-formal-qualification.json"
    result = qualify(documents,json.loads(qualification.read_text())["source_sha256"])
    evidence = [*paths,qualification,Path(__file__)]
    for document in documents:
        if document["mode"]=="candidate":
            evidence += [Path(r["report"]) for r in document["runs"]]
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    (RESULTS/"phase62-batch-comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# Phase 62: complete repeated-scenario comparison","",result["scope"],"",
             "Seconds for all three scenarios, including the first run and process setup. These are batch totals, not single-run times.","",
             "| Engine | Three fresh processes | One persistent process | Fresh / persistent |","|---|---:|---:|---:|"]
    for mode,label in (("candidate","Phase 62 hybrid"),("regular","Regular CPU / 48 threads")):
        med = result["medians"]
        lines.append(f"| {label} | {med[mode+'_fresh']:.2f} | {med[mode+'_batch']:.2f} | {result['fresh_over_persistent'][mode]:.3f}x |")
    lines += ["",f"Persistent CPU / persistent hybrid: {result['persistent_cpu_over_hybrid']:.3f}x.","",
              "The selected worker reuses programs and private raw-input snapshots, not modeled decisions. Numeric input tables are deep-copied for each scenario. GPU skim content caching is implemented and tested but disabled in the selected path because its cost and cache misses outweighed its benefit in development. Fresh single-process results and the 70-second target are reported separately.",""]
    (ROOT/"docs/phase62-batch-comparison.md").write_text("\n".join(lines))
    print(json.dumps(result["medians"],indent=2))


if __name__ == "__main__":
    main()
