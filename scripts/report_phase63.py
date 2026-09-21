"""Complete-model/component evidence; never attribute all service savings to kernels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_phase58_comparison import ROOT, RESULTS, source_fingerprint


def require(condition,message):
    if not condition:
        raise ValueError(message)


def matched_default_scenarios(sequence):
    """Use equal-preparation fresh-process A cases; do not average small cases."""
    selected = {"candidate":[],"regular":[]}
    for name in sequence["evidence_sha256"]:
        path = Path(name)
        if path.suffix != ".json":
            continue
        doc = json.loads(path.read_text())
        if doc.get("execution") != "fresh" or "series_trial" not in doc:
            continue
        require(len(doc["runs"])==len(doc["process_wall_seconds"]),"Fresh worker clocks misaligned")
        for run,wall in zip(doc["runs"],doc["process_wall_seconds"]):
            if run["scenario"] == "A":
                selected[doc["mode"]].append(dict(output=run["output"],seconds=wall,
                    trial=doc["series_trial"],components=run["components"]))
    require(all(len(v)==8 for v in selected.values()),"Need eight fresh default-scenario cases per engine")
    medians = {mode:statistics.median(r["seconds"] for r in runs) for mode,runs in selected.items()}
    components = {mode:{name:statistics.median(r["components"][name] for r in runs)
                        for name in runs[0]["components"]} for mode,runs in selected.items()}
    return dict(medians=medians,cpu_over_hybrid=medians["regular"]/medians["candidate"],
                cases=selected,component_medians=components,
                scope="Eight fresh 50,000-household default-A processes per engine within the balanced sequence campaign. Both receive plans/files/RSS, 48-thread capacity, PASSIVE waiting, private inputs, per-scenario input hashing and reset telemetry. External output audits excluded. This clock differs from the separate fresh-pair harness.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    args = parser.parse_args()
    fresh_path = RESULTS/f"phase58-{args.tag}-fresh-summary.json"
    cpu_path = RESULTS/f"phase58-{args.tag}-cpu48-summary.json"
    sequence_path = RESULTS/"phase63-sequence-qualification.json"
    fresh,cpu,sequence = [json.loads(p.read_text()) for p in (fresh_path,cpu_path,sequence_path)]
    require(fresh["complete"] and cpu["complete"],"Incomplete fresh controls")
    require(sequence["status"]=="outputs_qualified","Long-scenario memory/output qualification unresolved")
    require(len(fresh["runs"])==12 and len(cpu["runs"])==2,"Expected six pairs and two CPU controls")
    require(all(r["mode"]=="regular" for r in cpu["runs"]),"CPU controls are not regular CPU")
    require([r["trial"] for r in cpu["runs"]]==[1,2],"CPU control repetitions differ")
    source = source_fingerprint()
    configurations = fresh["runs"][0]["configuration_sha256"]
    reference = fresh["runs"][0]["reference"]
    require(bool(configurations) and bool(reference),"Missing configuration/reference provenance")
    outputs = set()
    for run in fresh["runs"]+cpu["runs"]:
        require(run["source_sha256"]==source,"Fresh measurement source differs from delivery")
        require(run["configuration_sha256"]==configurations,"Fresh model configuration differs")
        require(run["reference"]==reference and run["output"]!=reference,"Independent reference differs or replays output")
        require(run["output"] not in outputs,"Output reused between runs")
        outputs.add(run["output"])
        require(run["thread_environment"]["NUMBA_NUM_THREADS"]=="48","Unexpected CPU capacity")
        require(not run["profiled_not_performance_evidence"],"Diagnostic included as timing")
        require(run["exact"]["decision_columns_exact"] and run["matrices"]["exact"] and run["summary_reports"]["exact"],"Fresh output audit failed")
        require(len(run["components"])==34 and math.isfinite(run["process_wall_seconds"]) and run["process_wall_seconds"]>0,"Invalid complete clock")
    require(all(sequence["source_sha256"].get(k)==v for k,v in source.items()),"Long and fresh series use different production source")
    pairs = []
    for trial in range(1,7):
        group = [r for r in fresh["runs"] if r["trial"]==trial]
        require([r["mode"] for r in group]==(["gpu","candidate"] if trial%2 else ["candidate","gpu"]),"Fresh order not balanced")
        old,new = [next(r for r in group if r["mode"]==mode) for mode in ("gpu","candidate")]
        pairs.append(dict(trial=trial,old_wall=old["process_wall_seconds"],new_wall=new["process_wall_seconds"],
                          old_charged=old["charged_total_seconds"],new_charged=new["charged_total_seconds"]))
    grouped = {"phase62":[r for r in fresh["runs"] if r["mode"]=="gpu"],
               "phase63":[r for r in fresh["runs"] if r["mode"]=="candidate"],"regular48":cpu["runs"]}
    fields = ("process_wall_seconds","charged_total_seconds","model_steps_seconds","validation_seconds","prewarm_seconds")
    medians = {name:{field:statistics.median(r[field] for r in runs) for field in fields} for name,runs in grouped.items()}
    components = []
    names = list(grouped["phase63"][0]["components"])
    require(all(set(r["components"])==set(names) for runs in grouped.values() for r in runs),"Component inventory differs")
    for name in names:
        row = {key:statistics.median(r["components"][name] for r in runs) for key,runs in grouped.items()}
        row.update(component=name,cpu_over_phase63=row["regular48"]/row["phase63"] if row["phase63"] else None)
        components.append(row)
    new,old,regular = [medians[key]["process_wall_seconds"] for key in ("phase63","phase62","regular48")]
    matched = matched_default_scenarios(sequence)
    wins = all(p["new_wall"]<p["old_wall"] and p["new_charged"]<p["old_charged"] for p in pairs)
    result = dict(status=("replicated_improvement" if wins else "outputs_qualified_performance_not_replicated")+
                  ("_target_met" if new<70 else "_target_not_met"),medians=medians,pairs=pairs,components=components,
                  all_pairs_faster=wins,median_wall_under_70=new<70,median_wall_under_65=new<65,
                  cpu_over_phase63=regular/new,phase62_over_phase63=old/new,source_sha256=source,
                  sequence_status=sequence["status"],sequence_model_runs=sequence["model_runs"],
                  matched_preparation_default_fresh=matched,
                  scope="Fresh warm-installed workstation processes; not a clean install or GPU-only arithmetic claim. CPU preparation equivalence is separately tested in the 80-model sequence.")
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fresh_path,cpu_path,sequence_path,Path(__file__))}
    target = RESULTS/"phase63-complete-comparison.json"
    require(not target.exists(),"Refusing to replace a delivered comparison")
    target.write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# Phase 63: complete CPU/hybrid comparison","",result["scope"],"",
        "Fresh wall clock includes process startup, complete model work, prewarming and live validation. External output audits are outside that clock.","",
        "| Clock | Regular CPU48 | Phase 62 hybrid | Phase 63 hybrid |","|---|---:|---:|---:|"]
    for field,label in (("process_wall_seconds","Complete process seconds"),("charged_total_seconds","Charged service seconds")):
        lines.append(f"| {label} | {medians['regular48'][field]:.2f} | {medians['phase62'][field]:.2f} | {medians['phase63'][field]:.2f} |")
    lines += ["",f"Regular CPU48 / latest hybrid: {regular/new:.3f}x. Previous / latest hybrid: {old/new:.3f}x.",
        f"Under-70-second fresh target: {'met' if new<70 else 'NOT met'}. Under-65 stretch: {'met' if new<65 else 'NOT met'}.","",
        "| Model component | CPU48 seconds | Phase 62 seconds | Phase 63 seconds | CPU / latest |","|---|---:|---:|---:|---:|"]
    for row in components:
        ratio = f"{row['cpu_over_phase63']:.2f}x" if row["cpu_over_phase63"] is not None else "not resolved"
        lines.append(f"| {row['component']} | {row['regular48']:.2f} | {row['phase62']:.2f} | {row['phase63']:.2f} | {ratio} |")
    lines += ["","Component logs are rounded to 0.1 seconds before medians. Their median rows need not sum to the median full clock. A component ratio is not a standalone kernel speedup.",
              "","Phase 63 includes reusable CPU expression preparation, parsed specification copies, and RSS-only in-step memory tracing with chunking/training disabled. These are disclosed service optimizations; equivalent options are given to CPU in the separate repeated-scenario controls.",""]
    lines += ["## Matched preparation: default 50,000-household scenario only","",matched["scope"],"",
              "| Engine | Fresh worker median seconds | Measured default-A processes |",
              "|---|---:|---:|",
              f"| Regular CPU with Phase 63 preparation | {matched['medians']['regular']:.2f} | 8 |",
              f"| Phase 63 hybrid with the same preparation options | {matched['medians']['candidate']:.2f} | 8 |","",
              f"Matched-preparation CPU / hybrid: {matched['cpu_over_hybrid']:.3f}x. Neither row mixes in the smaller 10,000-household scenario.","",
              "Equal preparation does not mean identical implementations everywhere: the hybrid also contains earlier CPU-side, data-layout and output-writing optimizations. This remains a complete-system comparison, not an isolated GPU-hardware experiment.",""]
    (ROOT/"docs").mkdir(exist_ok=True)
    (ROOT/"docs/phase63-component-comparison.md").write_text("\n".join(lines),newline="\n")
    print(json.dumps(dict(status=result["status"],medians=medians),indent=2))


if __name__=="__main__":
    main()
