"""Require complete, balanced Phase 60/61 evidence without hiding target misses."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_phase60 import audit as audit_phase60
from qualify_phase59 import require


def audit(run,candidate):
    proof = audit_phase60(run,True)  # Both versions retain all Phase 60 changes.
    require(("--phase61-features" in run["command"]) == candidate,"wrong Phase 61 backend")
    report = proof["phase61_shared_inputs"]
    require(report["enabled"] is candidate,"Phase 61 report differs from command")
    if candidate:
        require(report.get("diagnostic_capture") is False and "--phase61-capture-inputs" not in run["command"],"captured run is not performance evidence")
        require(set(report["features"]) == {"skims","timetable","tour_modes","entities","normals","uniforms","labels","packing"},"incomplete Phase 61 features")
        require(len(report["events"]) == 34 and all(e["saved_answers_read"] is False for e in report["events"]),"incomplete live step audit")
        require(set(e["step"] for e in report["events"]) == set(run["components"]),"wrong live step inventory")
        require(report["skim_calls"] > 0 and report["availability_events"],"no preparation consumers")
        require(run["thread_environment"].get("OMP_WAIT_POLICY")=="PASSIVE","wrong selected worker wait policy")
        require(all(e["backend"]=="cpu" and e["cpu_threads"]==24 for e in report["availability_events"]),"wrong selected timetable backend")
        require({e["table"] for e in report["shared_input_events"]} == {"tours","trips"},"missing live entity consumers")
        require(all(e["columns"] and e["rows"] > 0 for e in report["shared_input_events"]),"empty entity use")
        require(report["tour_mode_events"] and report["tour_rng_events"] and report["normal_events"] and report["uniform_events"],"missing mode or random execution")
    return proof


def qualify(comparison,scenarios):
    require(comparison.get("complete") is True and len(comparison["runs"]) == 12,"six completed pairs required")
    runs = comparison["runs"]
    first = runs[0]
    require(first["source_sha256"] and first["configuration_sha256"],"missing fingerprints")
    require(all(r["source_sha256"] == first["source_sha256"] and r["configuration_sha256"] == first["configuration_sha256"] for r in runs),"source/config changed")
    pairs = []
    for trial in range(1,7):
        group = [r for r in runs if r["trial"] == trial]
        order = ["gpu","candidate"] if trial%2 else ["candidate","gpu"]
        require([r["mode"] for r in group] == order,"missing or unbalanced pair")
        require(all(r["order"] == order and r["position"] == order.index(r["mode"]) for r in group),"invalid pair order")
        by_mode = {r["mode"]:r for r in group}
        old,new = by_mode["gpu"],by_mode["candidate"]
        a,b = audit(old,False),audit(new,True)
        for key in ("choosers","failures_before_final_coercion"):
            require([e[key] for e in a["phase58_trip_runtime"]["chain_events"]] ==
                    [e[key] for e in b["phase58_trip_runtime"]["chain_events"]],"retry trace changed")
        pairs.append({"trial":trial,"order":order,"old_wall":old["process_wall_seconds"],"new_wall":new["process_wall_seconds"],
                      "old_charged":old["charged_total_seconds"],"new_charged":new["charged_total_seconds"]})
    require(len(scenarios) == 3,"three changed scenarios required")
    checks = []
    for scenario in scenarios:
        require(scenario.get("complete") is True,"incomplete scenario")
        candidates = [r for r in scenario["runs"] if r["mode"] == "candidate"]
        require(len(candidates) == 1,"one scenario candidate required")
        r = candidates[0]
        require(r["source_sha256"] == first["source_sha256"] and r["configuration_sha256"] and r["scenario_overlay"],"scenario fingerprint differs or absent")
        audit(r,True)
        checks.append({"overlay":r["scenario_overlay"],"reference":r["reference"],"exact_decisions":True,
                       "exact_matrices":True,"summary_reports":r["summary_reports"]})
    require(len({c["overlay"] for c in checks}) == 3,"duplicate scenario")
    medians = {m:{k:statistics.median(r[k] for r in runs if r["mode"]==m)
                  for k in ("charged_total_seconds","process_wall_seconds")} for m in ("gpu","candidate")}
    wins = all(p["new_wall"]<p["old_wall"] and p["new_charged"]<p["old_charged"] for p in pairs)
    target = medians["candidate"]["process_wall_seconds"] < 75
    return {"status":"replicated_improvement_wall_target_met" if wins and target else "replicated_improvement_wall_target_not_met" if wins else "correctness_qualified_performance_not_replicated",
            "pairs":pairs,"medians":medians,"all_pairs_faster":wins,"median_wall_under_75":target,
            "median_wall_under_70":medians["candidate"]["process_wall_seconds"] < 70,
            "wall_speedup":medians["gpu"]["process_wall_seconds"]/medians["candidate"]["process_wall_seconds"],
            "charged_speedup":medians["gpu"]["charged_total_seconds"]/medians["candidate"]["charged_total_seconds"],
            "component_medians":{k:{m:statistics.median(r["components"][k] for r in runs if r["mode"]==m)
                                      for m in ("gpu","candidate")} for k in first["components"]},
            "source_sha256":first["source_sha256"],"configuration_sha256":first["configuration_sha256"],
            "scenario_checks":checks,"scope":"complete warm-cache model; exact outputs under existing contracts; hybrid CPU/GPU changes; finite replication"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--scenarios",type=Path,nargs=3,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    summaries = [json.loads(p.read_text()) for p in (args.comparison,*args.scenarios)]
    result = qualify(summaries[0],summaries[1:])
    paths = [args.comparison,*args.scenarios,Path(__file__),Path(__file__).with_name("qualify_phase60.py"),
             Path(__file__).with_name("qualify_phase59.py"),Path(__file__).with_name("verify_phase59_reports.py")]
    for summary in summaries:
        for r in summary["runs"]:
            if r["mode"] in {"gpu","candidate"}:
                paths.append(Path(r["command"][r["command"].index("--report")+1]))
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["status"])
