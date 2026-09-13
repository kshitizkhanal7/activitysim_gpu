"""Phase 60 requires full-model replication and all Phase 59 live output gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_phase59 import audit_run, require


def audit(run, candidate):
    proof = audit_run(run, live=True)
    require(("--phase60-preparation" in run["command"]) == candidate, "wrong preparation backend")
    require(not proof.get("phase60_frequency_control",{}).get("instrumented_not_performance"), "instrumented thread control")
    require(proof["phase60_preparation"]["enabled"] is candidate, "preparation report differs")
    if candidate:
        events = proof["phase60_preparation"]["events"]
        require(len(events) == 34 and {e["step"] for e in events} == set(run["components"]), "missing preparation steps")
        require(all(e["saved_choices_read"] is False for e in events), "saved answer dependency")
        frequency = [e for e in events if e["step"] == "non_mandatory_tour_frequency"]
        requested = int(run["command"][run["command"].index("--phase60-frequency-threads")+1])
        require(frequency[0]["frequency_threads"] == requested, "frequency thread mask differs")
    require(run["thread_environment"].get("NUMBA_NUM_THREADS") == "48", "pool capacity differs")
    require(run["thread_environment"].get("CHOICEFORGE_NUMBA_INITIAL_THREADS") == "1", "initial thread mask differs")
    return proof


def qualify(comparison, scenarios):
    require(comparison.get("complete") is True, "comparison incomplete")
    runs = comparison["runs"]
    require(len(runs) == 12, "six complete pairs required")
    first = runs[0]
    for run in runs:
        require(run["source_sha256"] == first["source_sha256"] and bool(first["source_sha256"]), "source changed")
        require(run["configuration_sha256"] == first["configuration_sha256"] and bool(first["configuration_sha256"]), "configuration changed")
    pairs = []
    for trial in range(1,7):
        group = [r for r in runs if r["trial"] == trial]
        order = ["gpu","candidate"] if trial % 2 else ["candidate","gpu"]
        require(len(group) == 2 and [r["mode"] for r in group] == order, "unbalanced pair")
        require(all(r["order"] == order and r["position"] == order.index(r["mode"]) for r in group), "invalid pair metadata")
        by_mode = {r["mode"]:r for r in group}
        old, new = by_mode["gpu"], by_mode["candidate"]
        a, b = audit(old,False), audit(new,True)
        for key in ("choosers","failures_before_final_coercion"):
            require([e[key] for e in a["phase58_trip_runtime"]["chain_events"]] ==
                    [e[key] for e in b["phase58_trip_runtime"]["chain_events"]], "retry trace changed")
        pairs.append({"trial":trial,"order":order,"old_wall":old["process_wall_seconds"],
                      "new_wall":new["process_wall_seconds"],"old_charged":old["charged_total_seconds"],
                      "new_charged":new["charged_total_seconds"]})
    require(len(scenarios) == 3, "three changed scenarios required")
    scenario_checks = []
    for scenario in scenarios:
        require(scenario.get("complete") is True,"scenario incomplete")
        candidates = [r for r in scenario["runs"] if r["mode"] == "candidate"]
        require(len(candidates) == 1,"one scenario candidate required")
        run = candidates[0]
        require(run["source_sha256"] == first["source_sha256"],"scenario source differs")
        require(bool(run["configuration_sha256"]) and bool(run["scenario_overlay"]),"scenario config absent")
        audit(run,True)
        scenario_checks.append({"overlay":run["scenario_overlay"],"reference":run["reference"],
                                "exact_decisions":True,"exact_matrices":True,"summary_reports":run["summary_reports"]})
    require(len({s["overlay"] for s in scenario_checks}) == 3,"duplicate scenarios")
    medians = {m:{k:statistics.median(r[k] for r in runs if r["mode"] == m)
                  for k in ("charged_total_seconds","process_wall_seconds")} for m in ("gpu","candidate")}
    wins = all(p["new_wall"] < p["old_wall"] and p["new_charged"] < p["old_charged"] for p in pairs)
    target = all(p["new_wall"] < 100 for p in pairs)
    return {"status":"replicated_improvement_wall_target_met" if wins and target else
                    "replicated_improvement_wall_target_not_met" if wins else "correctness_qualified_performance_not_replicated",
            "pairs":pairs,"medians":medians,"all_pairs_faster":wins,"all_wall_runs_under_100":target,
            "all_wall_runs_under_95":all(p["new_wall"] < 95 for p in pairs),
            "wall_speedup":medians["gpu"]["process_wall_seconds"]/medians["candidate"]["process_wall_seconds"],
            "charged_speedup":medians["gpu"]["charged_total_seconds"]/medians["candidate"]["charged_total_seconds"],
            "component_medians":{k:{m:statistics.median(r["components"][k] for r in runs if r["mode"]==m)
                                     for m in ("gpu","candidate")} for k in first["components"]},
            "source_sha256":first["source_sha256"],"configuration_sha256":first["configuration_sha256"],
            "scenario_checks":scenario_checks,"summary_reports":{r["output"]:r["summary_reports"] for r in runs},
            "claim_scope":"hybrid whole-step engineering; finite scenario replication, not universal equivalence or GPU-only attribution"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comparison",type=Path,required=True)
    p.add_argument("--scenarios",type=Path,nargs=3,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result = qualify(json.loads(args.comparison.read_text()),[json.loads(s.read_text()) for s in args.scenarios])
    paths = [args.comparison,*args.scenarios,Path(__file__),Path(__file__).with_name("qualify_phase59.py"),
             Path(__file__).with_name("verify_phase59_reports.py")]
    for summary in (args.comparison,*args.scenarios):
        for run in json.loads(summary.read_text())["runs"]:
            if run["mode"] in {"gpu","candidate"}:
                command = run["command"]
                paths.append(Path(command[command.index("--report")+1]))
    result["evidence_sha256"] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["status"])
