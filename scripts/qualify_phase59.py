"""Fail-closed Phase 59 qualification; targets never override correctness."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_phase59_reports import verify as verify_reports

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(run, *, live):
    require(run["exact"].get("decision_columns_exact") is True, "modeled decisions not exact")
    require(run.get("matrices", {}).get("exact") is True, "matrices not verified")
    run["summary_reports"] = verify_reports(Path(run["reference"]), Path(run["output"]))
    require(len(run["components"]) == 34, "incomplete model steps")
    require(all(math.isfinite(v) and v >= 0 for v in run["components"].values()), "invalid component timing")
    diagnostics = run["exact"].get("diagnostic_columns", {})
    require(bool(diagnostics), "diagnostic error bounds missing")
    for item in diagnostics.values():
        require(math.isfinite(item["max_abs"]) and 0 <= item["max_abs"] <= item["gate"], "diagnostic error bound exceeded")
    require(not run.get("profiled_not_performance_evidence"), "profiling is not timing evidence")
    for name in ("charged_total_seconds", "process_wall_seconds"):
        require(math.isfinite(run[name]) and run[name] > 0, "invalid timing")
    command = run["command"]
    require("--phase58-trip-runtime" in command, "Phase 58 runtime absent")
    if live:
        require("--phase59-live-mandatory" in command and "--phase59-device-retries" in command, "wrong candidate command")
        require("--phase59-sparse-matrices" in command, "matrix writer candidate absent")
    else:
        require(not any(flag.startswith("--phase59-") for flag in command), "control contains Phase 59 changes")
    report_name = command[command.index("--report")+1].replace("\\", "/").rsplit("/", 1)[-1]
    proof = json.loads((ROOT / "benchmark-results" / report_name).read_text())
    require(proof.get("proof_gates") and all(v is True for v in proof["proof_gates"].values()), "implementation gate failed")
    if live:
        mandatory = proof.get("phase59_live_mandatory", {})
        require(mandatory.get("captured_input_artifact_read") is False, "mandatory input artifact dependency")
        require(mandatory.get("saved_boundary_answers_used") is False, "saved boundary answer dependency")
        rechecks = mandatory.get("live_cpu_logsum_rechecks")
        require(isinstance(rechecks, list), "complete live CPU boundary rechecks missing")
        require(sum(e["choosers"] for e in rechecks) == mandatory["live_boundary_cpu_rows"], "boundary recheck coverage differs")
        require(all(e["additional_random_draws"] == 0 for e in rechecks), "boundary recheck consumed extra randomness")
        trip = proof.get("phase58_trip_runtime", {})
        require(trip.get("device_retry_controller") is True, "device retry controller absent")
        require(trip.get("entity_store", {}).get("contract") == "phase59-keyed-entity-columns-v1", "entity contract absent")
        require(set(trip["entity_store"]["tables"]) >= {"persons", "tours", "trips"}, "incomplete entity store")
    return proof


def qualify(comparison, scenarios):
    require(comparison.get("complete") is True, "comparison incomplete")
    runs = comparison["runs"]
    require(len(runs) >= 12 and len(runs) % 2 == 0, "six full pairs required")
    require(all(r["source_sha256"] == runs[0]["source_sha256"] for r in runs), "source changed between runs")
    require(bool(runs[0].get("configuration_sha256")) and all(
        r.get("configuration_sha256") == runs[0]["configuration_sha256"] for r in runs), "configuration changed between timed runs")
    grouped = {}
    for run in runs:
        require(run["mode"] in {"gpu", "candidate"}, "unexpected control mode")
        pair = grouped.setdefault(run["trial"], {})
        require(run["mode"] not in pair, "duplicate pair member")
        pair[run["mode"]] = run
    require(sorted(grouped) == list(range(1, len(grouped)+1)), "noncontiguous trials")
    pairs = []
    for trial, pair in sorted(grouped.items()):
        require(set(pair) == {"gpu", "candidate"}, "missing pair member")
        order = ["gpu", "candidate"] if trial % 2 else ["candidate", "gpu"]
        for mode in order:
            require(pair[mode]["order"] == order and pair[mode]["position"] == order.index(mode), "unbalanced order")
        old, new = pair["gpu"], pair["candidate"]
        old_proof, new_proof = audit_run(old, live=False), audit_run(new, live=True)
        old_chain = old_proof["phase58_trip_runtime"]["chain_events"]
        new_chain = new_proof["phase58_trip_runtime"]["chain_events"]
        for key in ("choosers", "failures_before_final_coercion"):
            require([e[key] for e in old_chain] == [e[key] for e in new_chain], "per-attempt retry trace changed")
        pairs.append({"trial":trial, "order":order,
            "old_wall":old["process_wall_seconds"], "new_wall":new["process_wall_seconds"],
            "old_charged":old["charged_total_seconds"], "new_charged":new["charged_total_seconds"]})
    scenario_checks = []
    require(len(scenarios) >= 3, "three changed scenarios required")
    for scenario in scenarios:
        require(scenario.get("complete") is True, "scenario incomplete")
        candidates = [r for r in scenario["runs"] if r["mode"] == "candidate"]
        require(len(candidates) == 1, "one candidate per changed scenario required")
        run = candidates[0]
        require(bool(run.get("configuration_sha256")), "scenario configuration hashes missing")
        require(run["source_sha256"] == runs[0]["source_sha256"], "scenario source differs from timed source")
        audit_run(run, live=True)
        require(run.get("scenario_overlay") is not None, "scenario overlay missing")
        scenario_checks.append({"overlay":run["scenario_overlay"], "reference":run["reference"],
                                "exact_decisions":True, "exact_matrices":True,
                                "summary_reports":run["summary_reports"]})
    require(len({s["overlay"] for s in scenario_checks}) == len(scenario_checks), "duplicate scenario")
    medians = {mode:{field:statistics.median(r[field] for r in runs if r["mode"] == mode)
                     for field in ("process_wall_seconds", "charged_total_seconds")}
               for mode in ("gpu", "candidate")}
    all_faster = all(p["new_wall"] < p["old_wall"] and p["new_charged"] < p["old_charged"] for p in pairs)
    all_under_target = all(p["new_wall"] < 100 for p in pairs)
    status = ("replicated_improvement_wall_target_met" if all_faster and all_under_target else
              "replicated_improvement_wall_target_not_met" if all_faster else "correctness_qualified_performance_not_replicated")
    components = {name:{mode:statistics.median(r["components"][name] for r in runs if r["mode"] == mode)
                        for mode in ("gpu", "candidate")} for name in runs[0]["components"]}
    return {"status":status, "pairs":pairs, "medians":medians, "component_medians":components,
            "scenario_checks":scenario_checks, "all_pairs_faster":all_faster, "all_wall_runs_under_100":all_under_target,
            "summary_reports":{r["output"]:r["summary_reports"] for r in runs},
            "wall_speedup":medians["gpu"]["process_wall_seconds"]/medians["candidate"]["process_wall_seconds"],
            "charged_speedup":medians["gpu"]["charged_total_seconds"]/medians["candidate"]["charged_total_seconds"],
            "claim_scope":"matched complete public benchmark plus three changed-scenario output comparisons; not a universal floating-point theorem",
            "source_sha256":runs[0]["source_sha256"],
            "configuration_sha256":runs[0]["configuration_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(json.loads(args.comparison.read_text()), [json.loads(p.read_text()) for p in args.scenarios])
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.comparison, *args.scenarios)}
    result["qualification_code_sha256"] = {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(), ROOT/"scripts/verify_phase59_reports.py")}
    result["implementation_report_sha256"] = {}
    for summary in (args.comparison, *args.scenarios):
        for run in json.loads(summary.read_text())["runs"]:
            if run["mode"] not in {"gpu", "candidate"}:
                continue
            command = run["command"]
            report_name = command[command.index("--report")+1].replace("\\", "/").rsplit("/", 1)[-1]
            p = ROOT/"benchmark-results"/report_name
            result["implementation_report_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(result["status"])
