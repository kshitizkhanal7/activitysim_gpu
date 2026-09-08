"""Qualify repeated full-model improvements without conflating kernel and system gains."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def summarize(controls, matched, *, target=105.0):
    for document in (controls, matched):
        if not document.get("complete"):
            raise ValueError("Incomplete timing series")
        for run in document["runs"]:
            if run.get("profiled_not_performance_evidence"):
                raise ValueError("Profiling is not performance evidence")
            if not run["exact"]["success"] or not run["exact"]["decision_columns_exact"]:
                raise ValueError("Exact outputs are required")
            if len(run["components"]) != 34:
                raise ValueError("All 34 steps must be timed")
    grouped = {mode:[r for r in controls["runs"] if r["mode"] == mode]
               for mode in ("regular", "cpu")}
    grouped.update({mode:[r for r in matched["runs"] if r["mode"] == mode]
                    for mode in ("gpu", "candidate")})
    if any(not rows for rows in grouped.values()):
        raise ValueError("All four comparison arms are required")
    if ({r["trial"] for r in grouped["gpu"]} != {r["trial"] for r in grouped["candidate"]}
            or len(grouped["gpu"]) != len(grouped["candidate"])):
        raise ValueError("Baseline and candidate trial identities differ")
    pairs = []
    fingerprints = []
    for trial in sorted({r["trial"] for r in grouped["gpu"]}):
        old = [r for r in grouped["gpu"] if r["trial"] == trial]
        new = [r for r in grouped["candidate"] if r["trial"] == trial]
        if len(old) != 1 or len(new) != 1:
            raise ValueError("A trial must contain one baseline and one candidate")
        old, new = old[0], new[0]
        if old["order"] != new["order"] or old["order"] not in (["gpu","candidate"], ["candidate","gpu"]):
            raise ValueError("Pair order is not a two-arm matched comparison")
        for run in (old, new):
            if not run.get("source_sha256"):
                raise ValueError("Matched measurements require frozen source fingerprints")
            fingerprints.append(run["source_sha256"])
        pairs.append({"trial":trial, "order":old["order"],
                      "baseline_seconds":old["charged_total_seconds"],
                      "candidate_seconds":new["charged_total_seconds"],
                      "saved_seconds":old["charged_total_seconds"]-new["charged_total_seconds"],
                      "wall_saved_seconds":old["process_wall_seconds"]-new["process_wall_seconds"]})
    if any(f != fingerprints[0] for f in fingerprints):
        raise ValueError("Source differs between matched trials")
    totals = {mode:{field:statistics.median(r[field] for r in rows)
                    for field in ("charged_total_seconds", "model_steps_seconds", "process_wall_seconds")}
              for mode, rows in grouped.items()}
    components = [{"component":step, **{mode:statistics.median(r["components"][step] for r in rows)
                                        for mode, rows in grouped.items()}}
                  for step in grouped["gpu"][0]["components"]]
    for row in components:
        row["regular_over_candidate"] = row["regular"]/row["candidate"] if row["candidate"] else None
        row["previous_gpu_minus_candidate"] = row["gpu"]-row["candidate"]
    positive = all(p["saved_seconds"] > 0 and p["wall_saved_seconds"] > 0 for p in pairs)
    balanced = (sum(p["order"].index("gpu") < p["order"].index("candidate") for p in pairs)*2 == len(pairs))
    repeated = len(pairs) >= 4 and positive and balanced
    target_met = totals["candidate"]["charged_total_seconds"] < target
    old, new = totals["gpu"]["charged_total_seconds"], totals["candidate"]["charged_total_seconds"]
    cpu_control_gpu = [r["charged_total_seconds"] for r in controls["runs"] if r["mode"] == "gpu"]
    return {"phase":58, "status":("replicated_improvement_target_met" if target_met else
            "replicated_improvement_target_not_met") if repeated else "not_qualified",
        "gates":{"exact_outputs_in_all_runs":True, "at_least_four_balanced_pairs":len(pairs)>=4 and balanced,
                 "every_pair_improves_model_and_wall":positive, "source_frozen_across_pairs":True,
                 "candidate_median_below_target":target_met},
        "target_seconds":target, "pairs":pairs, "medians":totals, "components":components,
        "regular_cpu_over_latest_gpu":totals["regular"]["charged_total_seconds"]/new,
        "previous_gpu_over_latest_gpu":old/new, "whole_model_time_reduction_percent":100*(1-new/old),
        "minimum_pair_saving_seconds":min(p["saved_seconds"] for p in pairs),
        "compact_cpu_hardware_control":{"cpu_median":totals["cpu"]["charged_total_seconds"],
            "gpu_median":statistics.median(cpu_control_gpu),
            "interpretation":"Two full-model pairs cannot establish a material GPU advantage for compaction alone."},
        "limitations":["Regular CPU controls precede the matched Phase57/58 series; no contemporaneous per-pair regular-CPU arm.",
            "Model-step CSV timings are rounded to tenths of a second; process wall measured separately.",
            "Fresh processes reuse existing compiler and filesystem caches; these are not cold-machine measurements.",
            "Repeated deterministic runs measure performance variability, not new populations or random seeds.",
            "Changed-seed and changed-input unit tests qualify targeted new kernels, not the entire benchmark-specific hybrid.",
            "Earlier mandatory-scheduling reference artifacts remain; this is not a general scenario-ready GPU-only model.",
            "The live mode CDF boundary guard is an engineering envelope, not a universal floating-point theorem."],
        "source_sha256":fingerprints[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", type=Path, default=ROOT/"benchmark-results/phase58-controls2-summary.json")
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.controls.read_text()), json.loads(args.matched.read_text()))
    # Recheck each candidate's complete implementation gates from the saved run reports.
    reports = []
    scheduling_counts = {}
    for run in json.loads(args.matched.read_text())["runs"]:
        command = run["command"]
        # Archived commands retain the original host's absolute paths. Locate
        # the committed reports relative to this checkout for portable rechecks.
        path = ROOT / "benchmark-results" / Path(command[command.index("--report")+1].replace("\\", "/")).name
        report = json.loads(path.read_text())
        if not all(report["proof_gates"].values()):
            raise ValueError(f"Failed implementation proof: {path}")
        scheduling_counts.setdefault(run["trial"], {})[run["mode"]] = {
            k:report["phase35_trip_scheduling"][k] for k in ("chooser_rows", "failed_choices")}
        if run["mode"] == "candidate":
            reports.append({"report":str(path.relative_to(ROOT)),
                "runtime":report["phase58_trip_runtime"], "trip_scheduling":report["phase35_trip_scheduling"]})
    result["candidate_runtime_evidence"] = reports
    if any(pair["gpu"] != pair["candidate"] for pair in scheduling_counts.values()):
        raise ValueError("Scheduling chooser/failure counts differ between paired runs")
    result["paired_scheduling_counts"] = scheduling_counts
    result["gates"]["paired_scheduling_draw_and_failure_counts_match"] = True
    result["gates"]["all_implementation_proof_gates_pass"] = True
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status", "gates", "medians", "previous_gpu_over_latest_gpu")}))
    return 0 if result["status"] == "replicated_improvement_target_met" else 2


if __name__ == "__main__":
    raise SystemExit(main())
