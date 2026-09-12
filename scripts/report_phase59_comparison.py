"""Publish all 34 step medians and an explicitly unpaired regular-CPU comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_phase59_reports import verify as verify_reports


def report(qualified, regular):
    if not regular.get("complete") or len(regular["runs"]) < 2:
        raise ValueError("Two completed fresh regular CPU controls required")
    runs = regular["runs"]
    names = list(qualified["component_medians"])
    if len(names) != 34:
        raise ValueError("All 34 model steps required")
    reports = {}
    for run in runs:
        if (run["mode"] != "regular" or run.get("profiled_not_performance_evidence")
                or run["source_sha256"] != qualified["source_sha256"]
                or run["configuration_sha256"] != qualified["configuration_sha256"]):
            raise ValueError("Invalid regular control source/mode")
        if (not run["exact"]["decision_columns_exact"] or not run["matrices"]["exact"]
                or set(run["components"]) != set(names)):
            raise ValueError("Incomplete regular control output")
        for metric in ("process_wall_seconds", "charged_total_seconds"):
            if not math.isfinite(run[metric]) or run[metric] <= 0:
                raise ValueError("Invalid regular timing")
        if not all(math.isfinite(x) and x >= 0 for x in run["components"].values()):
            raise ValueError("Invalid regular component timing")
        reports[run["output"]] = verify_reports(Path(run["reference"]), Path(run["output"]))
    totals = dict(qualified["medians"])
    totals["regular"] = {key:statistics.median(r[key] for r in runs)
                         for key in ("charged_total_seconds", "process_wall_seconds")}
    components = []
    for name in names:
        row = {"step":name, **qualified["component_medians"][name]}
        row["regular"] = statistics.median(r["components"][name] for r in runs)
        row["regular_over_candidate"] = row["regular"]/row["candidate"] if row["candidate"] else None
        components.append(row)
    return {"scope":"warm-cache fresh processes on one workstation; 6 balanced Phase58/59 pairs, then 2 unpaired regular CPU controls; not strongest possible CPU rewrite",
            "phase59_status":qualified["status"], "totals":totals, "components":components,
            "regular_cpu_summary_reports":reports,
            "regular_over_candidate":{key:totals["regular"][key]/totals["candidate"][key]
                                      for key in totals["regular"]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--regular", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = report(json.loads(args.qualification.read_text()), json.loads(args.regular.read_text()))
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.qualification, args.regular, Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Phase 59: complete CPU/accelerated timing comparison", "", result["scope"], "",
             "Seconds; medians are computed separately for each row. Upstream step timings are rounded to tenths of a second; displayed extra decimal places do not imply finer precision. Component medians do not necessarily add to the full-model median.", "",
             "| Clock | Regular CPU | Phase 58 hybrid | Phase 59 hybrid | CPU / Phase 59 |",
             "|---|---:|---:|---:|---:|"]
    for key in ("process_wall_seconds", "charged_total_seconds"):
        t = result["totals"]
        lines.append(f"| {key} | {t['regular'][key]:.3f} | {t['gpu'][key]:.3f} | {t['candidate'][key]:.3f} | {result['regular_over_candidate'][key]:.3f}x |")
    lines += ["", "## Every model step", "",
              "| Step | Regular CPU (s) | Phase 58 (s) | Phase 59 (s) | CPU / Phase 59 |",
              "|---|---:|---:|---:|---:|"]
    for r in result["components"]:
        ratio = f"{r['regular_over_candidate']:.3f}x" if r['regular_over_candidate'] is not None else "not defined"
        lines.append(f"| {r['step']} | {r['regular']:.3f} | {r['gpu']:.3f} | {r['candidate']:.3f} | {ratio} |")
    lines += ["", "Individual step ratios are descriptive, not separately randomized component benchmarks. Unchanged CPU steps can differ because of timing noise. Full-model improvement combines GPU kernels, CPU algorithm changes, data handling and output compression. The CPU baseline is pinned regular single-process ActivitySim with numerical libraries limited to one thread, not a purpose-built optimized CPU model. The Phase 59 matrix writer uses four CPU compression workers.", "",
              "Charged time adds recorded validation/prewarming to the model steps. Launch-to-exit also includes process overhead. Independent output verification is outside these clocks. All modeled decisions and matrix values pass; published summary report values pass the declared exact-value contract. Diagnostic logsums retain explicit bounds.", ""]
    args.markdown.write_text("\n".join(lines))
    print(json.dumps({"totals":result["totals"], "regular_over_candidate":result["regular_over_candidate"]}, indent=2))
