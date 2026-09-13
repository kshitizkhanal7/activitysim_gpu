"""Publish both ordinary and stronger CPU controls beside the matched hybrids."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_phase59_comparison import report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("qualification","regular1","regular48","output","markdown"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    qualified = json.loads(args.qualification.read_text())
    controls = {}
    for threads, path in ((1,args.regular1),(48,args.regular48)):
        summary = json.loads(path.read_text())
        if any(r["thread_environment"].get("NUMBA_NUM_THREADS") != str(threads) for r in summary["runs"]):
            raise ValueError("CPU thread capacity differs from label")
        if threads > 1 and any("--fast" not in r["command"] for r in summary["runs"]):
            raise ValueError("CLI would reset the CPU thread setting")
        controls[threads] = report(qualified,summary)
    totals = {**qualified["medians"],"regular1":controls[1]["totals"]["regular"],
              "regular48":controls[48]["totals"]["regular"]}
    rows = []
    for a,b in zip(controls[1]["components"],controls[48]["components"]):
        if a["step"] != b["step"]:
            raise ValueError("Component alignment differs")
        rows.append({"step":a["step"],"regular1":a["regular"],"regular48":b["regular"],
                     "gpu":a["gpu"],"candidate":a["candidate"]})
    result = {"status":qualified["status"],"totals":totals,"components":rows,
              "cpu_over_phase60":{str(t):controls[t]["regular_over_candidate"] for t in controls},
              "cpu_reports":{str(t):controls[t]["regular_cpu_summary_reports"] for t in controls},
              "scope":"six balanced Phase59/60 pairs; two fresh regular CPU runs each at 1 and 48 Numba threads, separately measured; BLAS/OpenMP numerical libraries remain one thread; not a strongest-possible CPU rewrite"}
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.qualification,args.regular1,args.regular48,Path(__file__),Path(__file__).with_name("report_phase59_comparison.py"))}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# Phase 60: all model steps and stronger CPU controls","",result["scope"],"",
             "Seconds, separate medians for each row. Upstream component clocks are rounded to tenths; extra displayed decimals are not additional measurement precision. Row medians need not sum to total medians.","",
             "| Clock | CPU 1 thread | CPU 48 threads | Phase 59 | Phase 60 | CPU 48 / Phase 60 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("process_wall_seconds","charged_total_seconds"):
        values = [totals[m][key] for m in ("regular1","regular48","gpu","candidate")]
        lines.append("| "+key+" | "+" | ".join(f"{v:.2f}" for v in values)+f" | {values[1]/values[3]:.3f}x |")
    lines += ["","## Every model step","",
              "| Step | CPU 1 (s) | CPU 48 (s) | Phase 59 (s) | Phase 60 (s) | CPU 48 / Phase 60 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row[k] for k in ("regular1","regular48","gpu","candidate")]
        ratio = f"{values[1]/values[3]:.3f}x" if values[3] else "not defined"
        lines.append("| "+row["step"]+" | "+" | ".join(f"{v:.2f}" for v in values)+" | "+ratio+" |")
    lines += ["","CPU 48 means 48 Numba threads, not 48 threads in every library or 48 independent model processes. The Phase 59 control has the same 48-thread pool capacity but masks it to one throughout. Phase 60 uses 48 only in non-mandatory tour frequency and restores one afterward. Both hybrids use four CPU matrix-compression workers. No reports or model steps are removed.","",
              "The accelerated system remains hybrid. This phase improves CPU preparation and enables an existing parallel CPU calculation; it does not establish a new GPU-only kernel speedup. Independent output auditing is outside the model clocks, while actual execution, validation and process overhead remain charged as documented.",""]
    args.markdown.write_text("\n".join(lines))
    print(json.dumps({"totals":totals,"cpu_over_phase60":result["cpu_over_phase60"]},indent=2))
