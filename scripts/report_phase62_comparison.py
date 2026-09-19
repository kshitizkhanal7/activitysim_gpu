"""Publish every step, fresh CPU controls, and CPU wins in the kernel controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_phase59_comparison import report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("qualification","regular1","regular48","kernel-controls","output","markdown"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    qualified=json.loads(args.qualification.read_text())
    controls={}
    for threads,path in ((1,args.regular1),(48,args.regular48)):
        summary=json.loads(path.read_text())
        if any(r["thread_environment"].get("NUMBA_NUM_THREADS")!=str(threads) for r in summary["runs"]):
            raise ValueError("CPU capacity differs from label")
        if threads>1 and any("--fast" not in r["command"] for r in summary["runs"]):
            raise ValueError("CLI would reset CPU thread setting")
        controls[threads]=report(qualified,summary)
    kernels=json.loads(args.kernel_controls.read_text())
    if not kernels["timetable"]["exact"] or not kernels["normals"]["bit_exact"]:
        raise ValueError("Missing live primitive correctness")
    for p,h in kernels["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError("Kernel control source changed")
    totals={**qualified["medians"],"regular1":controls[1]["totals"]["regular"],"regular48":controls[48]["totals"]["regular"]}
    rows=[]
    for a,b in zip(controls[1]["components"],controls[48]["components"]):
        if a["step"]!=b["step"]:raise ValueError("Step alignment differs")
        rows.append(dict(step=a["step"],regular1=a["regular"],regular48=b["regular"],gpu=a["gpu"],candidate=a["candidate"]))
    result=dict(status=qualified["status"],totals=totals,components=rows,
        cpu_over_phase62={str(t):controls[t]["regular_over_candidate"] for t in controls},
        cpu_reports={str(t):controls[t]["regular_cpu_summary_reports"] for t in controls},
        kernel_controls=kernels,
        scope="Six balanced Phase61/62 pairs; two fresh regular CPU runs each at 1 and 48 Numba threads, separately measured; BLAS/OpenMP one thread. Not a strongest-possible CPU rewrite. Exact output contracts unchanged. Hybrid improvements, not GPU-only attribution.")
    paths=(args.qualification,args.regular1,args.regular48,args.kernel_controls,Path(__file__),Path(__file__).with_name("report_phase59_comparison.py"))
    result["evidence_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Phase 62: complete CPU/accelerated comparison","",result["scope"],"",
           "Separate medians; component clocks are rounded to tenths and row medians need not sum to total medians.","",
           "| Clock | CPU 1 | CPU 48 | Phase 61 | Phase 62 | CPU 48 / Phase 62 |","|---|---:|---:|---:|---:|---:|"]
    for key in ("process_wall_seconds","charged_total_seconds"):
        v=[totals[m][key] for m in ("regular1","regular48","gpu","candidate")]
        lines.append("| "+key+" | "+" | ".join(f"{x:.2f}" for x in v)+f" | {v[1]/v[3]:.3f}x |")
    lines += ["","## All 34 model steps","","| Step | CPU 1 (s) | CPU 48 (s) | Phase 61 (s) | Phase 62 (s) | CPU 48 / Phase 62 |","|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        v=[row[m] for m in ("regular1","regular48","gpu","candidate")]
        ratio=f"{v[1]/v[3]:.3f}x" if v[3] else "not defined"
        lines.append("| "+row["step"]+" | "+" | ".join(f"{x:.2f}" for x in v)+" | "+ratio+" |")
    lines += ["","## Strong equivalent CPU primitive controls","",
        "These are isolated calculations on captured live inputs, not full-component or whole-model ratios. Greater than one favors GPU; less than one favors CPU.","",
        f"- Timetable, encoding and transfers included: CPU/GPU {kernels['timetable']['transfer_inclusive_cpu_over_gpu']:.3f}x.",
        f"- Tour-mode reducer, resident inputs: CPU/GPU {kernels['tour_modes']['resident_cpu_over_gpu']:.3f}x; including transfers: {kernels['tour_modes']['transfer_inclusive_cpu_over_gpu']:.3f}x.",
        "- Standard normals are a compiled CPU batching improvement, verified bit for bit against original NumPy on live seeded streams; not a GPU result.","",
        "The full pipeline retains GPU utility evaluation and live CPU numerical-boundary checks. Independent output audits are outside the clocks; execution, validation/prewarming, reports and process overhead remain charged. CPU 48 denotes Numba capacity, not 48 threads in every library.",""]
    args.markdown.write_text("\n".join(lines))
    print(json.dumps({"totals":totals,"cpu_over_phase62":result["cpu_over_phase62"]},indent=2))
