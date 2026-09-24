"""Guarded full-model scale feasibility, not a repeated performance qualification.

Runs the ordinary CPU reference and the hybrid sequentially, then retains the
existing decision/matrix/report audits. Monitoring is disclosed instrumentation.
Only the subprocess tree created by this invocation can be stopped by guards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import psutil

from phase63_commands import ROOT, PYTHON, output_directory
from run_phase58_comparison import source_fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--households",type=int,choices=(100000,250000),required=True)
    parser.add_argument("--features",default="expressions,inputs,location_boundary")
    parser.add_argument("--reference",type=Path,help="Reuse an existing independent CPU reference for diagnostic candidate-only iteration")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    output_directory(f"o-p58-{args.tag}-candidate-1")
    target=ROOT/f"benchmark-results/phase64-{args.tag}-scale.json"
    if target.exists():
        raise FileExistsError(target)
    free=shutil.disk_usage(ROOT).free
    available=psutil.virtual_memory().available
    if free<12*1024**3 or available<12*1024**3:
        raise ValueError("At least 12 GiB available disk and host RAM required to attempt scale")
    command=[str(PYTHON),str(ROOT/"scripts/run_phase58_comparison.py"),"--tag",args.tag,
        "--phase64","--phase64-features",args.features,"--phase63-features","plans,files,rss",
        "--households",str(args.households),"--modes","regular,candidate","--repetitions","1",
        "--regular-numba-threads","48","--output-root",str(ROOT/"phase63-runs")]
    if args.reference:
        if not (args.reference/"pipeline.parquetpipeline").is_dir():
            raise FileNotFoundError("Independent CPU reference absent")
        command[command.index("--modes")+1]="candidate"
        command += ["--scenario-baseline",str(args.reference.resolve())]
    source=source_fingerprint()
    samples=[]
    reason=None
    started=time.perf_counter()
    stdout=target.with_suffix(".stdout.log")
    stderr=target.with_suffix(".stderr.log")
    with stdout.open("w") as out,stderr.open("w") as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        owner=psutil.Process(child.pid)
        low_memory=0
        while child.poll() is None:
            try:
                processes=[owner,*owner.children(recursive=True)]
                rss=[]
                for process in processes:
                    try:
                        rss.append(process.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                memory=psutil.virtual_memory()
                disk=shutil.disk_usage(ROOT).free
                samples.append(dict(elapsed_seconds=time.perf_counter()-started,
                    maximum_child_rss_bytes=max(rss,default=0),host_available_bytes=memory.available,
                    disk_free_bytes=disk))
                low_memory=low_memory+1 if memory.available<2*1024**3 else 0
                if low_memory>=3 or disk<8*1024**3:
                    reason="host_available_below_2GiB_three_samples" if low_memory>=3 else "disk_below_8GiB"
                    for process in reversed(processes):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    child.wait()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(1)
        exit_code=child.wait()
    elapsed=time.perf_counter()-started
    unchanged=source==source_fingerprint()
    summary=ROOT/f"benchmark-results/phase58-{args.tag}-summary.json"
    doc=json.loads(summary.read_text()) if summary.exists() else {}
    complete=exit_code==0 and doc.get("complete",False) and unchanged
    payload=dict(status="outputs_qualified_scale_feasibility" if complete else "failed_retained",
        households=args.households,command=command,exit_code=exit_code,guard_stop_reason=reason,
        monitored_campaign_seconds=elapsed,initial_disk_free_bytes=free,initial_host_available_bytes=available,
        observed_maximum_child_rss_bytes=max((s["maximum_child_rss_bytes"] for s in samples),default=0),
        minimum_host_available_bytes=min((s["host_available_bytes"] for s in samples),default=available),
        samples=samples,source_sha256=source,source_unchanged=unchanged,
        diagnostic_not_repeated_performance_qualification=True,
        memory_scope="One-second maximum individual owned-child RSS; not aggregate USS, GPU memory or guaranteed peak.",
        reference_scope=("Candidate-only diagnostic checked against existing independent CPU reference; no contemporaneous CPU timing." if args.reference else
                         "New independent ordinary CPU model, then candidate; CPU self-audit is not external GPU proof."),
        summary=str(summary),evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),stdout,stderr,summary) if p.exists()})
    target.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({k:payload[k] for k in ("status","households","exit_code","guard_stop_reason",
        "observed_maximum_child_rss_bytes","minimum_host_available_bytes")}))
    if not complete:
        raise SystemExit(1)


if __name__=="__main__":
    main()
