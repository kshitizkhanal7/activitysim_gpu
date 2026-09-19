"""Compare identical A-B-A model sequences in fresh versus persistent processes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_phase58_comparison import ROOT, PROJECT, RESULTS, PYTHON, REFERENCE, source_fingerprint, config_fingerprint
from phase62_batch_worker import file_digest, verify_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--mode",choices=("candidate","regular"),required=True)
    parser.add_argument("--execution",choices=("batch","fresh"),required=True)
    parser.add_argument("--skim-cache",choices=("none","content"),default="none")
    parser.add_argument("--series-trial",type=int,choices=(0,1,2),default=0)
    parser.add_argument("--series-position",type=int,choices=(0,1,2,3),default=0)
    parser.add_argument("--candidate-template",type=Path,default=RESULTS/"phase58-p62dev1-summary.json")
    args = parser.parse_args()
    if args.skim_cache == "content" and (args.execution != "batch" or args.mode != "candidate"):
        raise ValueError("Content skim pool requires the candidate batch worker")
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    prefix = f"phase62-{args.tag}"
    target = RESULTS/f"{prefix}.json"
    if target.exists():
        raise FileExistsError(target)
    env = {k:v for k,v in os.environ.items() if not k.startswith("CHOICEFORGE_")}
    settings = (ROOT/"scripts/run_phase32_full_model_ab.ps1").read_text()
    env.update(dict(re.findall(r'^\$env:([A-Z_0-9]+) = "([^"$]*)"$',settings,re.M)))
    env["PATH"] = str(PYTHON.parent)+os.pathsep+env["PATH"]
    env.update(NUMBA_NUM_THREADS="48",OMP_WAIT_POLICY="PASSIVE",
               CHOICEFORGE_STRICT_CUDA_CANDIDATE="0" if args.mode=="regular" else "1",
               CHOICEFORGE_STRICT_CUDA_MODE_CHOICE="0" if args.mode=="regular" else "1")
    if args.mode == "candidate":
        env["CHOICEFORGE_NUMBA_INITIAL_THREADS"] = "1"
    runs = []
    overlay = ROOT/"benchmark-data/configs_phase59_seed17"
    for index,name in enumerate(("A1","B","A2")):
        output = PROJECT/f"o-{prefix}-{name}"
        report = RESULTS/f"{prefix}-{name}-report.json"
        kernels = RESULTS/f"{prefix}-{name}-kernels"
        reference = PROJECT/"o-p58-p59finalseed17-regular-1" if name=="B" else REFERENCE
        for p in (output,report,kernels):
            if p.exists():
                raise FileExistsError(p)
        child_env = dict(env)
        if args.mode == "candidate":
            template = next(r for r in json.loads(args.candidate_template.read_text())["runs"]
                            if r["mode"] == "candidate")
            if "--phase62-features" not in template["command"]:
                raise ValueError("Candidate template is not Phase 62")
            command = list(template["command"])
            for flag,value in (("--output",output),("--report",report),
                ("--checkpoint",RESULTS/f"{prefix}-{name}-checkpoint.json"),("--kernel-reports",kernels),
                ("--reference-pipeline",reference/"pipeline.parquetpipeline")):
                command[command.index(flag)+1] = str(value)
            if name == "B":
                command[2:2] = ["--config-overlay",str(overlay),"--phase59-scenario-gates"]
            (kernels/"mode").mkdir(parents=True)
            child_env.update(CHOICEFORGE_PHASE17_REPORT_DIR=str(kernels),
                CHOICEFORGE_PHASE17_MODE_REPORT_DIR=str(kernels/"mode"),CHOICEFORGE_PHASE17_RUN_ID=f"{prefix}-{name}")
            cwd = ROOT
        else:
            command = [str(PYTHON.parent/"activitysim.exe"),"run","-c","configs_sh","-c","configs",
                       "-d","data_full","-o",str(output),"--households_sample_size","50000","--fast"]
            if name == "B":
                command[2:2] = ["-c",str(overlay)]
            cwd = PROJECT
        runs.append({"name":name,"mode":args.mode,"command":command,"cwd":str(cwd),
                     "environment":child_env,"output":str(output),"reference":str(reference),"report":str(report)})
    fingerprints = source_fingerprint()
    fingerprints.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__),ROOT/"scripts/phase62_batch_worker.py")})
    config = config_fingerprint(overlay)
    data_hashes = {str(p):file_digest(p) for p in sorted((PROJECT/"data_full").iterdir()) if p.is_file()}
    worker_result = RESULTS/f"{prefix}-worker.json"
    # Do not persist the inherited environment (which might contain credentials).
    # The private launch manifest has only task-specific environment overrides.
    for r in runs:
        r["environment"] = {k:v for k,v in r["environment"].items()
                            if k.startswith("CHOICEFORGE_") or k in {"NUMBA_NUM_THREADS","OMP_WAIT_POLICY",
                                "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"}}
    manifest = RESULTS/f"{prefix}-manifest.json"
    manifest.write_text(json.dumps({"runs":runs,"result":str(worker_result),"mode":args.mode,
                                   "skim_cache":args.skim_cache,"data_sha256":data_hashes},indent=2)+"\n")
    times = []
    started_at_ns = time.time_ns()
    with (RESULTS/f"{prefix}.stdout.log").open("w") as stdout,(RESULTS/f"{prefix}.stderr.log").open("w") as stderr:
        if args.execution == "batch":
            started = time.perf_counter()
            subprocess.run([str(PYTHON),str(ROOT/"scripts/phase62_batch_worker.py"),str(manifest)],
                           env=env,cwd=ROOT,stdout=stdout,stderr=stderr,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            elapsed = time.perf_counter()-started
            worker = json.loads(worker_result.read_text())
            assert worker["complete"] and len(worker["runs"])==3
            times = [r["scenario_wall_seconds"] for r in worker["runs"]]
        else:
            for r in runs:
                started = time.perf_counter()
                verify_files(data_hashes)
                subprocess.run(r["command"],cwd=r["cwd"],env={**env,**r["environment"]},stdout=stdout,stderr=stderr,
                               check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                times.append(time.perf_counter()-started)
            elapsed = sum(times)
            worker = None
    assert config_fingerprint(overlay) == config
    verify_files(data_hashes)
    for p,digest in fingerprints.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, f"Source changed: {p}"
    from verify_phase59_matrices import verify as matrices
    from verify_phase59_reports import verify as reports
    for r,seconds in zip(runs,times):
        output,reference = Path(r["output"]),Path(r["reference"])
        exact = RESULTS/f"{prefix}-{r['name']}-exact.json"
        subprocess.run([str(PYTHON),str(ROOT/"scripts/verify_phase15_outputs.py"),"--reference",str(reference),
                        "--candidate",str(output),"--output",str(exact)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        r.update(scenario_wall_seconds=seconds,exact=json.loads(exact.read_text()),
                 matrices=matrices(reference,output),summary_reports=reports(reference,output))
        with (output/"timing_log.csv").open() as stream:
            r["components"] = {row["model_name"]:float(row["seconds"]) for row in csv.DictReader(stream)}
        assert len(r["components"]) == 34
        if args.mode == "candidate":
            proof = json.loads(Path(r["report"]).read_text())
            assert all(proof["proof_gates"].values())
            assert proof["phase62_reusable_execution"]["enabled"]
    result = {"complete":True,"mode":args.mode,"execution":args.execution,"sequence":"A-B-A",
              "series_trial":args.series_trial,"series_position":args.series_position,"started_at_ns":started_at_ns,
              "skim_cache":args.skim_cache,
              "batch_process_wall_seconds":elapsed,"runs":runs,"worker":worker,
              "source_sha256":fingerprints,"configuration_sha256":config,
              "data_sha256":data_hashes,
              "setup_included":True,"verification_outside_timed_region":True}
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("mode","execution","batch_process_wall_seconds","complete")}))


if __name__ == "__main__":
    main()
