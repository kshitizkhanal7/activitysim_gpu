"""Sequential scenario durability: identical fresh/persistent worker entry points.

No output replay, no simultaneous models, no pruning. Both CPU and hybrid get
the same content-validated input-table reuse and Phase 63 preparation options.
"""
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
from phase63_commands import candidate_command, output_directory

SEQUENCE = "A,B,C,D,A,C,B,A,D,A"


def scenarios(reference_map=None):
    cases = {
        "A":(50000,None,REFERENCE),
        "B":(50000,ROOT/"benchmark-data/configs_phase59_seed17",PROJECT/"o-p58-p59finalseed17-regular-1"),
        "C":(10000,ROOT/"benchmark-data/configs_phase59_seed991",PROJECT/"o-p58-p59scenario10k1-regular-1"),
        "D":(50000,ROOT/"benchmark-data/configs_phase63_coeff",PROJECT/"o-p58-p63coeffdev1-regular-1"),
    }
    if reference_map is not None:
        if set(reference_map)!=set(cases):
            raise ValueError("Reference map must contain exactly A, B, C, D")
        cases = {key:(size,overlay,Path(reference_map[key]).resolve()) for key,(size,overlay,_) in cases.items()}
    return cases


def fingerprints(cases,worker_script):
    source = source_fingerprint()
    source[str(Path(__file__).relative_to(ROOT))] = file_digest(__file__)
    source[str(worker_script.relative_to(ROOT))] = file_digest(worker_script)
    helper = ROOT/"scripts/phase63_commands.py"
    source[str(helper.relative_to(ROOT))] = file_digest(helper)
    configs = config_fingerprint()
    for _,overlay,_ in cases.values():
        if overlay:
            configs.update(config_fingerprint(overlay))
    return source,configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--mode",choices=("candidate","regular"),required=True)
    parser.add_argument("--execution",choices=("fresh","batch"),required=True)
    parser.add_argument("--sequence",default=SEQUENCE)
    parser.add_argument("--trial",type=int,default=0)
    parser.add_argument("--position",type=int,default=0)
    parser.add_argument("--features",default="plans,files,rss")
    parser.add_argument("--template",type=Path,help="Optional development override; default is the portable explicit command")
    parser.add_argument("--reference-map",type=Path)
    parser.add_argument("--worker-script",type=Path,default=ROOT/"scripts/phase62_batch_worker.py")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    cases = scenarios(json.loads(args.reference_map.read_text()) if args.reference_map else None)
    worker_script = args.worker_script.resolve()
    if worker_script not in {ROOT/"scripts/phase62_batch_worker.py",ROOT/"scripts/diagnose_phase63_retention.py"}:
        raise ValueError("Unrecognized worker")
    sequence = args.sequence.split(",")
    if not sequence or not set(sequence)<=set(cases):
        raise ValueError("Unknown scenario")
    prefix = "phase63-"+args.tag
    target = RESULTS/f"{prefix}.json"
    if target.exists():
        raise FileExistsError(target)
    env = {k:v for k,v in os.environ.items() if not k.startswith("CHOICEFORGE_")}
    settings = (ROOT/"scripts/run_phase32_full_model_ab.ps1").read_text()
    env.update(dict(re.findall(r'^\$env:([A-Z_0-9]+) = "([^"$]*)"$',settings,re.M)))
    env.update(NUMBA_NUM_THREADS="48",OMP_WAIT_POLICY="PASSIVE",NUMEXPR_NUM_THREADS="1",
               CHOICEFORGE_STRICT_CUDA_CANDIDATE="1" if args.mode=="candidate" else "0",
               CHOICEFORGE_STRICT_CUDA_MODE_CHOICE="1" if args.mode=="candidate" else "0")
    env["PATH"] = str(PYTHON.parent)+os.pathsep+env["PATH"]
    if args.mode=="candidate":
        env["CHOICEFORGE_NUMBA_INITIAL_THREADS"] = "1"
    template = next(r for r in json.loads(args.template.read_text())["runs"] if r["mode"]=="candidate") if args.template else None
    if template and "--phase63-features" not in template["command"]:
        raise ValueError("Expected Phase 63 candidate template")
    runs = []
    for index,key in enumerate(sequence):
        households,overlay,reference = cases[key]
        if not (reference/"pipeline.parquetpipeline").is_dir():
            raise FileNotFoundError(f"Independent CPU reference missing: {reference}")
        name = f"{index+1:02d}-{key}"
        output = output_directory(f"o-{prefix}-{name}")
        output.parent.mkdir(parents=True,exist_ok=True)
        report = RESULTS/f"{prefix}-{name}-report.json"
        kernels = RESULTS/f"{prefix}-{name}-kernels"
        for path in (output,report,kernels):
            if path.exists():
                raise FileExistsError(path)
        child_env = dict(env)
        if args.mode=="candidate":
            command = list(template["command"]) if template else candidate_command(output,reference,report,kernels,
                RESULTS/f"{prefix}-{name}-checkpoint.json",households,overlay,args.features)
            for flag,value in (("--output",output),("--report",report),("--kernel-reports",kernels),
                               ("--checkpoint",RESULTS/f"{prefix}-{name}-checkpoint.json"),
                               ("--reference-pipeline",reference/"pipeline.parquetpipeline"),
                               ("--households-sample-size",households),("--phase63-features",args.features)):
                command[command.index(flag)+1] = str(value)
            if "--phase59-scenario-gates" not in command:
                command.append("--phase59-scenario-gates")
            if overlay and template:
                command[2:2] = ["--config-overlay",str(overlay.resolve())]
            (kernels/"mode").mkdir(parents=True)
            child_env.update(CHOICEFORGE_PHASE17_REPORT_DIR=str(kernels),
                CHOICEFORGE_PHASE17_MODE_REPORT_DIR=str(kernels/"mode"),CHOICEFORGE_PHASE17_RUN_ID=f"{prefix}-{name}")
            cwd = ROOT
        else:
            command = [str(PYTHON.parent/"activitysim.exe"),"run","-c","configs_sh","-c","configs",
                       "-d","data_full","-o",str(output),"--households_sample_size",str(households),"--fast"]
            if overlay:
                command[2:2] = ["-c",str(overlay.resolve())]
            cwd = PROJECT
        safe_env = {k:v for k,v in child_env.items() if k.startswith("CHOICEFORGE_") or k in
                    {"NUMBA_NUM_THREADS","OMP_WAIT_POLICY","OMP_NUM_THREADS","MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"}}
        runs.append(dict(name=name,scenario=key,mode=args.mode,command=command,cwd=str(cwd),
                         environment=safe_env,output=str(output),reference=str(reference),report=str(report)))
    source,configs = fingerprints(cases,worker_script)
    data = {str(p):file_digest(p) for p in sorted((PROJECT/"data_full").iterdir()) if p.is_file()}
    reference_hashes = {str(p):file_digest(p) for _,_,ref in cases.values() for p in sorted(ref.glob("final_*.csv"))}
    groups = [runs] if args.execution=="batch" else [[r] for r in runs]
    workers,process_times = [],[]
    started_at_ns = time.time_ns()
    with (RESULTS/f"{prefix}.stdout.log").open("w") as stdout,(RESULTS/f"{prefix}.stderr.log").open("w") as stderr:
        for index,group in enumerate(groups):
            manifest = RESULTS/f"{prefix}-manifest-{index}.json"
            result = RESULTS/f"{prefix}-worker-{index}.json"
            manifest.write_text(json.dumps(dict(runs=group,result=str(result),mode=args.mode,
                skim_cache="none",data_sha256=data,phase63_features=args.features),indent=2)+"\n")
            started = time.perf_counter()
            subprocess.run([str(PYTHON),str(worker_script),str(manifest)],
                           cwd=ROOT,env=env,stdout=stdout,stderr=stderr,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            process_times.append(time.perf_counter()-started)
            worker = json.loads(result.read_text())
            if not worker["complete"] or len(worker["runs"])!=len(group):
                raise ValueError("Incomplete worker")
            workers.append(worker)
    if fingerprints(cases,worker_script)!=(source,configs):
        raise ValueError("Source/configuration changed during experiment")
    verify_files(data)
    verify_files(reference_hashes)
    from verify_phase15_outputs import verify as decisions
    from verify_phase59_matrices import verify as matrices
    from verify_phase59_reports import verify as reports
    measured = [r for w in workers for r in w["runs"]]
    for run,measurement in zip(runs,measured):
        reference,output = Path(run["reference"]),Path(run["output"])
        run.update(measurement)
        run.update(exact=decisions(reference,output),matrices=matrices(reference,output),
                   summary_reports=reports(reference,output))
        with (output/"timing_log.csv").open() as stream:
            run["components"] = {r["model_name"]:float(r["seconds"]) for r in csv.DictReader(stream)}
        if len(run["components"])!=34:
            raise ValueError("Missing model steps")
        if args.mode=="candidate":
            proof = json.loads(Path(run["report"]).read_text())
            if not all(proof["proof_gates"].values()) or not proof["phase63_durable_execution"]["enabled"]:
                raise ValueError("Candidate proof failed")
    result = dict(complete=True,mode=args.mode,execution=args.execution,sequence=sequence,features=args.features,
                  series_trial=args.trial,series_position=args.position,started_at_ns=started_at_ns,
                  batch_process_wall_seconds=sum(process_times),process_wall_seconds=process_times,
                  runs=runs,workers=workers,source_sha256=source,configuration_sha256=configs,
                  data_sha256=data,reference_output_sha256=reference_hashes,
                  diagnostic_not_performance=worker_script.name=="diagnose_phase63_retention.py",
                  setup_included=True,verification_outside_timed_region=True,
                  clock="worker startup + all input revalidation + model + boundary memory sampling + resets")
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("complete","mode","execution","batch_process_wall_seconds")}))


if __name__=="__main__":
    main()
