"""Portable explicit commands for the selected Phase 63 workload."""
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/"benchmark-data/phase9-mtc-full/prototype_mtc_extended"
PYTHON = ROOT/".venv-phase8/Scripts/python.exe"
OUTPUTS = ROOT/"phase63-runs"


def output_directory(name):
    """Keep checkpoint leaves below legacy Windows path limits, without registry edits."""
    if not re.fullmatch(r"[A-Za-z0-9-]+",name):
        raise ValueError("Invalid output name")
    result = OUTPUTS/name
    longest = result/"pipeline.parquetpipeline/workplace_destination_size/initialize_households.parquet"
    if os.name=="nt" and len(str(longest))>=250:
        raise ValueError("Checkpoint path is too long; use a shorter checkout location or campaign tag")
    return result


def environment(candidate=False):
    env = {k:v for k,v in os.environ.items() if not k.startswith("CHOICEFORGE_")}
    settings = (ROOT/"scripts/run_phase32_full_model_ab.ps1").read_text()
    env.update(dict(re.findall(r'^\$env:([A-Z_0-9]+) = "([^"$]*)"$',settings,re.M)))
    env.update(NUMBA_NUM_THREADS="48",OMP_WAIT_POLICY="PASSIVE",NUMEXPR_NUM_THREADS="1",
               CHOICEFORGE_STRICT_CUDA_CANDIDATE="1" if candidate else "0",
               CHOICEFORGE_STRICT_CUDA_MODE_CHOICE="1" if candidate else "0")
    env["PATH"] = str(PYTHON.parent)+os.pathsep+env["PATH"]
    if candidate:
        env["CHOICEFORGE_NUMBA_INITIAL_THREADS"] = "1"
    return env


def candidate_command(output,reference,report,kernels,checkpoint,households=50000,overlay=None,features="plans,files,rss"):
    command = [str(PYTHON),str(ROOT/"scripts/run_phase22_integrated_scheduling.py"),
        "--project",str(PROJECT),"--data",str(PROJECT/"data_full"),"--output",str(output)]
    for path in ([overlay] if overlay else [])+[ROOT/"benchmark-data/configs_phase33_choiceforge",PROJECT/"configs_sh"]:
        command += ["--config-overlay",str(path)]
    command += ["--full-model","--households-sample-size",str(households),"--native-abi-live",
        "--reference-pipeline",str(reference/"pipeline.parquetpipeline"),"--report",str(report),
        "--checkpoint",str(checkpoint),"--kernel-reports",str(kernels),"--phase57-live-scheduling",
        "--phase58-trip-runtime","--phase59-device-retries","--phase59-live-mandatory",
        "--inputs",str(ROOT/"phase59-no-reference-artifact"),"--phase59-sparse-matrices",
        "--phase60-preparation","--phase60-frequency-threads","48",
        "--phase61-features","skims,timetable,tour_modes,entities,normals,uniforms,labels,packing",
        "--phase61-timetable-backend","cpu","--phase62-features","plans,trip,entities",
        "--phase63-features",features,"--phase59-scenario-gates"]
    return command
