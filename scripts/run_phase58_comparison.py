"""Fresh-process, reverse-order controls; profiling runs are never timing evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "benchmark-data/phase9-mtc-full/prototype_mtc_extended"
RESULTS = ROOT / "benchmark-results"
PYTHON = ROOT / ".venv-phase8/Scripts/python.exe"
REFERENCE = PROJECT / "o-p17modeproof16-baseline-50000-1"


def source_fingerprint():
    paths = sorted((ROOT / "src/choiceforge").rglob("*.py"))
    paths += sorted((ROOT / "tmp/activitysim-phase8-source/activitysim").rglob("*.py"))
    paths += [ROOT / "src/choiceforge/kernels" / name for name in (
        "phase52_public_destination_tile4.cu", "phase55_public_destination_sm86.json",
        "phase55_public_destination_plans.json")]
    paths += [Path(__file__), ROOT / "scripts/run_phase22_integrated_scheduling.py",
              ROOT / "scripts/verify_phase15_outputs.py", ROOT / "scripts/verify_phase59_matrices.py"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def config_fingerprint(overlay=None):
    directories = [PROJECT / "configs", PROJECT / "configs_sh", ROOT / "benchmark-data/configs_phase33_choiceforge"]
    if overlay is not None:
        directories.append(overlay.resolve())
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in directories for p in sorted(directory.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".csv"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modes", default="regular,cpu,gpu")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--phase59", action="store_true", help="Compare Phase 58 against the developing Phase 59 candidate")
    parser.add_argument("--live-mandatory", action="store_true", help="Use Phase 59 live mandatory scheduling in candidate")
    parser.add_argument("--prune-verified-output", action="store_true",
                        help="Remove only this run's reproducible output directory after saving exact audit and file hashes")
    parser.add_argument("--scenario-overlay", type=Path)
    parser.add_argument("--scenario-baseline", type=Path, help="Reuse an independently generated CPU scenario baseline")
    parser.add_argument("--households", type=int, default=50000)
    parser.add_argument("--output-root", type=Path, default=PROJECT)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--sparse-matrices", action="store_true")
    parser.add_argument("--capture-mode-inputs", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+", args.tag):
        parser.error("tag must contain only letters, digits, hyphens")
    modes = args.modes.split(",")
    if args.capture_mode_inputs and (modes != ["candidate"] or args.repetitions != 1):
        parser.error("input capture requires one candidate diagnostic run")
    scenario = args.scenario_overlay is not None or args.households != 50000
    scenario_modes_ok = modes == ["regular", "candidate"] or (modes == ["candidate"] and args.scenario_baseline is not None)
    if scenario and (not scenario_modes_ok or args.repetitions != 1 or not args.live_mandatory):
        parser.error("scenario runs require regular,candidate, one repetition and live mandatory scheduling")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"regular", "cpu", "gpu", "candidate"}:
        parser.error("modes must be unique regular,cpu,gpu,candidate entries")
    if args.repetitions < 1 or (args.profile and (modes not in (["gpu"], ["candidate"]) or args.repetitions != 1)):
        parser.error("profiling requires a single GPU or candidate run")
    summary_path = RESULTS / f"phase58-{args.tag}-summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    env = {k:v for k,v in os.environ.items() if not k.startswith("CHOICEFORGE_")}
    settings = (ROOT / "scripts/run_phase32_full_model_ab.ps1").read_text()
    env.update(dict(re.findall(r'^\$env:([A-Z_0-9]+) = "([^"$]*)"$', settings, re.M)))
    env["PATH"] = str(PYTHON.parent)+os.pathsep+env["PATH"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.scratch_root:
        for variable, directory in (("TEMP", "temp"), ("TMP", "temp"),
                                    ("CUPY_CACHE_DIR", "cupy"), ("NUMBA_CACHE_DIR", "numba")):
            path = args.scratch_root.resolve() / directory
            path.mkdir(parents=True, exist_ok=True)
            env[variable] = str(path)
    runs = []
    reference = args.scenario_baseline.resolve() if args.scenario_baseline else REFERENCE
    for trial in range(1, args.repetitions+1):
        order = modes if trial % 2 else modes[::-1]
        for position, mode in enumerate(order):
            prefix = f"phase58-{args.tag}-{mode}-{trial}"
            output = args.output_root.resolve() / f"o-p58-{args.tag}-{mode}-{trial}"
            report = RESULTS / f"{prefix}.json"
            exact = RESULTS / f"{prefix}-exact.json"
            kernels = RESULTS / f"{prefix}-kernels"
            for path in (output, report, exact, kernels):
                if path.exists():
                    raise FileExistsError(path)
            child_env = env.copy()
            child_env["NUMBA_NUM_THREADS"] = "48" if mode == "cpu" else "1"
            child_env["CHOICEFORGE_STRICT_CUDA_CANDIDATE"] = "0" if mode == "regular" else "1"
            child_env["CHOICEFORGE_STRICT_CUDA_MODE_CHOICE"] = "0" if mode == "regular" else "1"
            if mode == "regular":
                command = [str(PYTHON.parent / "activitysim.exe"), "run", "-c", "configs_sh",
                           "-c", "configs", "-d", "data_full", "-o", str(output),
                           "--households_sample_size", str(args.households)]
                if args.scenario_overlay:
                    command[2:2] = ["-c", str(args.scenario_overlay.resolve())]
                cwd = PROJECT
            else:
                (kernels / "mode").mkdir(parents=True)
                child_env.update(CHOICEFORGE_PHASE17_REPORT_DIR=str(kernels),
                                 CHOICEFORGE_PHASE17_MODE_REPORT_DIR=str(kernels / "mode"),
                                 CHOICEFORGE_PHASE17_RUN_ID=prefix)
                command = [str(PYTHON), str(ROOT / "scripts/run_phase22_integrated_scheduling.py"),
                           "--project", str(PROJECT), "--data", str(PROJECT / "data_full"),
                           "--output", str(output), "--config-overlay", str(ROOT / "benchmark-data/configs_phase33_choiceforge"),
                           "--config-overlay", str(PROJECT / "configs_sh"), "--full-model",
                           "--households-sample-size", str(args.households), "--native-abi-live",
                           "--reference-pipeline", str(reference / "pipeline.parquetpipeline"),
                           "--report", str(report), "--checkpoint", str(RESULTS / f"{prefix}-checkpoint.json"),
                           "--kernel-reports", str(kernels), "--phase57-live-scheduling"]
                if args.scenario_overlay:
                    insertion = command.index("--config-overlay")
                    command[insertion:insertion] = ["--config-overlay", str(args.scenario_overlay.resolve())]
                if scenario:
                    command += ["--phase59-scenario-gates"]
                if mode == "cpu":
                    command += ["--phase58-compact-cpu-control"]
                if mode == "candidate":
                    command += ["--phase58-trip-runtime"]
                    if args.phase59:
                        command += ["--phase59-device-retries"]
                    if args.live_mandatory:
                        command += ["--phase59-live-mandatory", "--inputs", str(ROOT / "phase59-no-reference-artifact")]
                    if args.sparse_matrices:
                        command += ["--phase59-sparse-matrices"]
                    if args.capture_mode_inputs:
                        command += ["--phase59-capture-mode-inputs", str(args.capture_mode_inputs.resolve())]
                if mode == "gpu" and args.phase59:
                    command += ["--phase58-trip-runtime"]
                if args.profile:
                    command += ["--phase58-profile-trips", str(RESULTS / f"{prefix}-profile")]
                cwd = ROOT
            fingerprint = source_fingerprint()
            config_hashes = config_fingerprint(args.scenario_overlay)
            started = time.perf_counter()
            with (RESULTS / f"{prefix}.stdout.log").open("w") as stdout, (RESULTS / f"{prefix}.stderr.log").open("w") as stderr:
                subprocess.run(command, cwd=cwd, env=child_env, stdout=stdout, stderr=stderr,
                               check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            wall = time.perf_counter()-started
            if source_fingerprint() != fingerprint:
                raise RuntimeError("Source changed during a timed child; reject this measurement")
            if config_fingerprint(args.scenario_overlay) != config_hashes:
                raise RuntimeError("Model configuration changed during a child; reject this measurement")
            if scenario and mode == "regular":
                reference = output
            subprocess.run([str(PYTHON), str(ROOT / "scripts/verify_phase15_outputs.py"),
                            "--reference", str(reference), "--candidate", str(output), "--output", str(exact)],
                           cwd=ROOT, stdout=subprocess.DEVNULL, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            matrices = None
            if args.phase59:
                from verify_phase59_matrices import verify as verify_matrices
                matrices = verify_matrices(reference, output)
            with (output / "timing_log.csv").open() as stream:
                components = {r["model_name"]:float(r["seconds"]) for r in csv.DictReader(stream)}
            assert len(components) == 34
            validation = prewarm = 0.0
            if mode != "regular":
                proof = json.loads(report.read_text())
                assert all(proof["proof_gates"].values())
                validation = proof["phase56_modelwide_resident_runtime"]["runtime_validation_seconds"]
                prewarm = sum((proof.get(f"phase{phase}_prewarm") or {}).get("seconds", 0)
                              for phase in (46, 47, 48, 52))
            run = {"mode": mode, "trial": trial, "position": position, "order": order,
                   "process_wall_seconds": wall, "components": components,
                   "model_steps_seconds": sum(components.values()),
                   "validation_seconds": validation, "prewarm_seconds": prewarm,
                   "charged_total_seconds": sum(components.values())+validation+prewarm,
                   "exact": json.loads(exact.read_text()), "command": command,
                   "profiled_not_performance_evidence": bool(args.profile or args.capture_mode_inputs),
                   "source_sha256": fingerprint,
                   "configuration_sha256": config_hashes,
                   "scratch_environment": {k:child_env.get(k) for k in ("TEMP", "TMP", "CUPY_CACHE_DIR", "NUMBA_CACHE_DIR")},
                   "thread_environment": {k:child_env[k] for k in (
                       "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if k in child_env},
                   "output": str(output)}
            run["reference"] = str(reference)
            run["scenario_overlay"] = str(args.scenario_overlay) if args.scenario_overlay else None
            run["regular_scenario_baseline_self_check_not_external_proof"] = bool(scenario and mode == "regular")
            if args.phase59:
                run["matrices"] = matrices
            prune_this_output = args.prune_verified_output and not (scenario and mode == "regular")
            if prune_this_output:
                target = output.resolve()
                if target.parent != args.output_root.resolve() or target.name != f"o-p58-{args.tag}-{mode}-{trial}":
                    raise ValueError("Refusing to prune outside the exact fresh benchmark output")
                run["retained_output_sha256"] = {str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(target.rglob("*")) if p.is_file() and "pipeline.parquetpipeline" not in p.parts}
                run["output_pruned_after_verification"] = True
            runs.append(run)
            summary_path.write_text(json.dumps({"complete": len(runs) == args.repetitions*len(modes),
                "design": "order reversed on even repetitions; no concurrent benchmarks",
                "runs": runs}, indent=2)+"\n")
            if prune_this_output:
                shutil.rmtree(target)
            print(json.dumps({k:run[k] for k in ("mode", "trial", "charged_total_seconds", "process_wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
