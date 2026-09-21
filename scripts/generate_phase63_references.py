"""Recompute four public scenarios with regular CPU ActivitySim, not old answers."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import time

from phase63_commands import ROOT, PROJECT, PYTHON, environment, output_directory
from phase62_batch_worker import file_digest, verify_files
from run_phase58_comparison import source_fingerprint, config_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    results = ROOT/"benchmark-results"
    target = results/f"phase63-{args.tag}-references.json"
    mapping = results/f"phase63-{args.tag}-reference-map.json"
    if target.exists() or mapping.exists():
        raise FileExistsError(target)
    configs = [("A",50000,None),("B",50000,ROOT/"benchmark-data/configs_phase59_seed17"),
               ("C",10000,ROOT/"benchmark-data/configs_phase59_seed991"),
               ("D",50000,ROOT/"benchmark-data/configs_phase63_coeff")]
    source = source_fingerprint()
    source.update({str(p.relative_to(ROOT)):file_digest(p) for p in (Path(__file__),ROOT/"scripts/phase63_commands.py")})
    config = config_fingerprint()
    for _,_,overlay in configs:
        if overlay:
            config.update(config_fingerprint(overlay))
    data = {str(p):file_digest(p) for p in sorted((PROJECT/"data_full").iterdir()) if p.is_file()}
    runs,paths = [],{}
    for name,size,overlay in configs:
        prefix = f"phase63-{args.tag}-reference-{name}"
        output = output_directory("o-"+prefix)
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        command = [str(PYTHON.parent/"activitysim.exe"),"run"]
        if overlay:
            command += ["-c",str(overlay)]
        command += ["-c","configs_sh","-c","configs","-d","data_full","-o",str(output),
                    "--households_sample_size",str(size),"--fast"]
        started = time.perf_counter()
        with (results/f"{prefix}.stdout.log").open("w") as out,(results/f"{prefix}.stderr.log").open("w") as err:
            subprocess.run(command,cwd=PROJECT,env=environment(),stdout=out,stderr=err,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        elapsed = time.perf_counter()-started
        with (output/"timing_log.csv").open() as stream:
            steps = {r["model_name"]:float(r["seconds"]) for r in csv.DictReader(stream)}
        if len(steps)!=34 or not (output/"pipeline.parquetpipeline").is_dir():
            raise ValueError("CPU reference is incomplete")
        paths[name] = str(output)
        runs.append(dict(scenario=name,output=str(output),command=command,process_wall_seconds=elapsed,components=steps,
                         final_output_sha256={p.name:file_digest(p) for p in sorted(output.glob("final_*.csv"))}))
        print(json.dumps(dict(scenario=name,seconds=elapsed)),flush=True)
    verify_files({str(ROOT/p):value for p,value in source.items()})
    verify_files(config)
    verify_files(data)
    import pandas as pd
    a = pd.read_csv(Path(paths["A"])/"final_persons.csv",usecols=["person_id","cdap_activity"]).set_index("person_id")
    d = pd.read_csv(Path(paths["D"])/"final_persons.csv",usecols=["person_id","cdap_activity"]).set_index("person_id")
    if not a.index.equals(d.index):
        raise ValueError("Coefficient test changed sampled persons unexpectedly")
    changed = int((a.cdap_activity!=d.cdap_activity).sum())
    if changed==0:
        raise ValueError("Coefficient scenario did not change its target behavior")
    payload = dict(complete=True,regular_cpu=True,preparation_cache_enabled=False,runs=runs,
                   coefficient_changed_person_activities=changed,source_sha256=source,
                   configuration_sha256=config,data_sha256=data,
                   scope="CPU-produced reference outputs, not a self-comparison proof of GPU correctness")
    target.write_text(json.dumps(payload,indent=2)+"\n")
    mapping.write_text(json.dumps(paths,indent=2)+"\n")


if __name__=="__main__":
    main()
