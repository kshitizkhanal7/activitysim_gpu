"""Two reversed pairs replacing only the mode reducer, including transfers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from run_phase58_comparison import ROOT, RESULTS, PYTHON, source_fingerprint
from run_phase63_campaign import execute


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--features",default="expressions,inputs,location_boundary")
    args=parser.parse_args()
    if "mode_cpu" in args.features.split(","):
        raise ValueError("Base features must use GPU reduction")
    target=RESULTS/f"phase64-{args.tag}-ablation.json"
    if target.exists():
        raise FileExistsError(target)
    source=source_fingerprint()
    runs=[]
    paths=[]
    for trial in (1,2):
        for backend in (("gpu","cpu") if trial==1 else ("cpu","gpu")):
            tag=f"{args.tag}-{backend}{trial}"
            features=args.features+(",mode_cpu" if backend=="cpu" else "")
            receipt=RESULTS/f"phase58-{tag}-summary.json"
            execute([str(PYTHON),str(ROOT/"scripts/run_phase58_comparison.py"),"--tag",tag,
                "--phase64","--phase64-features",features,"--phase63-features","plans,files,rss",
                "--modes","candidate","--repetitions","1","--output-root",str(ROOT/"phase63-runs")],receipt)
            run=json.loads(receipt.read_text())["runs"][0]
            if run["source_sha256"]!=source or source_fingerprint()!=source:
                raise ValueError("Ablation source changed")
            command=run["command"]
            proof_path=ROOT/command[command.index("--report")+1]
            proof=json.loads(proof_path.read_text())
            events=proof["phase64_pipeline"]["mode_events"]
            if backend=="cpu":
                if not events or not all(e["device_to_host_bytes"]>0 and e["host_to_device_bytes"]>0 for e in events):
                    raise ValueError("CPU ablation not exercised/transfers missing")
            elif events:
                raise ValueError("GPU baseline contains CPU ablation")
            runs.append(dict(trial=trial,backend=backend,process_wall_seconds=run["process_wall_seconds"],
                             components=run["components"],cpu_reducer_events=events,
                             source_receipt=str(receipt)))
            paths += [receipt,proof_path]
    medians={b:statistics.median(r["process_wall_seconds"] for r in runs if r["backend"]==b) for b in ("gpu","cpu")}
    result=dict(complete=True,runs=runs,medians_seconds=medians,cpu_reducer_over_gpu_reducer=medians["cpu"]/medians["gpu"],
        scope="Two reversed full-model pairs. Only tour/trip nested reduction changes; utility generation and RNG remain on GPU. CPU ablation charges device-host-device transfers. Not an all-CPU ActivitySim comparison or transfer-free CPU lower bound.",
        source_sha256=source,evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]})
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(medians))


if __name__=="__main__":
    main()
