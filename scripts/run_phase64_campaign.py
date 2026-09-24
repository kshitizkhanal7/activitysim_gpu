"""Sequential, resumable Phase 64 qualification; never overlap measured models.

Predeclared scope: 12 old/new fresh runs, two ordinary CPU runs, eight matched
preparation fresh runs, and two ten-scenario hybrid memory sequences (42 total).
No new persistent CPU/hybrid ratio is inferred from the hybrid-only sequences.
"""
import argparse
import json
import os
import re
import subprocess

from run_phase63_campaign import execute
from run_phase58_comparison import ROOT, RESULTS, PYTHON


def stages(tag, features, cpu_control_tag=None):
    compare=[str(PYTHON),str(ROOT/"scripts/run_phase58_comparison.py")]
    common=["--phase64","--phase64-features",features,"--phase63-features","plans,files,rss",
            "--output-root",str(ROOT/"phase63-runs")]
    yield compare+["--tag",tag+"-fresh",*common,"--modes","gpu,candidate","--repetitions","6"], RESULTS/f"phase58-{tag}-fresh-summary.json"
    cpu_tag=cpu_control_tag or tag+"-cpu48"
    yield compare+["--tag",cpu_tag,*common,"--modes","regular","--repetitions","2","--regular-numba-threads","48"], RESULTS/f"phase58-{cpu_tag}-summary.json"
    for trial in (1,2):
        for position,mode in enumerate(("regular","candidate") if trial==1 else ("candidate","regular")):
            name=f"{tag}-matched-{mode}-{trial}"
            yield [str(PYTHON),str(ROOT/"scripts/run_phase63_sequence.py"),"--tag",name,
                   "--mode",mode,"--execution","fresh","--sequence","A,A","--trial",str(trial),
                   "--position",str(position),"--phase64-features",features], RESULTS/f"phase64-{name}.json"
    for trial in (1,2):
        name=f"{tag}-memory-{trial}"
        yield [str(PYTHON),str(ROOT/"scripts/run_phase63_sequence.py"),"--tag",name,
               "--mode","candidate","--execution","batch","--trial",str(trial),
               "--phase64-features",features], RESULTS/f"phase64-{name}.json"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--features",default="expressions,inputs,location_boundary")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--cpu-control-tag",help="New tag for replacement CPU controls; preserve rejected earlier receipts")
    args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    if args.cpu_control_tag and not re.fullmatch(r"[a-zA-Z0-9-]+",args.cpu_control_tag):
        raise ValueError("Invalid CPU control tag")
    # The shared comparison harness sets this for hybrid children, not ordinary
    # CPU. Set it explicitly in the campaign's inherited environment for both.
    os.environ["OMP_WAIT_POLICY"]="PASSIVE"
    planned=list(stages(args.tag,args.features,args.cpu_control_tag))
    if args.plan:
        print(json.dumps([dict(command=c,receipt=str(p)) for c,p in planned],indent=2))
        return
    for command,receipt in planned:
        execute(command,receipt)
    qualification=[str(PYTHON),str(ROOT/"scripts/qualify_phase64_campaign.py"),"--tag",args.tag]
    if args.cpu_control_tag:
        qualification += ["--cpu-control",str(RESULTS/f"phase58-{args.cpu_control_tag}-summary.json")]
    subprocess.run(qualification,
                   cwd=ROOT,check=True)


if __name__=="__main__":
    main()
