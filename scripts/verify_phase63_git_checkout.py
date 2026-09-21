"""Verify staged publication bytes with both Git newline-conversion policies.

Run after model measurements and selective staging. Writes NEW, retained test
directories only; never changes the worktree, global Git settings or old evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_phase58_comparison import ROOT, source_fingerprint


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination",type=Path,required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not destination.is_relative_to(ROOT/"tmp") or destination.exists():
        raise ValueError("Choose a new verification directory beneath this repository's tmp")
    staged = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode("utf8").split("\0")
    required = {Path(p).as_posix() for p in source_fingerprint() if not Path(p).parts[0]=="tmp"}
    required.update({".gitattributes",".gitignore","README.md","requirements-phase63-lock.txt",
                     "integration/activitysim-phase63.patch","docs/choiceforge-plain-english-explainer.md",
                     "output/pdf/choiceforge-plain-english-explainer.pdf"})
    for name in staged:
        if ((name.startswith(("scripts/","tests/","docs/")) and "phase63" in name)
            or name.startswith(("reproducibility/","benchmark-data/configs_phase63_coeff/",
                "benchmark-data/configs_phase33_choiceforge/","benchmark-data/configs_phase59_seed",
                "benchmark-results/phase63","benchmark-results/phase58-p63"))):
            required.add(name)
    if not required <= set(staged):
        raise ValueError(f"Required files have not been staged: {sorted(required-set(staged))}")
    records = {name:digest(ROOT/name) for name in sorted(required)}
    destination.mkdir(parents=True)
    trials = []
    for policy in ("true","false"):
        checkout = destination/policy
        checkout.mkdir()
        command = ["git","-c",f"core.autocrlf={policy}","checkout-index","-z","--stdin",
                   "--prefix="+checkout.as_posix()+"/"]
        subprocess.run(command,cwd=ROOT,input=("\0".join(sorted(required))+"\0").encode("utf8"),check=True)
        differences = [name for name,expected in records.items() if digest(checkout/name)!=expected]
        if differences:
            raise ValueError(f"Git checkout bytes differ with autocrlf={policy}: {differences}")
        trials.append(dict(core_autocrlf=policy,matched_files=len(records),differences=[]))
    result = dict(status="passed",scope="Staged publication round trip, not another model run or hardware replica",
                  trials=trials,evidence_sha256={str(ROOT/name):value for name,value in records.items()})
    target = ROOT/"benchmark-results/phase63-git-byte-verification.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="passed",trials=trials)))


if __name__=="__main__":
    main()
