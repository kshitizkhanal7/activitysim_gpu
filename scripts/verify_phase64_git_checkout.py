"""Check staged Phase 64 bytes under both Git autocrlf settings, without cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_phase58_comparison import ROOT, source_fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    destination=args.destination.resolve()
    if not destination.is_relative_to(ROOT/"tmp") or destination.exists() or args.output.exists():
        raise ValueError("Choose new paths; checkout must be beneath repository tmp")
    staged=set(subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0"))
    required={Path(p).as_posix() for p in source_fingerprint() if Path(p).parts[0]!="tmp"}
    required.update({".gitattributes","README.md","docs/choiceforge-plain-english-explainer.md",
                     "scripts/build_plain_english_explainer_pdf.py","output/pdf/choiceforge-plain-english-explainer.pdf"})
    for name in staged:
        if name.startswith(("benchmark-results/phase64","benchmark-results/phase58-p64")) or (
                name.startswith(("scripts/","tests/","docs/")) and "phase64" in name):
            required.add(name)
    if not required<=staged:
        raise ValueError(f"Missing staged source: {sorted(required-staged)}")
    records={name:digest(ROOT/name) for name in sorted(required)}
    destination.mkdir(parents=True)
    trials=[]
    for policy in ("true","false"):
        target=destination/policy
        target.mkdir()
        subprocess.run(["git","-c",f"core.autocrlf={policy}","checkout-index","-z","--stdin",
                        "--prefix="+target.as_posix()+"/"],cwd=ROOT,check=True,
                       input=("\0".join(records)+"\0").encode())
        differences=[name for name,value in records.items() if digest(target/name)!=value]
        if differences:
            raise ValueError(f"Checkout changed measured bytes: {differences}")
        trials.append(dict(core_autocrlf=policy,matched_files=len(records),differences=[]))
    result=dict(status="passed",trials=trials,scope="Staged byte round trip only; not model rerun or hardware replication",
                evidence_sha256={str(ROOT/name):value for name,value in records.items()})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],trials=trials)))


if __name__=="__main__":
    main()
