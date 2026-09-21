"""One prepared-workspace command: independent references, fresh pairs, long controls.

Stages are sequential. Completed receipts can be resumed only if their exact
source/configuration/data hashes still match. Failed partial outputs are kept;
a changed implementation requires a new campaign tag, not overwritten evidence.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from phase63_commands import ROOT, PROJECT, PYTHON, OUTPUTS, output_directory
from phase62_batch_worker import verify_files
from qualify_phase63_sequences import CASES


def completed(path):
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if not doc.get("complete"):
        raise ValueError(f"Incomplete receipt retained; use a new tag after investigating: {path}")
    documents = doc["runs"] if "design" in doc else [doc]
    if not documents:
        raise ValueError("Receipt contains no measured runs")
    for item in documents:
        if not item.get("source_sha256") or not item.get("configuration_sha256"):
            raise ValueError("Receipt is missing source/configuration provenance")
        for field in ("source_sha256","configuration_sha256","data_sha256"):
            mapping = item.get(field,{})
            verify_files({str(Path(p) if Path(p).is_absolute() else ROOT/p):value for p,value in mapping.items()})
    return True


def execute(command,receipt):
    if completed(receipt):
        print(f"Verified completed stage: {receipt.name}",flush=True)
        return
    subprocess.run(command,cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if not completed(receipt):
        raise ValueError(f"Stage produced no complete receipt: {receipt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--reference-map",type=Path,help="Optional already independently generated CPU references")
    parser.add_argument("--plan",action="store_true",help="Print stage outline without running models")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    results = ROOT/"benchmark-results"
    # Preflight the longest naming pattern before expensive reference generation.
    output_directory(f"o-phase63-{args.tag}-candidate-fresh-2-10-A")
    if args.plan:
        print(json.dumps(dict(tag=args.tag,stages=["verify prepared public data and pins","generate four independent CPU references unless supplied",
            "build or fully hash raw skim image","six reversed fresh Phase62/63 pairs","two regular CPU48 fresh controls",
            "two reversed-order trials of four ten-scenario strategies","qualify all scenario outputs and memory"],
            sequential=True,model_runs_excluding_preparation=94,not_a_clean_install_claim=True),indent=2))
        return
    OUTPUTS.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(PYTHON),str(ROOT/"scripts/prepare_phase63_replication.py"),"check"],cwd=ROOT,check=True)
    reference_map = args.reference_map or results/f"phase63-{args.tag}-reference-map.json"
    if args.reference_map is None:
        execute([str(PYTHON),str(ROOT/"scripts/generate_phase63_references.py"),"--tag",args.tag],
                results/f"phase63-{args.tag}-references.json")
    refs = json.loads(reference_map.read_text())
    if set(refs)!={"A","B","C","D"}:
        raise ValueError("Four independent reference scenarios required")
    execute([str(PYTHON),str(ROOT/"scripts/build_phase63_image.py"),"--tag",args.tag,"--reference",refs["A"]],
            results/f"phase63-{args.tag}-image.json")
    execute([str(PYTHON),str(ROOT/"scripts/run_phase58_comparison.py"),"--tag",args.tag+"-fresh","--phase63",
             "--phase63-features","plans,files,rss","--modes","gpu,candidate","--repetitions","6",
             "--output-root",str(OUTPUTS),
             "--scenario-baseline",refs["A"]],results/f"phase58-{args.tag}-fresh-summary.json")
    execute([str(PYTHON),str(ROOT/"scripts/run_phase58_comparison.py"),"--tag",args.tag+"-cpu48","--phase63",
             "--modes","regular","--repetitions","2","--regular-numba-threads","48",
             "--output-root",str(OUTPUTS),
             "--scenario-baseline",refs["A"]],results/f"phase58-{args.tag}-cpu48-summary.json")
    for trial in (1,2):
        for position,(mode,execution) in enumerate(CASES if trial==1 else CASES[::-1]):
            tag = f"{args.tag}-{mode}-{execution}-{trial}"
            execute([str(PYTHON),str(ROOT/"scripts/run_phase63_sequence.py"),"--tag",tag,"--mode",mode,
                     "--execution",execution,"--trial",str(trial),"--position",str(position),
                     "--reference-map",str(reference_map.resolve())],results/f"phase63-{tag}.json")
    subprocess.run([str(PYTHON),str(ROOT/"scripts/qualify_phase63_sequences.py"),"--tag",args.tag],cwd=ROOT,check=True)
    print("Measurement campaign complete. Review fresh clocks, memory gates, tests and documentation before declaring Phase 63 delivered.")


if __name__=="__main__":
    main()
