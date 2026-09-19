"""Phase 62 gates: all prior contracts plus actual reusable-preparation use."""
import argparse
import hashlib
import json
from pathlib import Path
import qualify_phase61 as previous
from qualify_phase59 import require

original_audit = previous.audit


def audit(run,candidate):
    proof = original_audit(run,True)
    require(("--phase62-features" in run["command"]) == candidate,"wrong Phase 62 backend")
    result = proof["phase62_reusable_execution"]
    require(result["enabled"] is candidate,"wrong runtime enabled state")
    if candidate:
        require(set(result["features"])=={"plans","trip","entities"},"missing Phase 62 feature")
        require(len(result["events"])==34 and result["saved_answers_read"] is False,"missing live audit")
        require(result["expression_hits"]>0 and result["expression_misses"]>0,"compiler did not execute")
        require(sum(e["directory_probe_hits"] for e in result["events"])>0,"no executable setup reuse")
        require(result["demand_requests"],"no demand publication")
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--scenarios",type=Path,nargs=3,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = [args.comparison,*args.scenarios]
    summaries = [json.loads(p.read_text()) for p in paths]
    previous.audit = audit
    try:
        result = previous.qualify(summaries[0],summaries[1:])
    finally:
        previous.audit = original_audit
    result.pop("median_wall_under_75")
    wall = result["medians"]["candidate"]["process_wall_seconds"]
    result["median_wall_under_70"] = wall<70
    result["median_wall_under_65"] = wall<65
    result["status"] = ("replicated_improvement" if result["all_pairs_faster"] else "correctness_qualified_performance_not_replicated") + ("_target_met" if wall<70 else "_target_not_met")
    result["scope"] = "Six balanced fresh-process Phase 61/62 pairs; unchanged output contracts; hybrid preparation gains, not new GPU arithmetic. Batch results separate."
    paths += [Path(__file__),Path(previous.__file__),Path(__file__).with_name("qualify_phase60.py"),Path(__file__).with_name("qualify_phase59.py")]
    for summary in summaries:
        for run in summary["runs"]:
            paths.append(Path(run["command"][run["command"].index("--report")+1]))
    result["evidence_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["status"])


if __name__ == "__main__":
    main()
