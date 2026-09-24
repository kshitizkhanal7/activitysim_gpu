"""Consolidate qualified Phase 64 evidence; never equate primitive/model clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from run_phase58_comparison import ROOT, RESULTS, source_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--ablation",required=True)
    parser.add_argument("--scales",nargs="+",required=True)
    parser.add_argument("--controls",type=Path,required=True)
    parser.add_argument("--tests",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),"Preserve previous result")
    paths=[]
    def read(path):
        path=Path(path)
        paths.append(path)
        return json.loads(path.read_text())
    source=source_fingerprint()
    campaign=read(RESULTS/f"phase64-{args.campaign}-qualification.json")
    ablation=read(RESULTS/f"phase64-{args.ablation}-ablation.json")
    controls=read(args.controls)
    require(campaign["status"]=="outputs_and_memory_qualified" and campaign["model_runs"]==42,
            "Repeated outputs/memory not qualified")
    require(campaign["source_sha256"]==ablation["source_sha256"]==source,"Timed source differs")
    require(ablation["complete"] and len(ablation["runs"])==4 and controls["complete"],"Controls incomplete")
    for document in (campaign,ablation,controls):
        for name,expected in document["evidence_sha256"].items():
            path=Path(name)
            require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,f"Evidence changed: {path}")
    scales=[]
    for tag in args.scales:
        receipt=read(RESULTS/f"phase64-{tag}-scale.json")
        require(receipt["status"]=="outputs_qualified_scale_feasibility" and receipt["source_unchanged"]
                and receipt["guard_stop_reason"] is None,"Scale failed")
        require(receipt["source_sha256"]==source,"Scale uses another production version")
        summary=read(receipt["summary"])
        require(summary["complete"],"Incomplete scale outputs")
        candidate=next(r for r in summary["runs"] if r["mode"]=="candidate")
        require(candidate["source_sha256"]==source and candidate["exact"]["decision_columns_exact"]
                and candidate["matrices"]["exact"] and candidate["summary_reports"]["exact"],"Scale output mismatch")
        command=candidate["command"]
        proof=read(command[command.index("--report")+1])
        require(bool(proof["proof_gates"]) and all(proof["proof_gates"].values()),"Scale live proof failed")
        boundary=proof["phase64_pipeline"]["location_boundary"]
        require(boundary["contexts_outstanding"]==0 and any(e["guarded_owners"]>0 for e in boundary["events"]),
                "Scale boundary safeguard not exercised/retained state")
        require(all(e["complete"] and e["rng_unchanged"] and not e["saved_answers_read"] for e in boundary["events"]),
                "Scale boundary contract failed")
        scales.append(dict(tag=tag,households=receipt["households"],
            process_wall_seconds=candidate["process_wall_seconds"],charged_total_seconds=candidate["charged_total_seconds"],
            observed_maximum_child_rss_bytes=receipt["observed_maximum_child_rss_bytes"],
            reference_scope=receipt["reference_scope"],diagnostic_not_repeated_performance=True,
            exact_decisions_matrices_reports=True))
    require({s["households"] for s in scales}=={100000,250000},"Both larger sizes required")
    paths.append(args.tests)
    suites=ET.parse(args.tests).getroot().findall("testsuite")
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ("tests","errors","failures","skipped")}
    require(counts["tests"]>=662 and counts["errors"]==counts["failures"]==counts["skipped"]==0,"Test regression")
    result=dict(status="outputs_memory_controls_qualified",campaign=campaign,ablation=ablation,
        primitive_controls=controls,scale=scales,tests=counts,source_sha256=source,
        target_met=campaign["under_70"],stretch_target_met=campaign["under_65"],
        resident_control_scope="Large utility/draw/result arrays remain on GPU. Timed API includes six coefficient uploads, scalar invalid-row download and synchronization; not CPU-free or zero-transfer execution.",
        scope="One pinned workstation. Full-model, matched-worker, reducer-only and scale clocks remain separate. Scale rechecks may reuse independently generated CPU references and do not imply contemporary CPU timings. No new clean-install or independent-hardware replication claim.",
        evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],target_met=result["target_met"],
        whole_model_seconds=campaign["medians_seconds"],cpu_over_hybrid=campaign["ordinary_cpu_over_hybrid"])))


if __name__=="__main__":
    main()
