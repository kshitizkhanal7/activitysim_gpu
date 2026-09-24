"""Verify final evidence and reviewed explainer; run only after all measurements."""
import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from run_phase58_comparison import ROOT, source_fingerprint


def require(value,message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--visual-review",type=Path,required=True)
    parser.add_argument("--checkout",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),"Preserve existing delivery receipt")
    result=json.loads(args.results.read_text())
    review=json.loads(args.visual_review.read_text())
    checkout=json.loads(args.checkout.read_text())
    require(result["status"]=="outputs_memory_controls_qualified","Incomplete results")
    require(result["source_sha256"]==source_fingerprint(),"Production changed after measurements")
    for document in (result,result["campaign"],result["ablation"],result["primitive_controls"],checkout):
        require(bool(document["evidence_sha256"]),"Missing evidence hashes")
        for name,expected in document["evidence_sha256"].items():
            require(digest(name)==expected,f"Changed evidence: {name}")
    require(checkout["status"]=="passed" and {t["core_autocrlf"] for t in checkout["trials"]}=={"true","false"}
            and all(not t["differences"] for t in checkout["trials"]),"Git byte round trip failed")
    pdf=ROOT/"output/pdf/choiceforge-plain-english-explainer.pdf"
    pages=[p.extract_text() or "" for p in PdfReader(pdf).pages]
    start=next((i+1 for i,p in enumerate(pages) if "347. " in p),None)
    require(start is not None,"Phase 64 appendix absent")
    needed={1,2,3,*range(max(1,start-1),len(pages)+1)}
    require(review["status"]=="passed" and review["method"]=="Poppler rendering and assistant visual inspection",
            "Missing actual visual inspection")
    require(review["pdf_sha256"]==digest(pdf) and review["pdf_pages"]==len(pages)
            and needed<=set(review["inspected_pages"]),"Review incomplete or stale")
    clocks=result["campaign"]["medians_seconds"]
    for needle in ("Phase 64",f'{clocks["phase64"]:.2f}',f'{clocks["ordinary_cpu"]:.2f}'):
        require(needle in pages[0],f"Cover missing final value: {needle}")
    artifacts=[ROOT/name for name in ("README.md","docs/choiceforge-plain-english-explainer.md",
        "docs/phase64-pipeline-and-scale.md","docs/phase64-reproduction.md","docs/phase64-component-comparison.md")]
    for path in artifacts[:3]:
        content=path.read_text(encoding="utf8")
        for needle in (f'{clocks["phase64"]:.2f}',f'{clocks["ordinary_cpu"]:.2f}',"250,000"):
            require(needle in content,f"Documentation lacks qualified result: {path}: {needle}")
    components=result["campaign"]["components"]
    checked=set()
    for line in artifacts[-1].read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells=[c.strip() for c in line.strip("|").split("|")]
        name=cells[0]
        if name not in components["ordinary_cpu"]:
            continue
        require(name not in checked and len(cells)==5,"Duplicate/malformed component row")
        values=[components[group][name] for group in ("ordinary_cpu","phase63","phase64")]
        expected=[f"{v:.2f}" for v in values]+[f"{values[0]/values[2]:.2f}x"]
        require(cells[1:]==expected,f"Incorrect component table: {name}")
        checked.add(name)
    require(len(checked)==34,"Incomplete component table")
    artifacts += [pdf,args.results,args.visual_review,args.checkout,Path(__file__),ROOT/"scripts/build_plain_english_explainer_pdf.py"]
    payload=dict(status="passed",formal_campaign_models=42,reducer_ablation_models=4,
        scale_households=[r["households"] for r in result["scale"]],target_met=result["target_met"],
        source_sha256=source_fingerprint(),pdf_pages=len(pages),visual_review=review,
        scope="Local evidence, source, staged checkout bytes and explainer audit; not independent hardware replication or a new clean installation",
        artifact_sha256={str(p):digest(p) for p in artifacts})
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(dict(status=payload["status"],pdf_pages=len(pages),target_met=payload["target_met"])))


if __name__=="__main__":
    main()
