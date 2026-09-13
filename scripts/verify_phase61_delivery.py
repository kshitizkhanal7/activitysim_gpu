"""Check delivered claims, PDF text and frozen benchmark-source provenance."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text())


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qualified=read("benchmark-results/phase61-formal-qualification.json")
    comparison=read("benchmark-results/phase61-complete-comparison.json")
    for name,expected in qualified["source_sha256"].items():
        if digest(ROOT/name)!=expected:raise ValueError(f"Timed source changed: {name}")
    for document in (qualified,comparison):
        for name,expected in document["evidence_sha256"].items():
            path=Path(name)
            if not path.is_absolute():path=ROOT/path
            if digest(path)!=expected:raise ValueError(f"Evidence changed: {name}")
    if not qualified["all_pairs_faster"] or qualified["median_wall_under_75"]:
        raise ValueError("Published improvement/target statement no longer matches qualification")
    pdf=ROOT/"output/pdf/choiceforge-plain-english-explainer.pdf"
    pages=[p.extract_text() for p in PdfReader(pdf).pages]
    text="\n".join(pages)
    wall=f"{qualified['medians']['candidate']['process_wall_seconds']:.2f}"
    cpu=f"{comparison['totals']['regular48']['process_wall_seconds']:.2f}"
    ratio=f"{comparison['cpu_over_phase61']['48']['process_wall_seconds']:.2f}"
    for name in ("README.md","docs/choiceforge-plain-english-explainer.md","docs/phase61-component-comparison.md"):
        source=(ROOT/name).read_text(encoding="utf8")
        if wall not in source or cpu not in source:raise ValueError(f"Missing final times: {name}")
    for needle in ("Phase 61",wall,"target is NOT met"):
        if needle not in pages[0]:raise ValueError(f"Cover omits {needle}")
    for needle in (cpu,ratio,"545",*(f"{n}. " for n in range(321,329))):
        if needle not in text:raise ValueError(f"PDF omits {needle}")
    paths=[pdf,ROOT/"README.md",ROOT/"docs/choiceforge-plain-english-explainer.md",
           ROOT/"docs/phase61-shared-live-inputs.md",ROOT/"docs/phase61-component-comparison.md",
           ROOT/"scripts/build_plain_english_explainer_pdf.py",Path(__file__)]
    result={"status":"passed","pdf_pages":len(pages),"frozen_source_files":len(qualified["source_sha256"]),
            "source_and_evidence_hashes_unchanged":True,"final_claim_text_checks":True,
            "visual_review_is_separate":"Poppler pages 1,2,3,154,155,156,157,158 inspected by the assistant; automated text checks do not establish layout quality",
            "artifact_sha256":{str(p.relative_to(ROOT)):digest(p) for p in paths}}
    (ROOT/"benchmark-results/phase61-delivery-verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
