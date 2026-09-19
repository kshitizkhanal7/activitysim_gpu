"""Check Phase 62 delivered claims and provenance after visual PDF review.

Text extraction is not a visual layout test. The separate review manifest
records the rendered pages actually inspected; this verifier checks coverage.
Run only after all timed experiments have finished (input hashing reads GBs).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def read(name):
    return json.loads(resolve(name).read_text(encoding="utf8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(document, key):
    mapping = document[key]
    require(bool(mapping), f"Empty {key}")
    for name, expected in mapping.items():
        require(digest(resolve(name)) == expected, f"Changed {key}: {name}")
    return len(mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-review", type=Path, required=True)
    args = parser.parse_args()
    qualified = read("benchmark-results/phase62-formal-qualification.json")
    comparison = read("benchmark-results/phase62-complete-comparison.json")
    batches = read("benchmark-results/phase62-batch-comparison.json")
    review = read(args.visual_review)
    tests = read("benchmark-results/phase62-test-verification.json")
    require(tests["status"] == "passed" and tests["exit_code"] == 0 and tests["passed"] == 565,
            "Missing delivery test result")
    for document in (qualified, comparison, batches):
        verify_hashes(document, "evidence_sha256")
    verify_hashes(qualified, "source_sha256")
    verify_hashes(qualified, "configuration_sha256")
    verify_hashes(batches, "data_sha256")
    require(batches["source_sha256"] == qualified["source_sha256"], "Different timed sources")
    require(batches["status"] == "qualified" and batches["all_output_gates_pass"], "Unqualified batches")
    wall_value = qualified["medians"]["candidate"]["process_wall_seconds"]
    require(math.isfinite(wall_value) and wall_value > 0, "Invalid wall time")
    require(qualified["median_wall_under_70"] == (wall_value < 70), "Incorrect target gate")
    require(qualified["median_wall_under_65"] == (wall_value < 65), "Incorrect stretch gate")
    wins = all(p["new_wall"] < p["old_wall"] and p["new_charged"] < p["old_charged"]
               for p in qualified["pairs"])
    require(len(qualified["pairs"]) == 6 and wins == qualified["all_pairs_faster"], "Incorrect pair claim")
    expected = ("replicated_improvement" if wins else "correctness_qualified_performance_not_replicated")
    expected += "_target_met" if wall_value < 70 else "_target_not_met"
    require(qualified["status"] == comparison["status"] == expected, "Status mismatch")
    require(len(comparison["components"]) == 34 and len(qualified["scenario_checks"]) == 3,
            "Incomplete steps or scenarios")
    pdf = ROOT / "output/pdf/choiceforge-plain-english-explainer.pdf"
    pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
    text = "\n".join(pages)
    wall = f"{wall_value:.2f}"
    cpu = f"{comparison['totals']['regular48']['process_wall_seconds']:.2f}"
    ratio = f"{comparison['cpu_over_phase62']['48']['process_wall_seconds']:.2f}"
    for name in ("README.md", "docs/choiceforge-plain-english-explainer.md",
                 "docs/phase62-component-comparison.md", "docs/phase62-reusable-execution.md"):
        source = resolve(name).read_text(encoding="utf8")
        require(wall in source and cpu in source, f"Missing final times: {name}")
    target = "target is met" if wall_value < 70 else "target is NOT met"
    for needle in ("Phase 62", wall, target):
        require(needle in pages[0], f"Cover omits {needle}")
    for needle in (cpu, ratio, "565", *(f"{n}. " for n in range(329, 338)),
                   *(f"{batches['medians'][key]:.2f}" for key in batches["medians"])):
        require(needle in text, f"PDF omits {needle}")
    start = next((i + 1 for i, page in enumerate(pages) if "329. Why" in page), None)
    require(start is not None, "Missing Phase 62 appendix")
    required_pages = {1, 2, 3, *range(max(1, start - 1), len(pages) + 1)}
    require(review["pdf_sha256"] == digest(pdf), "Visual review is for a different PDF")
    require(review["pdf_pages"] == len(pages), "Visual review page count differs")
    require(required_pages <= set(review["inspected_pages"]), "Visual review omits new or transition pages")
    require(review["status"] == "passed" and review["method"] == "Poppler rendering and assistant visual inspection",
            "Missing explicit visual review")
    artifacts = [pdf, ROOT / "README.md", ROOT / "docs/choiceforge-plain-english-explainer.md",
                 ROOT / "docs/phase62-reusable-execution.md", ROOT / "docs/phase62-component-comparison.md",
                 ROOT / "docs/phase62-batch-comparison.md", ROOT / "scripts/build_plain_english_explainer_pdf.py",
                 ROOT / "benchmark-results/phase62-test-verification.json",
                 Path(__file__), resolve(args.visual_review)]
    result = {
        "status": "passed", "pdf_pages": len(pages), "phase62_appendix_start_page": start,
        "frozen_source_files": len(qualified["source_sha256"]),
        "source_configuration_input_and_evidence_hashes_unchanged": True,
        "final_claim_text_checks": True, "visual_review": review,
        "qualification_status": expected,
        "artifact_sha256": {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
    }
    output = ROOT / "benchmark-results/phase62-delivery-verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
