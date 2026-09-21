"""Recheck Phase 63 evidence and publication after all model timing is finished.

This hashes large inputs and reads the PDF; never run beside model measurements.
Visual review is separate: extracted text alone cannot establish good layout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pypdf import PdfReader

from qualify_phase63_sequences import qualify
from report_phase63 import matched_default_scenarios
from run_phase58_comparison import ROOT, source_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def read(name):
    return json.loads(resolve(name).read_text(encoding="utf8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hashes(mapping):
    require(bool(mapping), "Empty provenance mapping")
    for name, expected in mapping.items():
        require(digest(resolve(name)) == expected, f"Changed evidence: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-review", type=Path, required=True)
    args = parser.parse_args()
    comparison = read("benchmark-results/phase63-complete-comparison.json")
    sequence = read("benchmark-results/phase63-sequence-qualification.json")
    tests = read("benchmark-results/phase63-test-verification.json")
    reconstruction = read("benchmark-results/phase63-reconstruction-verification.json")
    checkout = read("benchmark-results/phase63-git-byte-verification.json")
    references_path = ROOT / "benchmark-results/phase63-p63formal-references.json"
    references = read(references_path)
    review = read(args.visual_review)
    for doc in (comparison, sequence, tests, reconstruction, checkout):
        check_hashes(doc["evidence_sha256"])
    require(checkout["status"] == "passed" and
            {trial["core_autocrlf"] for trial in checkout["trials"]} == {"true", "false"} and
            all(not trial["differences"] for trial in checkout["trials"]),
            "Publication Git byte round trip failed")
    require(comparison["source_sha256"] == source_fingerprint(), "Delivery production differs from timing")
    require(references["complete"] and references["regular_cpu"] and not references["preparation_cache_enabled"],
            "References were not independently generated with regular CPU")
    require(all(references["source_sha256"].get(name) == value for name, value in source_fingerprint().items()),
            "Reference model production differs")
    require({r["scenario"] for r in references["runs"]} == {"A", "B", "C", "D"}
            and len(references["runs"]) == 4 and references["coefficient_changed_person_activities"] > 0,
            "Independent scenario references incomplete")
    for run in references["runs"]:
        require(Path(run["command"][0]).name == "activitysim.exe" and len(run["components"]) == 34,
                "Reference command is not a complete CPU model")
        for name, value in run["final_output_sha256"].items():
            path = Path(run["output"]) / name
            require(digest(path) == value and sequence["reference_output_sha256"][str(path)] == value,
                    "Reference output changed since CPU generation")
    for field in ("source_sha256", "configuration_sha256", "data_sha256", "reference_output_sha256"):
        check_hashes(sequence[field])
    documents = []
    for name in sequence["evidence_sha256"]:
        if name.endswith(".json"):
            doc = read(name)
            if "series_trial" in doc:
                documents.append(doc)
    replay = qualify(documents)
    for key, value in replay.items():
        require(sequence[key] == value, f"Recomputed sequence differs: {key}")
    require(comparison["matched_preparation_default_fresh"] == matched_default_scenarios(sequence),
            "Matched-preparation comparison differs from scenario evidence")
    require(sequence["status"] == "outputs_qualified" and sequence["model_runs"] == 80,
            "Durability qualification incomplete")
    require(tests["main"]["exit_code"] == tests["clean_replica"]["exit_code"] == 0,
            "Test command failed")
    require(tests["main"]["passed"] >= 602 and tests["main"]["failed"] == 0,
            "Main test coverage regressed")
    require(tests["clean_replica"]["failed"] == 0 and tests["clean_replica"]["skipped"] == 1,
            "Unexpected clean-checkout test failure/skip")
    require(reconstruction["status"] == "passed" and reconstruction["source_differences"] == [],
            "Reconstruction differs")
    require(reconstruction["same_machine"] and not reconstruction["independent_hardware_replication"],
            "Reconstruction scope overstated")
    medians = comparison["medians"]
    wall = medians["phase63"]["process_wall_seconds"]
    cpu = medians["regular48"]["process_wall_seconds"]
    require(math.isfinite(wall) and wall > 0 and len(comparison["components"]) == 34,
            "Invalid whole-model result")
    require(comparison["median_wall_under_70"] == (wall < 70), "Incorrect target claim")
    require(comparison["median_wall_under_65"] == (wall < 65), "Incorrect stretch claim")
    require(abs(comparison["cpu_over_phase63"] - cpu / wall) < 1e-12, "Incorrect CPU ratio")
    wins = all(p["new_wall"] < p["old_wall"] and p["new_charged"] < p["old_charged"]
               for p in comparison["pairs"])
    require(len(comparison["pairs"]) == 6 and comparison["all_pairs_faster"] == wins,
            "Incorrect paired improvement claim")
    pdf = ROOT / "output/pdf/choiceforge-plain-english-explainer.pdf"
    pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
    text = "\n".join(pages)
    required_documents = ("README.md", "docs/choiceforge-plain-english-explainer.md",
                          "docs/phase63-component-comparison.md", "docs/phase63-durable-runtime.md")
    for name in required_documents:
        content = resolve(name).read_text(encoding="utf8")
        require(f"{wall:.2f}" in content and f"{cpu:.2f}" in content, f"Missing final clocks: {name}")
    for needle in ("Phase 63", f"{wall:.2f}", "target is met" if wall < 70 else "target is NOT met"):
        require(needle in pages[0], f"Cover omits {needle}")
    for needle in (f"{cpu:.2f}", f"{cpu / wall:.2f}", str(tests["main"]["passed"]), "338. "):
        require(needle in text, f"PDF omits {needle}")
    start = next((i + 1 for i, page in enumerate(pages) if "338. " in page), None)
    require(start is not None, "Missing Phase 63 appendix")
    needed = {1, 2, 3, *range(max(1, start - 1), len(pages) + 1)}
    require(review["pdf_sha256"] == digest(pdf) and review["pdf_pages"] == len(pages),
            "Visual review describes another PDF")
    require(needed <= set(review["inspected_pages"]), "New/transition pages not visually reviewed")
    require(review["status"] == "passed" and review["method"] == "Poppler rendering and assistant visual inspection",
            "Missing actual visual review")
    artifacts = [resolve(name) for name in required_documents]
    artifacts += [pdf, Path(__file__), references_path, resolve(args.visual_review), ROOT / "docs/phase63-replication.md",
                  ROOT / "scripts/build_plain_english_explainer_pdf.py",
                  ROOT / "benchmark-results/phase63-git-byte-verification.json"]
    result = dict(status="passed", qualification_status=comparison["status"], formal_measured_models=94,
                  sequence_replayed=True, source_configuration_inputs_evidence_unchanged=True,
                  pdf_pages=len(pages), phase63_appendix_start_page=start, visual_review=review,
                  artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in artifacts})
    target = ROOT / "benchmark-results/phase63-delivery-verification.json"
    require(not target.exists(), "Refusing to overwrite delivery evidence")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
