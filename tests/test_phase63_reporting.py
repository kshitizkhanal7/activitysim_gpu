import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import report_phase63 as report
from verify_phase63_reconstruction import test_result as parse_test_result


@pytest.fixture
def synthetic_report(tmp_path, monkeypatch):
    """Synthetic clocks test reporting only; never used as benchmark evidence."""
    results = tmp_path / "benchmark-results"
    results.mkdir()
    source = {"synthetic.py": "not-a-real-digest"}
    def run(mode, trial, wall):
        return dict(mode=mode, trial=trial, source_sha256=source,
                    configuration_sha256={"test.yaml": "synthetic"}, reference="independent-reference",
                    output=f"{mode}-{trial}", thread_environment={"NUMBA_NUM_THREADS": "48"},
                    profiled_not_performance_evidence=False,
                    exact={"decision_columns_exact": True}, matrices={"exact": True},
                    summary_reports={"exact": True}, components={str(i): 1.0 for i in range(34)},
                    process_wall_seconds=wall, charged_total_seconds=wall - 5,
                    model_steps_seconds=34, validation_seconds=1, prewarm_seconds=2)
    fresh = dict(complete=True, runs=[])
    for trial in range(1, 7):
        order = ["gpu", "candidate"] if trial % 2 else ["candidate", "gpu"]
        fresh["runs"] += [run(mode, trial, 80 if mode == "gpu" else 72) for mode in order]
    cpu = dict(complete=True, runs=[run("regular", trial, 200) for trial in (1, 2)])
    sequence = dict(status="outputs_qualified", source_sha256=source, model_runs=80, evidence_sha256={})
    for mode in ("regular", "candidate"):
        for trial in (1, 2):
            path = results / f"synthetic-{mode}-{trial}.json"
            cases = [dict(scenario="A", output=f"{mode}-{trial}-{i}", components={str(j):1. for j in range(34)}) for i in range(4)]
            path.write_text(json.dumps(dict(execution="fresh",series_trial=trial,mode=mode,runs=cases,
                process_wall_seconds=[190. if mode=="regular" else 75.]*4)))
            sequence["evidence_sha256"][str(path)] = "synthetic-not-a-digest"
    paths = [results / "phase58-test-fresh-summary.json", results / "phase58-test-cpu48-summary.json",
             results / "phase63-sequence-qualification.json"]
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "RESULTS", results)
    monkeypatch.setattr(report, "source_fingerprint", lambda: source)
    monkeypatch.setattr(sys, "argv", ["report_phase63.py", "--tag", "test"])
    return paths, [fresh, cpu, sequence], results


def write_inputs(paths, docs):
    for path, doc in zip(paths, docs):
        path.write_text(json.dumps(doc))


def test_complete_report_discloses_missed_target_and_creates_docs_directory(synthetic_report):
    paths, docs, results = synthetic_report
    write_inputs(paths, docs)
    report.main()
    actual = json.loads((results / "phase63-complete-comparison.json").read_text())
    assert actual["status"] == "replicated_improvement_target_not_met"
    assert actual["cpu_over_phase63"] == 200 / 72
    assert len(actual["components"]) == 34
    assert actual["matched_preparation_default_fresh"]["cpu_over_hybrid"] == 190 / 75
    assert "NOT met" in (results.parent / "docs/phase63-component-comparison.md").read_text()
    with pytest.raises(ValueError, match="replace"):
        report.main()


@pytest.mark.parametrize("defect", ["source", "config", "replay", "reference", "capacity",
                                    "cpu_mode", "diagnostic", "memory"])
def test_invalid_comparison_cannot_be_published(synthetic_report, defect):
    paths, original, _ = synthetic_report
    docs = copy.deepcopy(original)
    run = docs[0]["runs"][0]
    if defect == "source": run["source_sha256"] = {}
    if defect == "config": docs[0]["runs"][1]["configuration_sha256"] = {}
    if defect == "replay": docs[0]["runs"][1]["output"] = run["output"]
    if defect == "reference": run["output"] = run["reference"]
    if defect == "capacity": run["thread_environment"]["NUMBA_NUM_THREADS"] = "1"
    if defect == "cpu_mode": docs[1]["runs"][0]["mode"] = "candidate"
    if defect == "diagnostic": run["profiled_not_performance_evidence"] = True
    if defect == "memory": docs[2]["status"] = "outputs_qualified_memory_growth_requires_investigation"
    write_inputs(paths, docs)
    with pytest.raises(ValueError):
        report.main()


def test_junit_counts_passes_and_discloses_skip(tmp_path):
    path = tmp_path / "test.xml"
    path.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1" time="2.5">'
                    '<testcase><skipped message="expected fixture absent"/></testcase>'
                    '</testsuite></testsuites>')
    result = parse_test_result(path)
    assert result["passed"] == 2 and result["skipped"] == 1 and result["seconds"] == 2.5
    assert result["skip_reasons"] == ["expected fixture absent"]


@pytest.mark.parametrize("failure", ["failures", "errors"])
def test_junit_does_not_certify_failed_tests(tmp_path, failure):
    path = tmp_path / "test.xml"
    path.write_text(f'<testsuite tests="3" {failure}="1"/>')
    with pytest.raises(ValueError, match="Unsuccessful"):
        parse_test_result(path)
