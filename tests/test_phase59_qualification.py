"""Qualification rejects missing evidence, even if the reported times look good."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("qualify59", Path(__file__).parents[1]/"scripts/qualify_phase59.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "verify_reports", lambda *a:{"exact":True,"report_count":24})
    (tmp_path/"benchmark-results").mkdir()
    base = {"proof_gates":{"exact":True}, "phase58_trip_runtime":{
        "chain_events":[{"choosers":100, "failures_before_final_coercion":4}]}}
    candidate = copy.deepcopy(base)
    candidate["phase59_live_mandatory"] = {"captured_input_artifact_read":False, "saved_boundary_answers_used":False,
                                           "live_cpu_logsum_rechecks":[], "live_boundary_cpu_rows":0}
    candidate["phase58_trip_runtime"].update(device_retry_controller=True, entity_store={
        "contract":"phase59-keyed-entity-columns-v1", "tables":dict.fromkeys(("persons", "tours", "trips"), {})})
    for name, proof in (("old.json", base), ("new.json", candidate)):
        (tmp_path/"benchmark-results"/name).write_text(json.dumps(proof))
    runs = []
    for trial in range(1, 7):
        order = ["gpu", "candidate"] if trial % 2 else ["candidate", "gpu"]
        for mode in order:
            live = mode == "candidate"
            command = ["python", "--phase58-trip-runtime", "--report", "new.json" if live else "old.json"]
            if live:
                command += ["--phase59-live-mandatory", "--phase59-device-retries", "--phase59-sparse-matrices"]
            runs.append({"mode":mode, "trial":trial, "position":order.index(mode), "order":order,
                "source_sha256":{"x":"abc"}, "configuration_sha256":{"settings":"def"}, "exact":{"decision_columns_exact":True,
                    "diagnostic_columns":{"logsums":{"max_abs":1e-6,"gate":1e-5}}},
                "matrices":{"exact":True}, "components":{str(i):1.0 for i in range(34)},
                "charged_total_seconds":90. if live else 103., "process_wall_seconds":98. if live else 111.,
                "command":command, "reference":"cpu", "output":f"{trial}-{mode}"})
    scenarios = []
    for name in ("seed991", "seed17", "retry7"):
        run = copy.deepcopy(next(r for r in runs if r["mode"] == "candidate"))
        run.update(scenario_overlay=name, reference="fresh-cpu-"+name)
        scenarios.append({"complete":True, "runs":[run]})
    return {"complete":True,"runs":runs}, scenarios


def test_success_and_honest_target_miss(evidence):
    comparison, scenarios = evidence
    assert q.qualify(comparison, scenarios)["status"] == "replicated_improvement_wall_target_met"
    for run in comparison["runs"]:
        if run["mode"] == "candidate":
            run["process_wall_seconds"] = 105.
    assert q.qualify(comparison, scenarios)["status"] == "replicated_improvement_wall_target_not_met"
    comparison["runs"][1]["process_wall_seconds"] = 120.
    assert q.qualify(comparison, scenarios)["status"] == "correctness_qualified_performance_not_replicated"


@pytest.mark.parametrize("failure", ["source", "config", "scenario_source", "matrices", "decisions", "diagnostic", "order", "count", "profile", "component", "command", "scenario_count", "scenario_duplicate"])
def test_rejects_bad_evidence(evidence, failure):
    comparison, scenarios = evidence
    run = comparison["runs"][1]
    if failure == "source": run["source_sha256"] = {"x":"changed"}
    if failure == "config": run["configuration_sha256"] = {}
    if failure == "scenario_source": scenarios[0]["runs"][0]["source_sha256"] = {}
    if failure == "matrices": run["matrices"]["exact"] = False
    if failure == "decisions": run["exact"]["decision_columns_exact"] = False
    if failure == "diagnostic": run["exact"]["diagnostic_columns"]["logsums"]["max_abs"] = 1.
    if failure == "order": run["position"] = 0
    if failure == "count": comparison["runs"].pop()
    if failure == "profile": run["profiled_not_performance_evidence"] = True
    if failure == "component": run["components"]["0"] = float("nan")
    if failure == "command": run["command"].remove("--phase59-live-mandatory")
    if failure == "scenario_count": scenarios.pop()
    if failure == "scenario_duplicate": scenarios[1] = scenarios[0]
    with pytest.raises(ValueError): q.qualify(comparison, scenarios)


@pytest.mark.parametrize("failure", ["missing_rechecks", "extra_rng", "coverage", "artifact", "entity", "gate"])
def test_live_implementation_evidence_is_required(evidence, tmp_path, failure):
    comparison, scenarios = evidence
    path = tmp_path/"benchmark-results/new.json"
    proof = json.loads(path.read_text())
    mandatory = proof["phase59_live_mandatory"]
    if failure == "missing_rechecks": mandatory.pop("live_cpu_logsum_rechecks")
    if failure == "extra_rng":
        mandatory.update(live_boundary_cpu_rows=1, live_cpu_logsum_rechecks=[{"choosers":1,"additional_random_draws":1}])
    if failure == "coverage": mandatory["live_boundary_cpu_rows"] = 1
    if failure == "artifact": mandatory["captured_input_artifact_read"] = True
    if failure == "entity": proof["phase58_trip_runtime"]["entity_store"]["tables"].pop("trips")
    if failure == "gate": proof["proof_gates"]["exact"] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError): q.qualify(comparison, scenarios)


def test_summary_verifier_failure_rejects_qualification(evidence, monkeypatch):
    def reject(*args):
        raise ValueError("changed published report")
    monkeypatch.setattr(q, "verify_reports", reject)
    with pytest.raises(ValueError, match="published report"):
        q.qualify(*evidence)
