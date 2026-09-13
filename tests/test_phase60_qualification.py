import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("q60",Path(__file__).parents[1]/"scripts/qualify_phase60.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


@pytest.fixture
def evidence(monkeypatch):
    proof = {"phase58_trip_runtime":{"chain_events":[{"choosers":10,"failures_before_final_coercion":2}]}}
    monkeypatch.setattr(q,"audit",lambda r,c:copy.deepcopy(proof))
    runs = []
    for t in range(1,7):
        order = ["gpu","candidate"] if t%2 else ["candidate","gpu"]
        for position,mode in enumerate(order):
            runs.append({"trial":t,"mode":mode,"order":order,"position":position,
                "source_sha256":{"source":"abc"},"configuration_sha256":{"config":"def"},
                "process_wall_seconds":94. if mode == "candidate" else 110.,
                "charged_total_seconds":87. if mode == "candidate" else 103.,
                "components":{str(i):1. for i in range(34)},"summary_reports":{"exact":True},"output":str(t)+mode})
    scenarios = []
    for name in ("seed991","seed17","retry7"):
        r = copy.deepcopy(next(r for r in runs if r["mode"] == "candidate"))
        r.update(scenario_overlay=name,reference="regular-"+name)
        scenarios.append({"complete":True,"runs":[r]})
    return {"complete":True,"runs":runs},scenarios


def test_all_targets_and_honest_miss(evidence):
    a,b = evidence
    assert q.qualify(a,b)["all_wall_runs_under_95"]
    a["runs"][1]["process_wall_seconds"] = 101.
    assert q.qualify(a,b)["status"] == "replicated_improvement_wall_target_not_met"
    a["runs"][1]["process_wall_seconds"] = 111.
    assert q.qualify(a,b)["status"] == "correctness_qualified_performance_not_replicated"


@pytest.mark.parametrize("bad",["incomplete","count","source","config","order","position","scenario_source","scenario_duplicate","scenario_count"])
def test_incomplete_evidence_is_rejected(evidence,bad):
    a,b = evidence
    if bad == "incomplete": a["complete"] = False
    if bad == "count": a["runs"].pop()
    if bad == "source": a["runs"][1]["source_sha256"] = {}
    if bad == "config": a["runs"][1]["configuration_sha256"] = {}
    if bad == "order": a["runs"].reverse()
    if bad == "position": a["runs"][1]["position"] = 0
    if bad == "scenario_source": b[0]["runs"][0]["source_sha256"] = {}
    if bad == "scenario_duplicate": b[1] = b[0]
    if bad == "scenario_count": b.pop()
    with pytest.raises(ValueError): q.qualify(a,b)


@pytest.mark.parametrize("bad",[None,"instrumented","events","saved_answers","threads","capacity","initial_mask","command"])
def test_live_preparation_audit(monkeypatch,bad):
    names = ["non_mandatory_tour_frequency"]+[str(i) for i in range(33)]
    proof = {"phase60_preparation":{"enabled":True,"events":[
        {"step":n,"saved_choices_read":False,"frequency_threads":24 if n == names[0] else None} for n in names]}}
    run = {"command":["--phase60-preparation","--phase60-frequency-threads","24"],
           "components":dict.fromkeys(names,1.),"thread_environment":{"NUMBA_NUM_THREADS":"48","CHOICEFORGE_NUMBA_INITIAL_THREADS":"1"}}
    monkeypatch.setattr(q,"audit_run",lambda *a,**k:proof)
    if bad == "instrumented": proof["phase60_frequency_control"] = {"instrumented_not_performance":True}
    if bad == "events": proof["phase60_preparation"]["events"].pop()
    if bad == "saved_answers": proof["phase60_preparation"]["events"][0]["saved_choices_read"] = True
    if bad == "threads": proof["phase60_preparation"]["events"][0]["frequency_threads"] = 1
    if bad == "capacity": run["thread_environment"]["NUMBA_NUM_THREADS"] = "1"
    if bad == "initial_mask": run["thread_environment"]["CHOICEFORGE_NUMBA_INITIAL_THREADS"] = "48"
    if bad == "command": run["command"].remove("--phase60-preparation")
    if bad:
        with pytest.raises(ValueError): q.audit(run,True)
    else:
        assert q.audit(run,True) is proof
