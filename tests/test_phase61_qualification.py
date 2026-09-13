import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("q61",Path(__file__).parents[1]/"scripts/qualify_phase61.py")
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


@pytest.fixture
def evidence(monkeypatch):
    proof={"phase58_trip_runtime":{"chain_events":[{"choosers":10,"failures_before_final_coercion":2}]}}
    monkeypatch.setattr(q,"audit",lambda r,c:copy.deepcopy(proof))
    runs=[]
    for trial in range(1,7):
        order=["gpu","candidate"] if trial%2 else ["candidate","gpu"]
        for position,mode in enumerate(order):
            runs.append(dict(trial=trial,mode=mode,order=order,position=position,source_sha256={"a":"b"},
                configuration_sha256={"c":"d"},process_wall_seconds=74. if mode=="candidate" else 90.,
                charged_total_seconds=68. if mode=="candidate" else 84.,components={str(i):1. for i in range(34)},
                summary_reports={"exact":True},output=str(trial)+mode))
    scenarios=[]
    for name in ("seed991","seed17","retry7"):
        run=copy.deepcopy(runs[1])
        run.update(scenario_overlay=name,reference="regular-"+name)
        scenarios.append({"complete":True,"runs":[run]})
    return {"complete":True,"runs":runs},scenarios


def test_honest_median_target_and_losses(evidence):
    a,b=evidence
    assert q.qualify(a,b)["median_wall_under_75"]
    for run in a["runs"]:
        if run["mode"]=="candidate":run["process_wall_seconds"]=80.
    assert q.qualify(a,b)["status"]=="replicated_improvement_wall_target_not_met"
    a["runs"][1]["process_wall_seconds"]=91.
    assert q.qualify(a,b)["status"]=="correctness_qualified_performance_not_replicated"


@pytest.mark.parametrize("bad",["incomplete","count","source","config","order","position","scenario_source","scenario_duplicate","scenario_count"])
def test_incomplete_evidence_rejected(evidence,bad):
    a,b=evidence
    if bad=="incomplete":a["complete"]=False
    if bad=="count":a["runs"].pop()
    if bad=="source":a["runs"][1]["source_sha256"]={}
    if bad=="config":a["runs"][1]["configuration_sha256"]={}
    if bad=="order":a["runs"].reverse()
    if bad=="position":a["runs"][1]["position"]=0
    if bad=="scenario_source":b[0]["runs"][0]["source_sha256"]={}
    if bad=="scenario_duplicate":b[1]=b[0]
    if bad=="scenario_count":b.pop()
    with pytest.raises(ValueError):q.qualify(a,b)


@pytest.mark.parametrize("bad",[None,"capture","command","features","events","answers","entities","normals","uniforms"])
def test_live_audit(monkeypatch,bad):
    names=[str(i) for i in range(34)]
    report=dict(enabled=True,diagnostic_capture=False,
        features=["skims","timetable","tour_modes","entities","normals","uniforms","labels","packing"],
        events=[dict(step=n,saved_answers_read=False) for n in names],skim_calls=1,availability_events=[dict(backend="cpu",cpu_threads=24)],
        shared_input_events=[dict(table=n,columns=["id"],rows=1) for n in ("tours","trips")],
        tour_mode_events=[{}],tour_rng_events=[{}],normal_events=[{}],uniform_events=[{}])
    proof={"phase61_shared_inputs":report}
    monkeypatch.setattr(q,"audit_phase60",lambda *a:proof)
    run={"command":["--phase61-features"],"components":dict.fromkeys(names,1.),"thread_environment":{"OMP_WAIT_POLICY":"PASSIVE"}}
    if bad=="capture":report["diagnostic_capture"]=True
    if bad=="command":run["command"]=[]
    if bad=="features":report["features"].pop()
    if bad=="events":report["events"].pop()
    if bad=="answers":report["events"][0]["saved_answers_read"]=True
    if bad=="entities":report["shared_input_events"].pop()
    if bad=="normals":report["normal_events"]=[]
    if bad=="uniforms":report["uniform_events"]=[]
    if bad:
        with pytest.raises(ValueError):q.audit(run,True)
    else:assert q.audit(run,True) is proof
