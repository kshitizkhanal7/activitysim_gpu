import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from qualify_phase64_campaign import qualify
from run_phase64_campaign import stages
from test_phase63_sequences import evidence as old_evidence


@pytest.fixture
def evidence(old_evidence):
    docs=copy.deepcopy(old_evidence)
    pipeline=dict(enabled=True,features=["expressions","inputs","location_boundary"],saved_answers_read=False,
        location_boundary=dict(contexts_outstanding=0,saved_answers_read=False,
                               events=[dict(complete=True,rng_unchanged=True,saved_answers_read=False)]),
        code_entries=2,code_capacity=8192,code_hits=10,
        inputs=dict(hits=3,misses=0,saved_answers_read=False,
                    events=[dict(private_copy=True,source_sha256="s",artifact_sha256="a")]*3))
    matched=[]
    memory=[]
    for d in docs:
        d["phase64_features"]="expressions,inputs,location_boundary"
        d["diagnostic_not_performance"]=False
        for r in d["runs"]:
            r["cpu_pipeline_preparation"]=copy.deepcopy(pipeline)
        for w in d["workers"]:
            w["input_tables"]["retained_bytes"]=0
        if d["execution"]=="batch":
            if d["mode"]=="candidate":
                memory.append(d)
            continue
        d["series_position"]= (0 if d["mode"]=="regular" else 1) if d["series_trial"]==1 else (0 if d["mode"]=="candidate" else 1)
        d["started_at_ns"]=d["series_trial"]*10+d["series_position"]
        d["runs"]=[d["runs"][0],d["runs"][4]]
        d["sequence"]=["A","A"]
        d["workers"]=d["workers"][:2]
        d["process_wall_seconds"]=[100.,100.]
        d["batch_process_wall_seconds"]=200.
        matched.append(d)
    source=docs[0]["source_sha256"]
    def fresh(mode,trial,index):
        return dict(mode=mode,trial=trial,output=f"fresh-{index}",reference="reference-A",
            exact={"decision_columns_exact":True},matrices={"exact":True},summary_reports={"exact":True},
            components={str(i):1. for i in range(34)},source_sha256=source,
            configuration_sha256=docs[0]["configuration_sha256"],profiled_not_performance_evidence=False,
            thread_environment={"NUMBA_NUM_THREADS":"48","OMP_WAIT_POLICY":"PASSIVE"},
            command=["--report","synthetic"],process_wall_seconds=75.,charged_total_seconds=70.)
    runs=[]
    for t in range(1,7):
        for m in (["gpu","candidate"] if t%2 else ["candidate","gpu"]):
            runs.append(fresh(m,t,len(runs)))
    cpu=[fresh("regular",t,100+t) for t in (1,2)]
    proof=dict(proof_gates={"test":True},phase64_pipeline=pipeline)
    return [dict(complete=True,runs=runs),dict(complete=True,runs=cpu),matched,memory,source,lambda name:proof]


def test_complete_42_model_design(evidence):
    result=qualify(*evidence)
    assert result["model_runs"]==42
    assert result["status"]=="outputs_and_memory_qualified"
    assert not result["under_70"] and not result["all_pairs_faster"]


@pytest.mark.parametrize("bad",["old_order","source","replay","fairness","artifact","active_gpu","clock","scenario","proof","boundary","boundary_rng","boundary_unused","missing_wait_policy"])
def test_rejects_invalid_evidence(evidence,bad):
    fresh,cpu,matched,memory,source,load=evidence
    if bad=="old_order": fresh["runs"][0]["mode"]="candidate"
    if bad=="source": fresh["runs"][0]["source_sha256"]={}
    if bad=="replay": fresh["runs"][0]["output"]=fresh["runs"][1]["output"]
    if bad=="fairness": matched[0]["phase64_features"]="inputs"
    if bad=="artifact": load("")["phase64_pipeline"]["inputs"]["hits"]=0
    if bad=="active_gpu": memory[0]["runs"][1]["memory_before"]["gpu_pool_used_bytes"]=1
    if bad=="clock": matched[0]["process_wall_seconds"][0]=float("nan")
    if bad=="scenario": memory[0]["runs"][1]["reference"]="reference-A"
    if bad=="proof": load("")["proof_gates"]={}
    if bad=="missing_wait_policy": cpu["runs"][0]["thread_environment"].pop("OMP_WAIT_POLICY")
    if bad=="boundary": load("")["phase64_pipeline"]["location_boundary"]["contexts_outstanding"]=1
    if bad=="boundary_rng": load("")["phase64_pipeline"]["location_boundary"]["events"][0]["rng_unchanged"]=False
    if bad=="boundary_unused": load("")["phase64_pipeline"]["location_boundary"]["events"]=[]
    with pytest.raises(ValueError):
        qualify(*evidence)


def test_finite_growth_failure_not_hidden(evidence):
    evidence[3][0]["runs"][-1]["memory_after"]["uss_bytes"]=1024**3
    assert qualify(*evidence)["status"]=="memory_investigation_required"


def test_campaign_predeclared_8_stages():
    plan=list(stages("test","expressions,inputs"))
    assert len(plan)==8 and len({str(p) for _,p in plan})==8
    assert all("expressions,inputs" in c for c,p in plan)
