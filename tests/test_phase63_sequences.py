import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from qualify_phase63_sequences import CASES, SEQUENCE, qualify


@pytest.fixture
def evidence(tmp_path):
    proof = tmp_path/"synthetic-not-benchmark.json"
    proof.write_text(json.dumps(dict(proof_gates={"test":True},
        phase63_durable_execution=dict(enabled=True,features=["plans","files","rss"]))))
    documents = []
    for trial in (1,2):
        for position,(mode,execution) in enumerate(CASES if trial==1 else CASES[::-1]):
            runs = []
            for index,key in enumerate(SEQUENCE.split(",")):
                runs.append(dict(scenario=key,reference="reference-"+key,output=f"{trial}-{position}-{index}",
                    exact={"decision_columns_exact":True},matrices={"exact":True},summary_reports={"exact":True},
                    components={str(i):1. for i in range(34)},environment={"NUMBA_NUM_THREADS":"48","OMP_WAIT_POLICY":"PASSIVE"},
                    report=str(proof),memory_before={"uss_bytes":100,"gpu_pool_used_bytes":0},memory_after={"uss_bytes":100,"gpu_pool_used_bytes":0},
                    rss_sampling={"observed_peak_rss_bytes":200,"samples":10},
                    cpu_preparation={"enabled":True,"features":["plans","files","rss"]},reset_modules=10,
                    reused_cpu_programs=3,reused_source_keyed_cuda_programs=2))
            workers = [dict(complete=True,generated_program_sha256={"program":"hash"},
                            memory_after_final_reset={"gpu_pool_used_bytes":0},
                            input_tables=dict(returns_private_copies=True,retained_bytes=100,limit_bytes=1024**3))
                       for _ in range(1 if execution=="batch" else 10)]
            documents.append(dict(series_trial=trial,series_position=position,mode=mode,execution=execution,
                started_at_ns=trial*10+position,complete=True,setup_included=True,verification_outside_timed_region=True,
                sequence=SEQUENCE.split(","),features="plans,files,rss",batch_process_wall_seconds=1000.,
                workers=workers,process_wall_seconds=[1000./len(workers)]*len(workers),runs=runs,
                source_sha256={"source":"hash"},data_sha256={"data":"hash"},
                configuration_sha256={"config":"hash"},reference_output_sha256={"reference":"hash"}))
    return documents


def test_complete_design(evidence):
    report = qualify(evidence)
    assert report["model_runs"]==80 and report["status"]=="outputs_qualified"


@pytest.mark.parametrize("bad",["order","clock","source","data","reference","replay","decisions","cpu_fairness","memory","steps","copy","reuse"])
def test_bad_evidence_rejected(evidence,bad):
    docs = copy.deepcopy(evidence)
    run = docs[0]["runs"][0]
    if bad=="order":docs[0]["series_position"]=8
    if bad=="clock":docs[0]["batch_process_wall_seconds"]=float("nan")
    if bad=="source":docs[1]["source_sha256"]={}
    if bad=="data":docs[1]["data_sha256"]={}
    if bad=="reference":run["reference"]=run["output"]
    if bad=="replay":docs[0]["runs"][1]["output"]=run["output"]
    if bad=="decisions":run["exact"]["decision_columns_exact"]=False
    if bad=="cpu_fairness":docs[2]["runs"][0]["cpu_preparation"]["features"]=[]
    if bad=="memory":run["memory_before"]["uss_bytes"]=0
    if bad=="steps":run["components"]={}
    if bad=="copy":docs[0]["workers"][0]["input_tables"]["returns_private_copies"]=False
    if bad=="reuse":docs[1]["runs"][1]["reused_cpu_programs"]=0
    with pytest.raises(ValueError):
        qualify(docs)


def test_growth_is_reported_as_unresolved_not_hidden(evidence):
    evidence[1]["runs"][-1]["memory_after"]["gpu_pool_used_bytes"] = 1024**3
    result = qualify(evidence)
    assert result["status"]=="outputs_qualified_memory_growth_requires_investigation"
    assert not result["memory"][0]["within_retention_limits"]
