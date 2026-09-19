import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from report_phase62_batches import qualify


@pytest.fixture
def evidence(tmp_path):
    proof = tmp_path/"synthetic-test-proof.json"
    proof.write_text(json.dumps({"proof_gates":{"test":True},"phase62_reusable_execution":{"enabled":True}}))
    cases = [("candidate","fresh"),("candidate","batch"),("regular","fresh"),("regular","batch")]
    documents = []
    for trial in (1,2):
        for position,(mode,execution) in enumerate(cases if trial==1 else cases[::-1]):
            runs = []
            for name in ("A1","B","A2"):
                runs.append(dict(name=name,reference="baseline-B" if name=="B" else "baseline-A",
                    output=f"{trial}-{position}-{name}",report=str(proof),exact={"decision_columns_exact":True},
                    matrices={"exact":True},summary_reports={"exact":True},components={str(i):1. for i in range(34)},
                    environment={"NUMBA_NUM_THREADS":"48","OMP_WAIT_POLICY":"PASSIVE"}))
            worker = dict(complete=True,generated_program_sha256={"test":"test"},
                input_tables=dict(hits=6,returns_private_copies=True,retained_bytes=100,limit_bytes=1024**3),
                runs=[dict(reset_modules=10,reused_cpu_programs=3,reused_source_keyed_cuda_programs=2) for _ in range(3)])
            documents.append(dict(complete=True,sequence="A-B-A",setup_included=True,skim_cache="none",mode=mode,
                execution=execution,source_sha256={"source":"hash"},data_sha256={"data":"hash"},
                configuration_sha256={"config":"hash"},runs=runs,worker=worker,series_trial=trial,
                series_position=position,started_at_ns=trial*10+position,batch_process_wall_seconds=200.))
    return documents


def test_complete_batch_design(evidence):
    assert qualify(evidence,{"source":"hash"})["all_output_gates_pass"]


@pytest.mark.parametrize("bad",["order","timestamp","source","data","decisions","replay","time","copy","reuse"])
def test_bad_batch_evidence_rejected(evidence,bad):
    docs = copy.deepcopy(evidence)
    if bad=="order":docs[0]["series_position"]=3
    if bad=="timestamp":docs[0]["started_at_ns"]=99
    if bad=="source":docs[0]["source_sha256"]={}
    if bad=="data":docs[1]["data_sha256"]={}
    if bad=="decisions":docs[0]["runs"][0]["exact"]["decision_columns_exact"]=False
    if bad=="replay":docs[0]["runs"][1]["output"]=docs[0]["runs"][0]["output"]
    if bad=="time":docs[0]["batch_process_wall_seconds"]=float("nan")
    if bad=="copy":docs[1]["worker"]["input_tables"]["returns_private_copies"]=False
    if bad=="reuse":docs[1]["worker"]["runs"][1]["reused_cpu_programs"]=0
    with pytest.raises(ValueError):qualify(docs,{"source":"hash"})
