import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("qualify58", Path(__file__).resolve().parents[1]/"scripts/qualify_phase58.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    def run(mode, trial, seconds, order):
        return {"mode":mode, "trial":trial, "order":order,
            "charged_total_seconds":seconds, "process_wall_seconds":seconds+8,
            "model_steps_seconds":seconds-1, "components":{str(i):seconds/34 for i in range(34)},
            "exact":{"success":True,"decision_columns_exact":True},"source_sha256":{"file":"hash"}}
    controls={"complete":True,"runs":[run(mode,i,300 if mode=="regular" else 111,["cpu","gpu","regular"])
                                       for i in (1,2) for mode in ("cpu","gpu","regular")]}
    matched={"complete":True,"runs":[run(mode,i,111 if mode=="gpu" else 103,
                                          ["gpu","candidate"] if i%2 else ["candidate","gpu"])
                                      for i in range(1,7) for mode in ("gpu","candidate")]}
    return controls,matched


def test_complete_balanced_series_and_distinct_target_gate():
    controls,matched=fixture()
    assert module.summarize(controls,matched)["status"]=="replicated_improvement_target_met"
    assert module.summarize(controls,matched,target=100)["status"]=="replicated_improvement_target_not_met"


@pytest.mark.parametrize("failure", ["profile", "source", "decision", "missing_trial"])
def test_bad_evidence_is_not_qualified(failure):
    controls,matched=fixture()
    if failure=="profile":matched["runs"][0]["profiled_not_performance_evidence"]=True
    if failure=="source":matched["runs"][0]["source_sha256"]={"file":"changed"}
    if failure=="decision":matched["runs"][0]["exact"]["decision_columns_exact"]=False
    if failure=="missing_trial":matched["runs"].pop()
    with pytest.raises(ValueError):module.summarize(controls,matched)


def test_no_qualification_when_one_pair_is_slower():
    controls,matched=fixture()
    matched["runs"][1]["charged_total_seconds"]=112
    assert module.summarize(controls,matched)["status"]=="not_qualified"
