import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("report59", Path(__file__).parents[1]/"scripts/report_phase59_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("failure", [None, "incomplete", "few", "mode", "source", "steps", "matrices", "time", "profile"])
def test_complete_regular_comparison(monkeypatch, failure):
    monkeypatch.setattr(module, "verify_reports", lambda *args:{"exact":True})
    qualified = {"status":"replicated_improvement_wall_target_not_met", "source_sha256":{"a":"b"}, "configuration_sha256":{"c":"d"},
                 "component_medians":{str(i):{"gpu":3.,"candidate":2.} for i in range(34)},
                 "medians":{mode:{"charged_total_seconds":value,"process_wall_seconds":value+7}
                            for mode, value in (("gpu", 104.), ("candidate", 102.))}}
    run = {"mode":"regular", "source_sha256":{"a":"b"}, "configuration_sha256":{"c":"d"}, "exact":{"decision_columns_exact":True},
           "matrices":{"exact":True}, "components":{str(i):6. for i in range(34)},
           "charged_total_seconds":300., "process_wall_seconds":307., "reference":"cpu", "output":"new-cpu"}
    regular = {"complete":True,"runs":[copy.deepcopy(run),copy.deepcopy(run)]}
    target = regular["runs"][0]
    if failure == "incomplete": regular["complete"] = False
    if failure == "few": regular["runs"].pop()
    if failure == "mode": target["mode"] = "candidate"
    if failure == "source": target["source_sha256"] = {}
    if failure == "steps": target["components"].pop("0")
    if failure == "matrices": target["matrices"]["exact"] = False
    if failure == "time": target["process_wall_seconds"] = float("nan")
    if failure == "profile": target["profiled_not_performance_evidence"] = True
    if failure:
        with pytest.raises(ValueError): module.report(qualified, regular)
    else:
        result = module.report(qualified, regular)
        assert len(result["components"]) == 34
        assert result["components"][0]["regular_over_candidate"] == 3.
        assert result["regular_over_candidate"]["process_wall_seconds"] == 307/109
