import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from phase63_commands import candidate_command, ROOT, PROJECT, environment
from run_phase63_campaign import completed
import phase63_commands


def test_portable_command_preserves_live_reference_and_config_priority(tmp_path):
    overlay = tmp_path/"scenario"
    command = candidate_command(tmp_path/"output",tmp_path/"reference",tmp_path/"report.json",
                                tmp_path/"kernels",tmp_path/"checkpoint.json",10000,overlay)
    assert command[command.index("--config-overlay")+1]==str(overlay)
    assert command[command.index("--households-sample-size")+1]=="10000"
    assert command[command.index("--phase63-features")+1]=="plans,files,rss"
    assert command[command.index("--reference-pipeline")+1]==str(tmp_path/"reference/pipeline.parquetpipeline")
    assert command[command.index("--inputs")+1]==str(ROOT/"phase59-no-reference-artifact")
    assert "--phase59-live-mandatory" in command and "--phase59-scenario-gates" in command
    assert not (tmp_path/"output").exists()


def test_cpu_and_hybrid_environment_have_equal_capacity_and_math_library_policy():
    cpu,gpu = environment(),environment(True)
    for name in ("NUMBA_NUM_THREADS","OMP_WAIT_POLICY","OMP_NUM_THREADS","MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        assert cpu[name]==gpu[name]
    assert cpu["NUMBA_NUM_THREADS"]=="48" and cpu["CHOICEFORGE_STRICT_CUDA_CANDIDATE"]=="0"
    assert gpu["CHOICEFORGE_NUMBA_INITIAL_THREADS"]=="1"


@pytest.mark.parametrize("document",[{"complete":True},{"complete":False},
    {"complete":True,"design":"test","runs":[]}])
def test_campaign_never_skips_unproven_receipts(tmp_path,document):
    receipt = tmp_path/"receipt.json"
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        completed(receipt)


def test_checkpoint_output_path_avoids_deep_public_project_directory(monkeypatch):
    monkeypatch.setattr(phase63_commands,"OUTPUTS",Path("C:/short-replica/phase63-runs"))
    result = phase63_commands.output_directory("o-phase63-test-candidate-fresh-2-10-A")
    assert "phase9-mtc-full" not in str(result)
    assert len(str(result/"pipeline.parquetpipeline/workplace_destination_size/initialize_households.parquet"))<250


@pytest.mark.parametrize("name",["../escape","a/b","a\\b"])
def test_output_path_rejects_unsafe_names(name):
    with pytest.raises(ValueError,match="Invalid output"):
        phase63_commands.output_directory(name)


@pytest.mark.skipif(phase63_commands.os.name!="nt",reason="Windows legacy path contract")
def test_overlong_checkpoint_path_fails_before_starting_model(monkeypatch):
    monkeypatch.setattr(phase63_commands,"OUTPUTS",Path("C:/"+"x"*180))
    with pytest.raises(ValueError,match="too long"):
        phase63_commands.output_directory("o-phase63-test-candidate-fresh-2-10-A")
