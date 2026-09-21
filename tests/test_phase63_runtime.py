import os
from types import SimpleNamespace

import pandas as pd
import pytest

from choiceforge.phase63_runtime import SpecificationFiles, Runtime


def test_spec_file_cache_private_values_and_same_size_source_change(tmp_path):
    source = tmp_path / "model.csv"
    source.write_text("x\n1\n")
    fs = SimpleNamespace(get_config_file_path=lambda name:tmp_path/name)
    calls = []
    def read_model_spec(filesystem, file_name):
        calls.append(1)
        return pd.read_csv(filesystem.get_config_file_path(file_name))
    cache = SpecificationFiles()
    first = cache.read(read_model_spec,fs,"model.csv")
    first.loc[0,"x"] = 99
    assert cache.read(read_model_spec,fs,"model.csv").x.tolist() == [1]
    stamp = source.stat()
    source.write_text("x\n2\n")
    os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert cache.read(read_model_spec,fs,"model.csv").x.tolist() == [2]
    assert len(calls)==2 and cache.hits==1


def test_spec_cache_detaches_index_backing_arrays(tmp_path):
    source = tmp_path/"model.csv"
    source.write_text("name,x\na,1\nb,2\n")
    fs = SimpleNamespace(get_config_file_path=lambda name:tmp_path/name)
    def read_model_spec(filesystem,file_name):
        return pd.read_csv(filesystem.get_config_file_path(file_name),index_col=0)
    cache = SpecificationFiles()
    first = cache.read(read_model_spec,fs,"model.csv")
    first.index.values[0] = "damaged"
    first.columns.values[0] = "damaged"
    second = cache.read(read_model_spec,fs,"model.csv")
    assert second.index.tolist()==["a","b"] and second.columns.tolist()==["x"]
    second.index.values[1] = "damaged"
    assert cache.read(read_model_spec,fs,"model.csv").index.tolist()==["a","b"]


def test_spec_cache_tracks_resolved_overlay_and_settings(tmp_path):
    for name,value in (("a.csv",1),("b.csv",2)):
        (tmp_path/name).write_text(f"x\n{value}\n")
    fs = SimpleNamespace(get_config_file_path=lambda name:tmp_path/name)
    def read_model_coefficients(filesystem,model_settings=None,file_name=None):
        name = file_name if model_settings is None else model_settings["COEFFICIENTS"]
        return pd.read_csv(filesystem.get_config_file_path(name))
    cache = SpecificationFiles()
    settings = {"COEFFICIENTS":"a.csv"}
    assert cache.read(read_model_coefficients,fs,settings).x.tolist()==[1]
    settings["COEFFICIENTS"] = "b.csv"
    assert cache.read(read_model_coefficients,fs,settings).x.tolist()==[2]


def test_runtime_restoration_after_failure():
    from activitysim.core import simulate
    from sharrow.flows import Flow
    methods = (simulate.read_model_spec,simulate.read_model_coefficients,Flow.init_sub_funcs)
    runtime = Runtime()
    with pytest.raises(RuntimeError):
        with runtime.for_step(None,"test"):
            raise RuntimeError("deliberate")
    assert methods == (simulate.read_model_spec,simulate.read_model_coefficients,Flow.init_sub_funcs)


def test_rss_only_scope_restores_memory_diagnostics():
    from activitysim.core import mem
    prior = mem.USS
    state = SimpleNamespace(settings=SimpleNamespace(chunk_size=0,chunk_training_mode="disabled"))
    runtime = Runtime("rss")
    with pytest.raises(RuntimeError):
        with runtime.for_step(state,"failure"):
            assert mem.USS is False
            raise RuntimeError("deliberate")
    assert mem.USS == prior
    assert runtime.memory_events[0]["end_rss"] > 0
    assert runtime.memory_events[0]["uss_in_step"] == "not sampled"


@pytest.mark.parametrize("size,training",[(100,"disabled"),(0,"training")])
def test_rss_only_rejects_adaptive_chunking(size,training):
    from activitysim.core import mem
    prior = mem.USS
    state = SimpleNamespace(settings=SimpleNamespace(chunk_size=size,chunk_training_mode=training))
    with pytest.raises(ValueError):
        with Runtime("rss").for_step(state,"unsafe"):
            pytest.fail("guard was bypassed")
    assert mem.USS == prior


def test_cpu_runner_scope_restores_after_failure(monkeypatch):
    from activitysim.core.workflow.runner import Runner
    calls = []
    def original(runner,name):
        calls.append(name)
        raise RuntimeError("deliberate")
    monkeypatch.setattr(Runner,"by_name",original)
    with pytest.raises(RuntimeError):
        with Runtime("files").cpu_steps():
            Runner.by_name(SimpleNamespace(_obj=None),"test")
    assert Runner.by_name is original and calls == ["test"]


def test_diagnostic_failure_still_restores_hooks(monkeypatch):
    import psutil
    from activitysim.core import mem, simulate
    original,prior = simulate.read_model_spec,mem.USS
    class BrokenProcess:
        def memory_info(self):
            raise OSError("diagnostic failure")
    monkeypatch.setattr(psutil,"Process",BrokenProcess)
    state = SimpleNamespace(settings=SimpleNamespace(chunk_size=0,chunk_training_mode="disabled"))
    with pytest.raises(OSError,match="diagnostic failure"):
        with Runtime("files,rss").for_step(state,"failure"):
            pytest.fail("diagnostic setup should fail")
    assert mem.USS==prior and simulate.read_model_spec is original
