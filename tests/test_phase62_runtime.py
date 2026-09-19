from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from choiceforge.phase62_runtime import (SpecificationCompiler, DirectoryProbes,
                                        DemandInputs, Runtime, node_sources)
from choiceforge import sharrow_ir, sharrow_cuda


def test_specification_exact_and_mutation_isolation():
    compiler = SpecificationCompiler()
    compiler.original = sharrow_ir.specification_ir
    spec = pd.DataFrame({"Expression":["a + b * 2", "", "nan", "df.x > 0"],
                         "Label":["one","two","three","four"],
                         "A":[1.,2.,3.,4.], "B":["beta",0,0,1]})
    assert compiler(spec) == compiler.original(spec)
    first = compiler(spec)
    first["terms"][0]["tree"]["op"] = "broken"
    spec.loc[0,"A"] = 9.
    assert compiler(spec) == compiler.original(spec)
    assert compiler.hits > 0


def test_source_traversal_exact():
    for expression in ("a+b*2", "df.x.clip(lower=0, upper=3)", "a > 0", "od_skims['DIST']"):
        tree = sharrow_ir.expression_ir(expression)
        assert list(node_sources(tree)) == list(sharrow_cuda._node_sources(tree))


def test_probe_failure_and_replacement(tmp_path):
    directory = tmp_path / "cache"
    calls = []
    def original(locator):
        calls.append(1)
        directory.mkdir(exist_ok=True)
    locator = SimpleNamespace(get_cache_path=lambda:str(directory))
    probes = DirectoryProbes(original)
    probes.ensure(locator)
    probes.ensure(locator)
    assert len(calls) == 1
    # Move only this test's empty temporary directory; inode replacement reprobes.
    directory.rename(tmp_path / "old-cache")
    directory.mkdir()
    probes.ensure(locator)
    assert len(calls) == 2
    def failure(locator):
        raise PermissionError("test")
    probes = DirectoryProbes(failure)
    with pytest.raises(PermissionError):
        probes.ensure(locator)
    assert not probes.identities


def test_scope_restoration_on_failure():
    from numba.core.caching import _CacheLocator
    original = (_CacheLocator.ensure_cache_path, sharrow_ir.specification_ir, sharrow_cuda._node_sources)
    runtime = Runtime(SimpleNamespace(), "plans")
    with pytest.raises(RuntimeError):
        with runtime.for_step(None,"test"):
            raise RuntimeError("test")
    assert original == (_CacheLocator.ensure_cache_path, sharrow_ir.specification_ir, sharrow_cuda._node_sources)


def test_demand_only_live_referenced_columns_and_mutation():
    shared = DemandInputs()
    full = pd.DataFrame({"x":[1.,2.], "unused":[3.,4.]},index=[10,20])
    state = SimpleNamespace(get_dataframe=lambda _:full)
    shared.publish(state,"trips")
    assert not shared.store.tables
    doc = sharrow_ir.specification_ir(pd.DataFrame({"Expression":["x"],"A":[1.]}))
    frame = full.copy()
    env = {"df":{n:frame[n] for n in frame}, **dict(frame.items())}
    shared.bind("trips",frame,env,doc)
    np.testing.assert_array_equal(env["x"].get(),[1.,2.])
    assert list(shared.store.tables["trips"]["host"]) == ["x"]
    full.loc[10,"x"] = 8.
    frame = full.copy()
    env = {"df":{n:frame[n] for n in frame}, **dict(frame.items())}
    shared.bind("trips",frame,env,doc)
    np.testing.assert_array_equal(env["x"].get(),[8.,2.])


def test_content_addressed_skim_reuse_cannot_hide_mutation():
    import importlib.util
    from pathlib import Path
    from choiceforge.cuda_backend import _cupy
    path = Path(__file__).parents[1]/"scripts/phase62_batch_worker.py"
    spec = importlib.util.spec_from_file_location("phase62_worker_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = module.SkimPool(limit=32)
    values = np.arange(4,dtype=np.float32)
    first = pool.upload(_cupy(),values)
    assert pool.upload(_cupy(),values.copy()) is first
    values[0] = 99
    changed = pool.upload(_cupy(),values)
    assert changed is not first
    np.testing.assert_array_equal(changed.get(),values)
    pool.upload(_cupy(),np.arange(4,dtype=np.float32)+7)
    assert pool.bytes<=32 and pool.hits==1


@pytest.mark.parametrize("name",["trip_mode_choice","tour_mode_choice","cdap_indiv_and_hhsize1"])
def test_public_specification_document_identical(name):
    from pathlib import Path
    path = Path(__file__).parents[1]/"benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs"/(name+".csv")
    if not path.exists():
        pytest.skip("public benchmark not installed")
    spec = pd.read_csv(path,comment="#")
    compiler = SpecificationCompiler()
    compiler.original = sharrow_ir.specification_ir
    try:
        expected = compiler.original(spec)
    except sharrow_ir.ExpressionUnsupported:
        # Some upstream specifications need a preparation pass. The optimized
        # parser must fail at the same unsupported expression, not relax the IR.
        with pytest.raises(sharrow_ir.ExpressionUnsupported):
            compiler(spec)
    else:
        assert compiler(spec) == expected
        assert compiler(spec) == expected


def test_raw_input_cache_copies_and_dtype_keys(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/"scripts/phase62_batch_worker.py"
    spec = importlib.util.spec_from_file_location("phase62_inputs_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = str((tmp_path/"raw.csv").resolve())
    pool = module.InputTables({source:"digest"})
    calls = []
    def read(path,dtypes):
        calls.append(1)
        return pd.DataFrame({"x":np.array([1,2],dtype=(dtypes or {}).get("x","int64"))})
    first = pool.read(read,source)
    first.loc[0,"x"] = 99
    second = pool.read(read,source)
    assert second.x.tolist() == [1,2] and len(calls)==1
    second.loc[1,"x"] = 88
    assert pool.read(read,source).x.tolist()==[1,2]
    assert pool.read(read,source,{"x":"float32"}).x.dtype==np.float32
    assert len(calls)==2
