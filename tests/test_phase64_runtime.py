import builtins
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from choiceforge.phase64_runtime import ExpressionCode, Runtime


@pytest.mark.parametrize("source", ["0", "-0.0", "x * 1.1 + 2", "  \tx + 1", "[x * i for i in range(3)]", "float('nan')"])
def test_expression_semantics(source):
    cache = ExpressionCode()
    for x in (2., -7.):
        # Comprehensions resolve free names in globals, matching builtin eval.
        a, b = {"x":x}, {"x":x}
        expected = builtins.eval(source, a, a)
        actual = cache.evaluate(source, b, b)
        np.testing.assert_equal(actual, expected)
    assert cache.hits == 1 and cache.misses == 1


def test_live_inputs_and_side_effects_not_cached():
    cache = ExpressionCode()
    values = []
    for i in range(4):
        cache.evaluate("values.append(x)", {}, {"values":values,"x":i})
    assert values == [0,1,2,3]
    assert cache.hits == 3


@pytest.mark.parametrize("source,error", [("1/0",ZeroDivisionError),("missing",NameError),("1 +",SyntaxError)])
def test_errors_propagate(source,error):
    with pytest.raises(error):
        ExpressionCode().evaluate(source, {}, {})


def test_capacity_and_non_string_fallback():
    cache = ExpressionCode(2)
    for i in range(5):
        assert cache.evaluate(str(i), {}, {}) == i
        assert len(cache.codes) <= 2
    assert cache.evaluate(compile("x+1","test","eval"),{},dict(x=3)) == 4
    assert cache.fallbacks == 1


def test_foreign_thread_falls_back():
    cache = ExpressionCode()
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(cache.evaluate,"x+1",{},dict(x=2)).result() == 3
    assert not cache.codes and cache.fallbacks == 1


def test_scope_restores_after_exception_and_rejects_overlap():
    from activitysim.core import simulate, assign
    before = [vars(m).get("eval") for m in (simulate,assign)]
    runtime = Runtime()
    with pytest.raises(RuntimeError):
        with runtime.for_step(None,"test"):
            assert simulate.eval("x",{},dict(x=1)) == 1
            with pytest.raises(ValueError):
                with runtime.for_step(None,"overlap"):
                    pass
            raise RuntimeError("test")
    assert [vars(m).get("eval") for m in (simulate,assign)] == before
    assert not runtime.active


@pytest.mark.parametrize("sharrow", [False,True])
def test_upstream_coefficients_bit_exact_and_live(sharrow):
    from activitysim.core import simulate
    state = SimpleNamespace(settings=SimpleNamespace(sharrow=sharrow))
    spec = pd.DataFrame({"a":["x","-0.0","0","x * 1.5"],"b":["2*x","0","0","1.25"]})
    runtime = Runtime()
    for value in (1.2,7.5):
        expected = simulate.eval_coefficients(state,spec,dict(x=value),None)
        with runtime.for_step(state,"coefficients"):
            actual = simulate.eval_coefficients(state,spec,dict(x=value),None)
        pd.testing.assert_frame_equal(expected,actual)
        np.testing.assert_array_equal(expected.to_numpy().view(np.uint32),actual.to_numpy().view(np.uint32))


def test_cpu_mode_ablation_matches_gpu_and_restores_threads():
    import numba
    from test_nested_logit import NEST
    from choiceforge.cuda_backend import _cupy
    from choiceforge.phase58_mode_reduction import reduce_modes
    cp=_cupy()
    random=np.random.default_rng(64)
    values=cp.asarray(random.normal(-2,3,(1001,21)).astype(np.float32))
    draws=cp.asarray(random.random(1001))
    before=numba.get_num_threads()
    gpu=reduce_modes(values,draws,NEST)
    runtime=Runtime("mode_cpu")
    cpu=runtime.cpu_reduce(values,draws,NEST)
    guarded=cp.asnumpy((gpu[2]!=0)|(cpu[2]!=0))
    np.testing.assert_array_equal(cp.asnumpy(gpu[0])[~guarded],cp.asnumpy(cpu[0])[~guarded])
    np.testing.assert_allclose(cp.asnumpy(gpu[1]),cp.asnumpy(cpu[1]),atol=1e-12,rtol=1e-12)
    assert before==numba.get_num_threads()
    assert runtime.mode_events[0]["device_to_host_bytes"]>0
    assert runtime.mode_events[0]["download_seconds"]>=0
    with pytest.raises(ValueError):
        runtime.cpu_reduce(values,cp.full(1001,np.nan),NEST)


def test_cpu_ablation_scope_restores_transfer_marker():
    from choiceforge import phase58_mode_reduction as module
    runtime=Runtime("mode_cpu")
    original=module.reduce_modes
    with runtime.for_step(None,"trip_mode_choice"):
        assert module._cpu_reducer_control is runtime
        assert module.reduce_modes==runtime.cpu_reduce
    assert module.reduce_modes is original
    assert "_cpu_reducer_control" not in vars(module)
