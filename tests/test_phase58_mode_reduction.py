from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from activitysim.core import simulate, logit
from choiceforge.cuda_backend import _cupy
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from test_nested_logit import NEST


@pytest.mark.parametrize("seed", [58, 583, 584])
def test_mode_reduction_matches_upstream_and_guards_cdf_edges(seed):
    cp = _cupy()
    random = np.random.default_rng(seed)
    values = random.normal(-2, 4, (513, 21)).astype(np.float32)
    values[::7, 8:18] = -999
    values[::11, :6] = -999
    raw = pd.DataFrame(values, columns=MTC21_ALTERNATIVES)
    state = SimpleNamespace(settings=SimpleNamespace(skip_failed_choices=False))
    nested = simulate.compute_nested_exp_utilities(raw, NEST)
    expected_logsums = np.log(nested.root.to_numpy())
    conditional = simulate.compute_nested_probabilities(state, nested, NEST, "test")
    expected_probs = simulate.compute_base_probabilities(conditional, NEST, raw).to_numpy()
    draws = random.random((len(raw), 1))
    # Include deliberately ambiguous boundaries, not just random easy choices.
    draws[:20, 0] = expected_probs[:20, :7].sum(axis=1)
    choices, logsums, guards, probs = reduce_modes(cp.asarray(values), cp.asarray(draws), NEST, capture=True)
    expected_choices = logit.choice_maker(expected_probs, draws)
    guards = cp.asnumpy(guards).astype(bool)
    assert guards[:20].all()
    np.testing.assert_array_equal(cp.asnumpy(choices)[~guards], expected_choices[~guards])
    np.testing.assert_allclose(cp.asnumpy(probs), expected_probs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(logsums), expected_logsums, atol=1e-12, rtol=1e-12)


def test_reordered_nesting_and_bad_coefficients_fail_closed():
    nest = deepcopy(NEST)
    nest["alternatives"] = nest["alternatives"][::-1]
    with pytest.raises(ValueError, match="topology"):
        validate(nest, MTC21_ALTERNATIVES)
    nest = deepcopy(NEST)
    nest["alternatives"][0]["coefficient"] = .001
    with pytest.raises(ValueError, match="coefficients"):
        validate(nest, MTC21_ALTERNATIVES)


@pytest.mark.parametrize("capture", [False, True])
def test_live_cpu_boundary_adjudication_uses_same_draw_once(monkeypatch, tmp_path, capture):
    from choiceforge import activitysim_mode_choice, sharrow_cuda
    from choiceforge.phase58_mode_reduction import mode_choice_simulate
    cp = _cupy()
    values = np.random.default_rng(585).normal(-2, 3, (19,21)).astype(np.float32)
    frame = pd.DataFrame(index=pd.Index(np.arange(19)*31+7, name="trip_id"))
    spec = pd.DataFrame(np.ones((1,21)), columns=MTC21_ALTERNATIVES)
    state = SimpleNamespace(settings=SimpleNamespace(trace_hh_id=None, use_explicit_error_terms=False,
                                                     skip_failed_choices=False))
    raw = pd.DataFrame(values, index=frame.index, columns=spec.columns)
    nested = simulate.compute_nested_exp_utilities(raw, NEST)
    probs = simulate.compute_base_probabilities(
        simulate.compute_nested_probabilities(state,nested,NEST,"test"), NEST, spec).to_numpy()
    draws = probs[:,:7].sum(axis=1).reshape(-1,1)
    calls = []
    def uniform(*a, **k):
        calls.append(k)
        return cp.asarray(draws)
    runtime = SimpleNamespace(uniform=uniform, mode_events=[])
    if capture:
        runtime.mode_capture_directory = tmp_path
    telemetry = SimpleNamespace(**dict.fromkeys(("terms", "alternatives", "expression_dtype",
        "persistent_plan", "plan_cache_hit", "plan_build_ms", "binding_resolve_ms", "host_pack_ms",
        "input_upload_ms", "kernel_ms", "cache_key", "source_sha256"), 0))
    monkeypatch.setattr(activitysim_mode_choice,"_strict_inputs", lambda *a:({}, {}, False, 0))
    monkeypatch.setattr(activitysim_mode_choice,"_write_report", lambda *a:None)
    monkeypatch.setattr(sharrow_cuda,"evaluate_strict_cuda", lambda *a, **k:
        SimpleNamespace(utilities=cp.asarray(values), telemetry=telemetry))
    from activitysim.core.configuration.logit import LogitNestSpec
    live_nest = LogitNestSpec.model_validate(NEST) if capture else NEST
    result = mode_choice_simulate(runtime, state, frame, spec, live_nest, None, {},
                                  "trip_mode", "logsum", "test", "trip_mode")
    np.testing.assert_array_equal(result.trip_mode.cat.codes.to_numpy()-1, logit.choice_maker(probs,draws))
    np.testing.assert_array_equal(result.logsum.to_numpy(), np.log(nested.root.to_numpy()))
    assert len(calls) == 1 and calls[0]["device_only"]
    assert runtime.mode_events[0]["guard_rows"] == len(frame)
    if capture:
        with np.load(tmp_path/"batch-00.npz", allow_pickle=False) as data:
            assert set(data.files) == {"utilities", "draws", "chooser_ids", "nest_json", "trace_label"}
            np.testing.assert_array_equal(data["utilities"], values)
            np.testing.assert_array_equal(data["draws"], draws)
