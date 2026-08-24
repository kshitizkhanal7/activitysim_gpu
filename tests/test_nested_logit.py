import numpy as np
import pytest

pd = pytest.importorskip("pandas")
simulate = pytest.importorskip("activitysim.core.simulate")

from choiceforge.activitysim_destination import (
    _purpose_invariant_preprocessor,
    _single_preprocessor_settings,
)
from choiceforge.cuda_backend import cuda_available
from choiceforge.nested_logit import MTC21_ALTERNATIVES, mtc21_nested_logsums_cuda


NEST = {
    "name": "root", "coefficient": 1.0, "alternatives": [
        {"name": "AUTO", "coefficient": 0.72, "alternatives": [
            {"name": "DRIVEALONE", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[0:2])},
            {"name": "SHAREDRIDE2", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[2:4])},
            {"name": "SHAREDRIDE3", "coefficient": 0.35, "alternatives": list(MTC21_ALTERNATIVES[4:6])},
        ]},
        {"name": "NONMOTORIZED", "coefficient": 0.72, "alternatives": list(MTC21_ALTERNATIVES[6:8])},
        {"name": "TRANSIT", "coefficient": 0.72, "alternatives": [
            {"name": "WALKACCESS", "coefficient": 0.5, "alternatives": list(MTC21_ALTERNATIVES[8:13])},
            {"name": "DRIVEACCESS", "coefficient": 0.5, "alternatives": list(MTC21_ALTERNATIVES[13:18])},
        ]},
        {"name": "RIDEHAIL", "coefficient": 0.36, "alternatives": list(MTC21_ALTERNATIVES[18:21])},
    ],
}


def test_preprocessor_settings_accept_dict_and_reject_multiple():
    assert _single_preprocessor_settings({"preprocessor": {"SPEC": "x"}})["SPEC"] == "x"
    assert _single_preprocessor_settings({"preprocessor": [{"SPEC": "x"}, {"SPEC": "y"}]}) is None


def test_purpose_invariance_rejects_referenced_varying_coefficient():
    spec = pd.DataFrame({"expression": ["x = base + beta_cost"]})
    assert not _purpose_invariant_preprocessor(spec, [{"beta_cost": 1}, {"beta_cost": 2}])
    assert _purpose_invariant_preprocessor(spec, [{"unused": 1}, {"unused": 2}])


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_mtc21_cuda_matches_activitysim_nested_logsum():
    rng = np.random.default_rng(73)
    utilities = rng.normal(size=(257, 21))
    utilities[::7, 9:18] = -999.0
    frame = pd.DataFrame(utilities, columns=MTC21_ALTERNATIVES)
    expected = np.log(simulate.compute_nested_exp_utilities(frame, NEST)["root"].to_numpy())
    actual = mtc21_nested_logsums_cuda(utilities, NEST)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_mtc21_cuda_accepts_device_utilities_without_upload():
    from choiceforge.cuda_backend import _cupy

    utilities = np.random.default_rng(74).normal(size=(17, 21))
    actual, telemetry = mtc21_nested_logsums_cuda(
        _cupy().asarray(utilities), NEST, return_telemetry=True
    )
    expected = mtc21_nested_logsums_cuda(utilities, NEST)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert telemetry.host_to_device_ms == 0.0


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_mtc21_sharrow_float32_policy_matches_sharrow_reference_choices():
    from sharrow.nested_logit import (
        _utility_to_logsums_array,
        construct_nesting_tree,
    )

    rng = np.random.default_rng(75)
    utilities = rng.normal(size=(257, 21)).astype(np.float32)
    utilities[::7, 9:18] = np.float32(-999.0)
    tree = construct_nesting_tree(MTC21_ALTERNATIVES, NEST)
    edges_up, edges_dn, _, _ = tree.edge_slot_arrays()
    mu_params = np.asarray(
        [
            1.0 if tree.nodes[node].get("root") else tree.nodes[node].get("parameter", 1.0)
            for node in tree.standard_sort
        ],
        dtype=np.float32,
    )
    starts = np.zeros(len(tree.standard_sort), dtype=np.int32)
    lengths = np.zeros(len(tree.standard_sort), dtype=np.int32)
    for edge, parent in enumerate(edges_up):
        if lengths[parent] == 0:
            starts[parent] = edge
        lengths[parent] += 1
    expected = _utility_to_logsums_array(
        edges_up, edges_dn, mu_params, starts, lengths, utilities
    )
    actual = mtc21_nested_logsums_cuda(
        utilities, NEST, numeric_policy="sharrow_float32"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_mtc21_activitysim_pandas_policy_matches_legacy_nest_reducer():
    rng = np.random.default_rng(76)
    utilities = rng.normal(size=(257, 21)).astype(np.float32)
    utilities[::7, 9:18] = np.float32(-999.0)
    frame = pd.DataFrame(utilities, columns=MTC21_ALTERNATIVES)
    expected = np.log(
        simulate.compute_nested_exp_utilities(frame, NEST)["root"].to_numpy()
    )
    actual = mtc21_nested_logsums_cuda(
        utilities, NEST, numeric_policy="activitysim_pandas_float64"
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_mtc21_rejects_wrong_column_order():
    with pytest.raises(ValueError, match="canonical MTC order"):
        mtc21_nested_logsums_cuda(np.zeros((1, 21)), NEST, tuple(reversed(MTC21_ALTERNATIVES)))
