import numpy as np
import pytest

from choiceforge.destination_utility import LoweredDestinationUtility
from choiceforge.destination_utility import mtc21_logsums_from_lowered_cuda
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from test_nested_logit import NEST


def _model():
    return LoweredDestinationUtility(
        feature_names=("income", "walk_time", "toll"),
        alternative_names=("WALK", "DRIVE", "TNC"),
        coefficients=np.array([[0.001, 0.002, 0.0015], [-0.30, -0.04, -0.08], [0.0, -1.2, -0.9]]),
        constants=np.array([0.2, -0.4, -0.1]),
    )


def test_cpu_reference_matches_declared_linear_utility():
    features = np.array([[50000.0, 8.0, 0.0], [90000.0, 20.0, 4.5]])
    actual = _model().cpu_reference(features)
    expected = features @ _model().coefficients + _model().constants
    np.testing.assert_array_equal(actual, expected)


def test_cuda_lowered_utility_matches_cpu_and_can_stay_on_device():
    pytest.importorskip("cupy")
    model = _model()
    rng = np.random.default_rng(42)
    features = rng.normal(size=(4096, 3))
    actual, telemetry = model.cuda(features, return_telemetry=True)
    np.testing.assert_allclose(actual, model.cpu_reference(features), rtol=1e-12, atol=1e-12)
    device, device_telemetry = model.cuda(features, return_device=True, return_telemetry=True)
    assert hasattr(device, "__cuda_array_interface__")
    assert telemetry.host_to_device_ms >= 0
    assert device_telemetry.device_to_host_ms == 0


def test_ordered_float32_cuda_utility_matches_ordered_cpu_reference():
    pytest.importorskip("cupy")
    model = LoweredDestinationUtility(
        ("x", "y", "z"), ("A", "B"),
        np.array([[1.1, -0.3], [0.2, 2.0], [-3.0, 0.7]]), np.array([0.1, -0.2]),
        compute_dtype="float32",
    )
    features = np.array([[1.3, -2.2, 5.5], [4.0, 0.25, -1.0]], dtype=np.float32)
    actual = model.cuda(features, ordered=True)
    np.testing.assert_array_equal(actual, model.cpu_reference(features, ordered=True))


def test_rejects_ambiguous_or_malformed_abi():
    with pytest.raises(ValueError, match="unique"):
        LoweredDestinationUtility(("x", "x"), ("A",), np.ones((2, 1)), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        LoweredDestinationUtility(("x",), ("A",), np.ones((2, 1)), np.zeros(1))
    with pytest.raises(ValueError, match="row-by-3"):
        _model().cpu_reference(np.ones((2, 2)))


def test_lowered_mtc21_pipeline_keeps_utilities_on_device():
    pytest.importorskip("cupy")
    rng = np.random.default_rng(19)
    model = LoweredDestinationUtility(
        feature_names=("x", "y"),
        alternative_names=MTC21_ALTERNATIVES,
        coefficients=rng.normal(size=(2, 21)),
        constants=rng.normal(size=21),
    )
    features = rng.normal(size=(256, 2))
    actual, telemetry = mtc21_logsums_from_lowered_cuda(
        model, features, NEST, return_telemetry=True
    )
    from choiceforge.nested_logit import mtc21_nested_logsums_cuda

    expected = mtc21_nested_logsums_cuda(model.cpu_reference(features), NEST)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert telemetry.utility.device_to_host_ms == 0
    assert telemetry.nested_logsum.host_to_device_ms == 0
