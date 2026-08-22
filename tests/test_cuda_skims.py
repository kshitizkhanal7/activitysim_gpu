import numpy as np
import pytest

from choiceforge.cuda_skims import (
    CudaChooserColumns, CudaDatasetWrapper, CudaSkimDictionary, CudaSkimWrapper,
    activitysim_cuda_environment,
)


class _Mapper:
    def map(self, zones):
        mapping = {10: 0, 20: 1, 30: 2, -1: -1}
        return np.asarray([mapping.get(int(zone), -99) for zone in zones])


def _skims():
    # block, origin, destination: values make the selected element obvious.
    data = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
    return CudaSkimDictionary(
        data, {"DIST": 0, "SOV_TIME": 1}, {"TRANSIT": {"AM": 2, "PM": 3}}, _Mapper(), (3, 3)
    ), data


def test_cuda_2d_skim_matches_cpu_fancy_indexing_and_missing_zone():
    pytest.importorskip("cupy")
    skims, data = _skims()
    actual = skims.lookup([10, 30, -1], [20, 10, 20], "SOV_TIME", return_device=False)
    expected = np.array([data[1, 0, 1], data[1, 2, 0], np.nan])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_cuda_3d_skim_matches_period_specific_blocks():
    pytest.importorskip("cupy")
    skims, data = _skims()
    actual = skims.lookup_3d([10, 20], [30, 10], ["PM", "AM"], "TRANSIT", return_device=False)
    np.testing.assert_allclose(actual, [data[3, 0, 2], data[2, 1, 0]])


def test_cuda_skims_reject_invalid_zone_and_unknown_period():
    pytest.importorskip("cupy")
    skims, _ = _skims()
    with pytest.raises(AssertionError, match="OD pairs"):
        skims.lookup([999], [10], "DIST")
    with pytest.raises(KeyError, match="unknown"):
        skims.lookup_3d([10], [20], ["XX"], "TRANSIT")


def test_activitysim_style_wrapper_is_lazy_and_uses_real_column_names():
    pd = pytest.importorskip("pandas")
    cp = pytest.importorskip("cupy")
    skims, data = _skims()
    choosers = pd.DataFrame({"orig": [10, 20], "dest": [30, 10], "period": ["PM", "AM"], "age": [16, 17]})
    wrapper = CudaSkimWrapper(skims, choosers, "orig", "dest", "period")
    env = activitysim_cuda_environment(choosers, {"walkSpeed": 3.0}, odt_skims=wrapper)
    np.testing.assert_allclose(cp.asnumpy(env["odt_skims"]["TRANSIT"]), [data[3, 0, 2], data[2, 1, 0]])
    np.testing.assert_array_equal(cp.asnumpy(env["df"]["age"]), [16, 17])
    assert env["walkSpeed"] == 3.0


def test_dataset_wrapper_uses_sharrow_precomputed_positions():
    xr = pytest.importorskip("xarray")
    cp = pytest.importorskip("cupy")
    pd = pytest.importorskip("pandas")
    cube = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    dataset = xr.Dataset({"SOV_TIME": (("time_period", "otaz", "dtaz"), cube)})
    wrapper = type("DatasetWrapper", (), {})()
    wrapper.dataset, wrapper.df = dataset, pd.DataFrame({"x": [1, 2]})
    wrapper.odim, wrapper.ddim = "otaz", "dtaz"
    wrapper.positions = pd.DataFrame({"otaz": [0, 2], "dtaz": [2, 1], "time_period": [1, 0]})
    actual = CudaDatasetWrapper(wrapper)["SOV_TIME"]
    np.testing.assert_allclose(cp.asnumpy(actual), [cube[1, 0, 2], cube[0, 2, 1]])
