from __future__ import annotations

import h5py
import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.resident_skim_cache import (
    MTC_PERIODS,
    ResidentOmxSkimCache,
    logical_skims_from_ir,
)


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")


def _ir():
    return {
        "terms": [
            {"tree": {"op": "skim", "direction": "od_skims", "key": {"op": "const", "value": "DIST"}}},
            {"tree": {"op": "skim", "direction": "od_skims_reverse", "key": {"op": "const", "value": "DIST"}}},
            {"tree": {"op": "skim", "direction": "odt_skims", "key": {"op": "const", "value": "TIME"}}},
            {"tree": {"op": "skim", "direction": "dot_skims", "key": {"op": "const", "value": "TIME"}}},
            {"tree": {"op": "skim", "direction": "odr_skims", "key": {"op": "const", "value": "TIME"}}},
            {"tree": {"op": "skim", "direction": "dor_skims", "key": {"op": "const", "value": "TIME"}}},
        ]
    }


def _omx(path):
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        base = np.arange(16, dtype=np.float64).reshape(4, 4)
        data["DIST"] = base + 0.25
        for number, period in enumerate(MTC_PERIODS):
            data[f"TIME__{period}"] = base + 100 * number + 0.5


def test_ir_hot_set_deduplicates_directional_physical_cube():
    logical = logical_skims_from_ir(_ir())
    assert len(logical) == 6
    assert len({item.physical_key for item in logical}) == 2


def test_resident_omx_cache_is_budgeted_and_bit_exact(tmp_path):
    path = tmp_path / "tiny.omx"
    _omx(path)
    required = 4 * 4 * 4 * 6
    with pytest.raises(MemoryError, match="exceeds budget"):
        ResidentOmxSkimCache.load(path, _ir(), budget_bytes=required - 1)

    cache = ResidentOmxSkimCache.load(path, _ir(), budget_bytes=required)
    assert cache.telemetry.logical_bindings == 6
    assert cache.telemetry.physical_cubes == 2
    assert cache.telemetry.resident_float32_bytes == required

    origin = np.asarray([0, 1, 2, 3], dtype=np.int64)
    destination = np.asarray([3, 2, 1, 0], dtype=np.int64)
    outbound = np.asarray([0, 1, 2, 3], dtype=np.int32)
    inbound = np.asarray([4, 3, 2, 1], dtype=np.int32)
    expected = cache.probe_cpu(origin, destination, outbound, inbound)
    cp = _cupy()
    actual = cache.probe_gpu(
        cp.asarray(origin), cp.asarray(destination),
        cp.asarray(outbound), cp.asarray(inbound),
    )
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(actual[0]), expected[0])
    np.testing.assert_array_equal(cp.asnumpy(actual[1]), expected[1])
