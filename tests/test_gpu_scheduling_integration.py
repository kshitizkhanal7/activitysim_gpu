import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.gpu_native import GpuOnlyViolation
from choiceforge.gpu_scheduling_integration import assemble_device_logsum_cache


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_device_logsum_cache_scatter_is_exact_and_stays_on_gpu():
    cp = _cupy()
    expected_ids = np.array([11, 22], dtype=np.int64)
    metadata = {
        "chooser_ids": np.array([11, 11, 11, 22, 22], dtype=np.int64),
        "start": np.array([5, 6, 10, 5, 19], dtype=np.int16),
        "end": np.array([5, 9, 14, 6, 23], dtype=np.int16),
    }
    values = np.array([1.25, 2.5, 3.75, -1.0, 9.0], dtype=np.float64)
    result = assemble_device_logsum_cache(cp.asarray(values), metadata, expected_ids)
    assert hasattr(result.cache, "__cuda_array_interface__")
    cache = cp.asnumpy(result.cache)
    present = cp.asnumpy(result.present)
    assert cache[0, 0].view(np.uint32) == np.float32(1.25).view(np.uint32)
    assert cache[0, 6].view(np.uint32) == np.float32(2.5).view(np.uint32)
    assert cache[0, 12].view(np.uint32) == np.float32(3.75).view(np.uint32)
    assert cache[1, 1].view(np.uint32) == np.float32(-1.0).view(np.uint32)
    assert cache[1, 24].view(np.uint32) == np.float32(9.0).view(np.uint32)
    assert int(present.sum()) == 5


@pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")
def test_device_logsum_cache_fails_closed_on_identity_and_slot_errors():
    cp = _cupy()
    metadata = {
        "chooser_ids": np.array([11, 11], dtype=np.int64),
        "start": np.array([5, 5], dtype=np.int16),
        "end": np.array([5, 5], dtype=np.int16),
    }
    repeated = assemble_device_logsum_cache(
        cp.asarray([1.0, 1.0]), metadata, [11]
    )
    assert int(cp.asnumpy(repeated.present).sum()) == 1
    with pytest.raises(ValueError, match="differ within a duplicate"):
        assemble_device_logsum_cache(cp.asarray([1.0, 2.0]), metadata, [11])
    with pytest.raises(ValueError, match="chooser order"):
        assemble_device_logsum_cache(
            cp.asarray([1.0]),
            {"chooser_ids": [11], "start": [5], "end": [5]},
            [12],
        )
    with pytest.raises(GpuOnlyViolation):
        assemble_device_logsum_cache(
            np.array([1.0]),
            {"chooser_ids": [11], "start": [5], "end": [5]},
            [11],
        )
