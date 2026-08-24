from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.gpu_native import GpuOnlyViolation
from choiceforge.tour_expansion import (
    POSSIBLE_TOURS_COUNT,
    mandatory_tours_cpu,
    mandatory_tours_gpu,
)


PIPELINE = Path(
    "benchmark-data/phase9-mtc-full/prototype_mtc_extended/"
    "o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline"
)


def _small_inputs():
    return {
        "person_id": np.asarray([10, 11, 12, 13, 14]),
        "household_id": np.asarray([1, 1, 2, 3, 4]),
        "mtf_choice": np.asarray([0, 1, 2, 3, 4]),
        "is_worker": np.asarray([1, 1, 0, 0, 0]),
        "workplace_zone_id": np.asarray([101, 102, 103, 104, 105]),
        "school_zone_id": np.asarray([201, 202, 203, 204, 205]),
        "home_zone_id": np.asarray([301, 302, 303, 304, 305]),
    }


def test_cpu_mandatory_tour_expansion_orders_types_and_swaps_nonworker():
    result = mandatory_tours_cpu(**_small_inputs())
    assert len(result["tour_id"]) == 8
    # Final person chooses work_and_school but is not a worker: physical row
    # order stays work then school while schedule order is school then work.
    np.testing.assert_array_equal(result["tour_type"][-2:], [0, 1])
    np.testing.assert_array_equal(result["tour_num"][-2:], [2, 1])
    np.testing.assert_array_equal(result["destination"][-2:], [105, 205])
    np.testing.assert_array_equal(
        result["tour_id"][-2:], [14 * POSSIBLE_TOURS_COUNT + 39, 14 * POSSIBLE_TOURS_COUNT + 31]
    )


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_gpu_mandatory_tour_expansion_is_exact_and_fail_closed():
    cp = _cupy()
    inputs = _small_inputs()
    expected = mandatory_tours_cpu(**inputs)
    actual = mandatory_tours_gpu(**{name: cp.asarray(value) for name, value in inputs.items()})
    for name, values in expected.items():
        np.testing.assert_array_equal(cp.asnumpy(actual.columns[name]), values)
    with pytest.raises(GpuOnlyViolation, match="reside on the GPU"):
        mandatory_tours_gpu(**inputs)


@pytest.mark.skipif(not PIPELINE.exists(), reason="public MTC checkpoint not staged")
def test_public_checkpoint_implies_the_reviewed_canonical_id_constants():
    tours = pd.read_parquet(PIPELINE / "tours" / "mandatory_tour_frequency.parquet")
    offsets = tours.index.to_numpy() - tours.person_id.to_numpy() * POSSIBLE_TOURS_COUNT
    expected = np.where(
        tours.tour_type.astype(str).to_numpy() == "work",
        38 + tours.tour_type_num.to_numpy(),
        30 + tours.tour_type_num.to_numpy(),
    )
    np.testing.assert_array_equal(offsets, expected)
