from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from choiceforge.calibrated_chain import (
    choice_from_probabilities_cpu,
    evaluate_mnl_features,
    gather_by_key_gpu,
    key_rows_gpu,
    mnl_probabilities,
    mnl_utilities,
    resolve_activitysim_mnl_spec,
)
from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.gpu_native import (
    GpuOnlyViolation,
    activitysim_uniforms_cpu,
    activitysim_uniforms_gpu,
)


CONFIG = Path(
    "benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs"
)


@pytest.mark.skipif(not CONFIG.exists(), reason="public MTC benchmark not staged")
def test_public_calibrated_specs_resolve_with_activitysim_float32_semantics():
    auto = resolve_activitysim_mnl_spec(
        "auto",
        CONFIG / "auto_ownership.csv",
        CONFIG / "auto_ownership_coefficients.csv",
    )
    mtf = resolve_activitysim_mnl_spec(
        "mtf",
        CONFIG / "mandatory_tour_frequency.csv",
        CONFIG / "mandatory_tour_frequency_coefficients.csv",
    )
    assert (len(auto.expressions), auto.coefficients.shape, auto.alternatives) == (
        29,
        (29, 5),
        ("cars0", "cars1", "cars2", "cars3", "cars4"),
    )
    assert (len(mtf.expressions), mtf.coefficients.shape) == (98, (98, 5))
    # Resolution quantizes exactly as ActivitySim's Series.astype(float32).
    assert np.array_equal(
        auto.coefficients.astype(np.float32).astype(np.float64), auto.coefficients
    )


def test_calibrated_mnl_cpu_boundary_is_ordered_and_normalized():
    features = np.asarray([[1.0, 2.0], [1.0, -1.0]])
    coefficients = np.asarray([[0.1, -0.2], [0.5, 0.25]])
    utilities = mnl_utilities(features, coefficients)
    probabilities, logsums = mnl_probabilities(utilities)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(logsums, np.log(np.exp(utilities).sum(axis=1)))
    choices = choice_from_probabilities_cpu(probabilities, [0.0, 0.999999])
    np.testing.assert_array_equal(choices, [0, 1])


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_activitysim_mt19937_gpu_draws_are_bit_exact_and_partition_stable():
    cp = _cupy()
    ids = np.asarray([1, 2, 99, 2**31, 2**32 + 4], dtype=np.int64)
    expected = activitysim_uniforms_cpu(
        ids, "households", "auto_ownership_simulate"
    )
    whole = cp.asnumpy(
        activitysim_uniforms_gpu(
            cp.asarray(ids), "households", "auto_ownership_simulate"
        )
    )
    pieces = np.concatenate(
        [
            cp.asnumpy(
                activitysim_uniforms_gpu(
                    cp.asarray(ids[:2]), "households", "auto_ownership_simulate"
                )
            ),
            cp.asnumpy(
                activitysim_uniforms_gpu(
                    cp.asarray(ids[2:]), "households", "auto_ownership_simulate"
                )
            ),
        ]
    )
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(pieces, expected)
    for offset in (1, 7, 311, 312, 313):
        expected_offset = activitysim_uniforms_cpu(
            ids, "households", "auto_ownership_simulate", offset=offset
        )
        actual_offset = cp.asnumpy(
            activitysim_uniforms_gpu(
                cp.asarray(ids),
                "households",
                "auto_ownership_simulate",
                offset=offset,
            )
        )
        np.testing.assert_array_equal(actual_offset, expected_offset)


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_gpu_key_gather_matches_many_to_one_join_and_rejects_missing_keys():
    cp = _cupy()
    source = cp.asarray([30, 10, 20], dtype=cp.int64)
    targets = cp.asarray([10, 30, 10, 20], dtype=cp.int64)
    gathered = gather_by_key_gpu(
        source,
        targets,
        {"value": cp.asarray([3.0, 1.0, 2.0], dtype=cp.float64)},
    )
    np.testing.assert_array_equal(cp.asnumpy(gathered["value"]), [1.0, 3.0, 1.0, 2.0])
    with pytest.raises(KeyError, match="missing"):
        gather_by_key_gpu(source, cp.asarray([40]), {"value": cp.asarray([3, 1, 2])})


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_gpu_key_row_map_is_reusable_and_rejects_ambiguous_sources():
    cp = _cupy()
    source = cp.asarray([30, 10, 20], dtype=cp.int64)
    target = cp.asarray([10, 30, 10, 20], dtype=cp.int64)
    rows = key_rows_gpu(source, target)
    values = cp.asarray([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(cp.asnumpy(values[rows]), [1.0, 3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="unique"):
        key_rows_gpu(cp.asarray([10, 10]), cp.asarray([10]))
    with pytest.raises(KeyError, match="missing"):
        key_rows_gpu(source, cp.asarray([40]))
