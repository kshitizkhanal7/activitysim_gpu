from __future__ import annotations

import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.gpu_native import (
    DeviceTable,
    GpuMemoryBudget,
    GpuNativeRuntime,
    GpuOnlyViolation,
    entity_uniforms_cpu,
    entity_uniforms_gpu,
    plan_household_partitions,
    segmented_sum_sorted_gpu,
)


def test_memory_budget_and_capacity_are_explicit():
    gib = 1024**3
    budget = GpuMemoryBudget(16 * gib, 2 * gib, 4 * gib, 2 * gib, 3 * gib)
    assert budget.committed_bytes == 11 * gib
    assert budget.unallocated_bytes == 5 * gib
    assert budget.max_entities(10_000, utilization=0.8) == int(4 * gib // 10_000)
    with pytest.raises(ValueError, match="overcommits"):
        GpuMemoryBudget(10, 3, 3, 3, 3)


def test_partition_plan_is_complete_and_deterministic():
    assert plan_household_partitions(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert plan_household_partitions(0, 4) == []
    with pytest.raises(ValueError):
        plan_household_partitions(1, 0)


def test_device_table_rejects_host_columns():
    with pytest.raises(GpuOnlyViolation, match="not a CUDA array"):
        DeviceTable({"host": np.arange(4)})


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_counter_rng_matches_oracle_and_is_partition_invariant():
    cp = _cupy()
    ids = np.array([1, 2, 19, 2**31 + 7, 9_999_999], dtype=np.int64)
    expected = entity_uniforms_cpu(ids, seed=20260824, stream=17)
    whole = cp.asnumpy(entity_uniforms_gpu(cp.asarray(ids), seed=20260824, stream=17))
    pieces = np.concatenate(
        [
            cp.asnumpy(entity_uniforms_gpu(cp.asarray(ids[:2]), seed=20260824, stream=17)),
            cp.asnumpy(entity_uniforms_gpu(cp.asarray(ids[2:]), seed=20260824, stream=17)),
        ]
    )
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(pieces, expected)
    assert np.all((whole > 0.0) & (whole < 1.0))


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_sorted_segment_sum_is_fixed_order_and_device_resident():
    cp = _cupy()
    groups = cp.asarray([1, 1, 2, 2, 2, 7], dtype=cp.int64)
    values = cp.asarray([0.25, 0.5, 1.0, 2.0, 3.0, -1.0], dtype=cp.float32)
    result = segmented_sum_sorted_gpu(groups, values)
    starts = cp.asnumpy(result.columns["is_start"]).astype(bool)
    sums = cp.asnumpy(result.columns["sum"])
    np.testing.assert_array_equal(starts, [True, False, True, False, False, True])
    np.testing.assert_array_equal(sums[starts], [0.75, 6.0, -1.0])
    with pytest.raises(ValueError, match="sorted"):
        segmented_sum_sorted_gpu(cp.asarray([2, 1]), cp.asarray([1.0, 1.0]))


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_empty_gpu_partition_is_valid():
    cp = _cupy()
    draws = entity_uniforms_gpu(cp.asarray([], dtype=cp.int64), seed=1)
    totals = segmented_sum_sorted_gpu(
        cp.asarray([], dtype=cp.int64), cp.asarray([], dtype=cp.float32)
    )
    assert draws.shape == (0,)
    assert totals.nrows == 0


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_runtime_seals_host_boundary_and_counts_no_fallback():
    runtime = GpuNativeRuntime()
    table = runtime.ingress_table("people", {"id": np.arange(8, dtype=np.int64)})
    runtime.seal_ingress()
    assert table.nrows == 8
    with pytest.raises(GpuOnlyViolation, match="ingress is closed"):
        runtime.ingress_table("late", {"id": np.arange(2)})
    with pytest.raises(GpuOnlyViolation, match="CPU fallback"):
        runtime.cpu_fallback("mode_choice")


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_two_stage_choices_are_exact_across_partitioning():
    cp = _cupy()
    n = 4097
    ids = np.arange(1, n + 1, dtype=np.int64)
    x = np.column_stack(
        [
            np.ones(n),
            (ids % 9) / 8.0,
            np.log1p(ids % 1000) / np.log(1001),
            (ids % 2).astype(np.float32),
        ]
    ).astype(np.float32)
    beta = np.linspace(-0.5, 0.7, 21 * 4, dtype=np.float32).reshape(21, 4)
    constants = np.linspace(0.1, -0.2, 21, dtype=np.float32)

    def run_piece(begin: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        runtime = GpuNativeRuntime()
        input_table = runtime.ingress_table(
            "choosers", {"id": ids[begin:end], "features": x[begin:end]}
        )
        params = runtime.ingress_table(
            "parameters", {"beta": beta, "constants": constants[:, None]}
        )
        runtime.seal_ingress()
        draws = entity_uniforms_gpu(input_table.columns["id"], 77, 2)
        first = runtime.linear_choice(
            input_table.columns["features"],
            params.columns["beta"],
            params.columns["constants"].reshape(-1),
            draws,
        )
        second_x = cp.ascontiguousarray(input_table.columns["features"].copy())
        second_x[:, 3] = first.columns["choice"].astype(cp.float32) / np.float32(20.0)
        second_draws = entity_uniforms_gpu(input_table.columns["id"], 77, 3)
        second = runtime.linear_choice(
            second_x,
            params.columns["beta"],
            params.columns["constants"].reshape(-1),
            second_draws,
        )
        runtime.assert_gpu_only()
        output = runtime.egress_table(second)
        return output["choice"], output["logsum"]

    whole_choice, whole_logsum = run_piece(0, n)
    ranges = plan_household_partitions(n, 1000)
    part_choice, part_logsum = zip(*(run_piece(a, b) for a, b in ranges))
    np.testing.assert_array_equal(np.concatenate(part_choice), whole_choice)
    np.testing.assert_array_equal(np.concatenate(part_logsum), whole_logsum)
