import numba as nb
import numpy as np
import pytest

from choiceforge.arithmetic_abi import (
    NUMPY_FLOAT32_CHOICE_ABI_SHA256,
    NUMPY_FLOAT32_CHOICE_ABI_VERSION,
    SHARROW15_ABI_SHA256,
    SHARROW15_ABI_VERSION,
    sharrow15_cpu_reduce,
    sharrow15_cuda_reduction,
    numpy_float32_choice_cuda_helpers,
    Float32ProbabilityPolicy,
    Float32ReductionPolicy,
    NumericPolicyCompiler,
    grouped_left_reduction,
    ordered_left_reduction,
    reduce_float32,
)


@nb.njit(cache=True, fastmath=False)
def _openblas_reference(features, coefficients):
    output = np.empty(1, dtype=np.float32)
    np.dot(features, coefficients.reshape(15, 1), out=output)
    return output[0]


def test_sharrow15_shared_schedule_is_bit_exact_to_openblas_sgemv():
    rng = np.random.default_rng(410031)
    coefficients = rng.normal(0.0, 8.0, size=15).astype(np.float32)
    features = rng.normal(0.0, 40.0, size=(10_000, 15)).astype(np.float32)
    expected = np.asarray(
        [_openblas_reference(row, coefficients) for row in features],
        dtype=np.float32,
    )
    actual = np.asarray(
        [sharrow15_cpu_reduce(row, coefficients) for row in features],
        dtype=np.float32,
    )
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_sharrow15_cuda_source_and_abi_are_stable():
    source = sharrow15_cuda_reduction()
    assert SHARROW15_ABI_VERSION == "sharrow15-openblas-sgemv-group4-left-v1"
    assert SHARROW15_ABI_SHA256 == (
        "b73ea10db5583c9d684fae16a67976d10580d284e602103b456f3a31174e84f4"
    )
    assert source.index("abi_product_0") < source.index("abi_product_3")
    assert source.index("utility += abi_group_0") < source.index("abi_product_4")
    assert "abi_group_5 += abi_product_14" in source


def test_numpy_float32_choice_codegen_is_versioned_and_fixed_shape():
    source = numpy_float32_choice_cuda_helpers()
    assert NUMPY_FLOAT32_CHOICE_ABI_VERSION == "numpy246-avx2-exp-pairwise128-v1"
    assert NUMPY_FLOAT32_CHOICE_ABI_SHA256 == (
        "08521462b486c72da65445b1e8e720ef15cd0929be7dacd4c9c4cabba0d0da63"
    )
    assert source.count("const float leaf_") == 16
    assert "numpy_pairwise_exp_sum_1454" in source
    assert "fmaf(quadrant, -6.93145752e-1f, value)" in source


@pytest.mark.parametrize("term_count", [1, 3, 4, 5, 15, 16, 31])
def test_general_reduction_compiler_matches_its_declared_schedule(term_count):
    rng = np.random.default_rng(420000 + term_count)
    features = rng.normal(size=(37, term_count)).astype(np.float32)
    coefficients = rng.normal(size=term_count).astype(np.float32)
    policy = grouped_left_reduction(term_count)
    actual = reduce_float32(features, coefficients, policy)
    expected = []
    for row in features:
        total = np.float32(0)
        for group in policy.groups:
            partial = np.float32(0)
            for position in group:
                partial = np.float32(
                    partial + np.float32(row[position] * coefficients[position])
                )
            total = np.float32(total + partial)
        expected.append(total)
    assert np.array_equal(actual.view(np.uint32), np.asarray(expected).view(np.uint32))


def test_policy_hash_changes_for_semantic_mutations():
    grouped = grouped_left_reduction(15)
    ordered = ordered_left_reduction(15)
    assert grouped.sha256 != ordered.sha256
    compiler = NumericPolicyCompiler(grouped, Float32ProbabilityPolicy(1454))
    changed_alternatives = NumericPolicyCompiler(
        grouped, Float32ProbabilityPolicy(1453)
    )
    assert compiler.abi_sha256 != changed_alternatives.abi_sha256


def test_general_probability_codegen_supports_changed_alternative_counts():
    for alternatives in (1, 7, 8, 129, 1453, 1454):
        source = numpy_float32_choice_cuda_helpers(alternatives)
        assert f"numpy_pairwise_exp_sum_{alternatives}" in source
        assert "if (count < 8)" in source


def test_general_compiler_fails_closed_on_invalid_policies_and_shapes():
    with pytest.raises(ValueError, match="cover every term"):
        Float32ReductionPolicy(3, ((0, 2),), "grouped-left")
    with pytest.raises(ValueError, match="one term per group"):
        Float32ReductionPolicy(2, ((0, 1),), "ordered-left")
    with pytest.raises(ValueError, match="at least one alternative"):
        Float32ProbabilityPolicy(0)
    with pytest.raises(ValueError, match="shape"):
        reduce_float32(np.ones((2, 4)), np.ones(3), grouped_left_reduction(3))
