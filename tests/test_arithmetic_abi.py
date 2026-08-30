import numba as nb
import numpy as np

from choiceforge.arithmetic_abi import (
    NUMPY_FLOAT32_CHOICE_ABI_SHA256,
    NUMPY_FLOAT32_CHOICE_ABI_VERSION,
    SHARROW15_ABI_SHA256,
    SHARROW15_ABI_VERSION,
    sharrow15_cpu_reduce,
    sharrow15_cuda_reduction,
    numpy_float32_choice_cuda_helpers,
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
