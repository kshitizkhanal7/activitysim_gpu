import numpy as np
import pytest

from choiceforge.cuda_backend import cuda_available
from choiceforge.scheduling_compiler import (
    CompiledCpuSchedulingModel,
    CompiledCudaSchedulingModel,
    SchedulingSchema,
    compile_cuda_expression,
)


def test_compiles_activitysim_boolean_arithmetic():
    value = compile_cuda_expression("(ptype == 1) & (duration < 9)")
    assert "&&" in value and "ptype" in value and "duration" in value


def test_rejects_function_calls():
    with pytest.raises(ValueError, match="unsupported syntax"):
        compile_cuda_expression("exp(duration)")


def test_rejects_unsafe_schema_name():
    schema = SchedulingSchema(("float",), (), ("start",))
    with pytest.raises(ValueError, match="invalid generated-source"):
        CompiledCpuSchedulingModel(("start",), (1.0,), schema)


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_compact_cpu_cuda_choices_match():
    schema = SchedulingSchema(("x", "ptype"), ("bonus",), ("start", "duration"))
    expressions = ("x * start", "bonus", "(ptype == 1) & (duration < 2)")
    coefficients = np.array([-0.4, 0.7, 1.2], dtype=np.float32)
    chooser = np.array([[2, 1], [1, 2]], dtype=np.float32)
    rows = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
    alternatives = np.array([[1, 1], [2, 2], [3, 1]], dtype=np.float32)
    alt_ids = np.array([0, 1, 0, 1, 2], dtype=np.int16)
    offsets = np.array([0, 2, 5], dtype=np.int64)
    draws = np.array([0.25, 0.75], dtype=np.float64)
    cpu = CompiledCpuSchedulingModel(expressions, coefficients, schema).choose(
        chooser, rows, alternatives, alt_ids, offsets, draws
    )
    gpu = CompiledCudaSchedulingModel(expressions, coefficients, schema).choose(
        chooser, rows, alternatives, alt_ids, offsets, draws
    )
    assert np.array_equal(cpu.choices, gpu.choices)
    assert np.allclose(cpu.logsums, gpu.logsums, atol=2e-6)


@pytest.mark.skipif(not cuda_available(), reason="CUDA device unavailable")
def test_scheduling_keeps_activitysim_float64_draw_at_float32_probability_boundary():
    schema = SchedulingSchema(("x",), (), ("start",))
    expressions = ("x * start",)
    coefficients = np.asarray([0.0], dtype=np.float32)
    chooser = np.asarray([[1.0]], dtype=np.float32)
    rows = np.empty((2, 0), dtype=np.float32)
    alternatives = np.asarray([[1.0], [2.0]], dtype=np.float32)
    alt_ids = np.asarray([0, 1], dtype=np.int16)
    offsets = np.asarray([0, 2], dtype=np.int64)
    # Rounding this draw to float32 changes it to exactly 0.5 and incorrectly
    # selects alternative zero. ActivitySim retains the float64 draw.
    draws = np.asarray([np.nextafter(0.5, 1.0)], dtype=np.float64)
    cpu = CompiledCpuSchedulingModel(expressions, coefficients, schema).choose(
        chooser, rows, alternatives, alt_ids, offsets, draws
    )
    gpu = CompiledCudaSchedulingModel(expressions, coefficients, schema).choose(
        chooser, rows, alternatives, alt_ids, offsets, draws
    )
    np.testing.assert_array_equal(cpu.choices, [1])
    np.testing.assert_array_equal(gpu.choices, [1])
