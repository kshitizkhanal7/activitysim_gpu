import numpy as np
import pytest
from choiceforge.cuda_backend import _cupy
from choiceforge.phase59_cpu_algorithms import modes_cpu
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from test_nested_logit import NEST


@pytest.mark.parametrize("seed", [59, 591, 592])
def test_compiled_cpu_mode_control_agrees_off_boundaries(seed):
    cp = _cupy()
    rng = np.random.default_rng(seed)
    values = rng.normal(-2, 4, (1027, 21)).astype(np.float32)
    values[::7, 8:18] = -999
    draws = rng.random(len(values))
    coefficients = validate(NEST, MTC21_ALTERNATIVES)
    cpu, cpu_logsums, cpu_guards = modes_cpu(values, draws, coefficients)
    device, logsums, guards, _ = reduce_modes(cp.asarray(values), cp.asarray(draws), NEST)
    safe = (cpu_guards == 0) & (cp.asnumpy(guards) == 0)
    np.testing.assert_array_equal(cpu[safe], cp.asnumpy(device)[safe])
    np.testing.assert_allclose(cpu_logsums, cp.asnumpy(logsums), atol=1e-12, rtol=1e-12)
