import numba
import numpy as np
from choiceforge.phase57_live_scheduling import period_pairs_cuda
from choiceforge.phase58_cpu_control import period_pairs_cpu


def test_cpu_control_exact_and_restores_thread_count():
    rng = np.random.default_rng(58)
    codes = np.array([0, 2, 4, 6, 7], dtype=np.int8)
    footprints = rng.choice(codes, size=(190, 21))
    footprints[:3] = 0
    slots = rng.integers(0, 25, 190, dtype=np.int32)
    previous = numba.get_num_threads()
    for windows in (rng.choice(codes, size=(123, 21)), np.zeros((123, 21), dtype=np.int8)):
        expected = period_pairs_cuda(windows, footprints, slots)
        for threads in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
            actual = period_pairs_cpu(windows, footprints, slots, threads)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            assert numba.get_num_threads() == previous
