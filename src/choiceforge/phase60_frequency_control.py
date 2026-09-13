"""Diagnostic-only real-input Sharrow thread sweep; never timing evidence."""
from contextlib import contextmanager
import hashlib
import inspect
import statistics
import time

import numba
import numpy as np


class MeasuredDispatcher:
    def __init__(self, original, records):
        self.original, self.records = original, records

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        previous = numba.get_num_threads()
        expected = self.original(*args, **kwargs)
        expected_bytes = np.ascontiguousarray(expected).view(np.uint8)
        samples = {t:[] for t in (1,4,12,24,48) if t <= numba.config.NUMBA_NUM_THREADS}
        try:
            # Warm every mask before balanced repeated measurements.
            for threads in samples:
                numba.set_num_threads(threads)
                np.testing.assert_array_equal(np.ascontiguousarray(self.original(*args, **kwargs)).view(np.uint8), expected_bytes)
            for repetition in range(3):
                order = list(samples) if repetition % 2 == 0 else list(reversed(samples))
                for threads in order:
                    numba.set_num_threads(threads)
                    started = time.perf_counter()
                    actual = self.original(*args, **kwargs)
                    samples[threads].append(time.perf_counter()-started)
                    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8), expected_bytes)
        finally:
            numba.set_num_threads(previous)
        arrays = [np.asarray(a) for a in args if isinstance(a,np.ndarray)]
        arrays += [np.asarray(v) for v in kwargs.values() if isinstance(v,np.ndarray)]
        self.records.append({"output_shape":list(expected.shape), "output_dtype":str(expected.dtype),
            "utilities_byte_exact_all_masks":True, "samples_seconds":samples,
            "median_seconds":{t:statistics.median(v) for t,v in samples.items()},
            "input_array_sha256":[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays]})
        return expected


@contextmanager
def measure(records):
    from sharrow.flows import Flow
    original = Flow._iload_raw
    signature = inspect.signature(original)

    def call(flow, *args, **kwargs):
        bound = signature.bind(flow, *args, **kwargs)
        if (bound.arguments.get("dot") is None or bound.arguments.get("mnl") is not None
                or bound.arguments.get("runner") is not None):
            return original(flow, *args, **kwargs)
        dispatcher = flow._idotter
        flow._idotter = MeasuredDispatcher(dispatcher, records)
        try:
            return original(flow, *args, **kwargs)
        finally:
            flow._idotter = dispatcher
    Flow._iload_raw = call
    try:
        yield
    finally:
        Flow._iload_raw = original
