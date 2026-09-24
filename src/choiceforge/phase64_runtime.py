"""Live expression bytecode reuse and explicit mode-reducer CPU ablation.

Only immutable Python code objects are reused. Each expression is evaluated
against the caller's current globals/locals on every request. This is a scoped
sequential adapter, not a concurrent interpreter or an expression sandbox.
"""
import builtins
from contextlib import contextmanager
import threading


class ExpressionCode:
    def __init__(self, capacity=8192):
        if capacity < 1:
            raise ValueError("Positive expression capacity required")
        self.capacity = capacity
        self.codes = {}
        self.owner = threading.get_ident()
        self.hits = self.misses = self.fallbacks = 0

    def evaluate(self, source, globals_dict, locals_dict):
        # ActivitySim's patched call sites supply both dictionaries explicitly.
        # Other threads/code-object/bytes inputs retain builtin execution.
        if (threading.get_ident() != self.owner or type(source) is not str
                or len(source) > 262144):
            self.fallbacks += 1
            return builtins.eval(source, globals_dict, locals_dict)
        code = self.codes.get(source)
        if code is None:
            # eval(string) strips initial spaces/tabs; compile(string) does not.
            code = compile(source.lstrip(" \t"), "<string>", "eval", dont_inherit=True)
            if len(self.codes) >= self.capacity:
                self.codes.clear()
            self.codes[source] = code
            self.misses += 1
        else:
            self.hits += 1
        return builtins.eval(code, globals_dict, locals_dict)


class Runtime:
    def __init__(self, features="expressions"):
        self.features = set(features.split(","))
        if not self.features or not self.features <= {"expressions", "mode_cpu", "inputs", "location_boundary"}:
            raise ValueError("Unsupported Phase 64 features")
        self.expressions = ExpressionCode()
        self.events = []
        self.mode_events = []
        self.active = False
        self.inputs = None
        self.location_boundary = None
        if "location_boundary" in self.features:
            from .phase64_location_boundary import LocationBoundary
            self.location_boundary = LocationBoundary()
        if "inputs" in self.features:
            from pathlib import Path
            from .phase64_inputs import NumericInputs
            self.inputs = NumericInputs(Path(__file__).resolve().parents[2]/"benchmark-data/phase64-numeric-inputs")

    def cpu_reduce(self, utilities, draws, nest):
        """Same nested algorithm on CPU, with device-boundary transfers charged.

        This ablates reduction only: expression generation/RNG remain unchanged.
        It is deliberately not called a complete CPU ActivitySim replacement.
        """
        import time
        import numpy as np
        from .cuda_backend import _cupy
        from .phase58_mode_reduction import validate
        from .phase59_cpu_algorithms import modes_cpu
        from .nested_logit import MTC21_ALTERNATIVES
        cp = _cupy()
        started = time.perf_counter()
        mus = validate(nest, MTC21_ALTERNATIVES)
        host_utilities = cp.asnumpy(utilities)
        host_draws = cp.asnumpy(draws).reshape(-1)
        download_seconds = time.perf_counter()-started
        if (host_utilities.shape != (len(host_draws),21) or not len(host_draws)
                or not np.isfinite(host_utilities).all() or not np.isfinite(host_draws).all()
                or (host_draws<0).any() or (host_draws>=1).any()):
            raise ValueError("CPU reducer ablation requires finite aligned live inputs")
        scales=np.r_[np.full(6,mus[0]*mus[1]),np.full(2,mus[2]),np.full(10,mus[3]*mus[4]),np.full(3,mus[5])]
        if (host_utilities.astype(np.float64)/scales>690).any():
            raise ValueError("CPU reducer ablation exceeds GPU finite-domain limit")
        import numba
        prior_threads=numba.get_num_threads()
        try:
            numba.set_num_threads(min(48,numba.config.NUMBA_NUM_THREADS))
            choices, logsums, guards = modes_cpu(host_utilities, host_draws, mus)
        finally:
            numba.set_num_threads(prior_threads)
        if not np.isfinite(logsums).all():
            raise ValueError("CPU reducer produced nonfinite logsums")
        result = tuple(cp.asarray(x) for x in (choices, logsums, guards))
        cp.cuda.Stream.null.synchronize()
        self.mode_events.append(dict(rows=len(host_draws), seconds=time.perf_counter()-started,
            download_seconds=download_seconds,
            device_to_host_bytes=host_utilities.nbytes+host_draws.nbytes,
            host_to_device_bytes=sum(x.nbytes for x in (choices,logsums,guards)),
            guarded_rows=int(np.count_nonzero(guards))))
        return (*result, None)

    @contextmanager
    def for_step(self, state, step):
        if self.active:
            raise ValueError("Phase 64 scopes cannot overlap")
        self.active = True
        patches = []
        sentinel = object()
        before = self.expressions.hits, self.expressions.misses
        def patch(module, name, value):
            patches.append((module, name, vars(module).get(name, sentinel)))
            setattr(module, name, value)
        try:
            if "expressions" in self.features:
                from activitysim.core import simulate, assign
                for module in (simulate, assign):
                    patch(module, "eval", self.expressions.evaluate)
            if self.inputs is not None:
                from activitysim.core import input as input_module
                original=input_module._read_csv_with_fallback_encoding
                patch(input_module,"_read_csv_with_fallback_encoding",
                      lambda path,dtypes=None:self.inputs.read(original,path,dtypes))
            if "mode_cpu" in self.features and step in {"tour_mode_choice_simulate", "trip_mode_choice"}:
                from . import phase58_mode_reduction
                patch(phase58_mode_reduction, "reduce_modes", self.cpu_reduce)
                patch(phase58_mode_reduction, "_cpu_reducer_control", self)
            if self.location_boundary is not None and step in {"school_location", "workplace_location",
                    "joint_tour_destination", "non_mandatory_tour_destination", "atwork_subtour_destination"}:
                from .destination_input_supergraph import DestinationInputSupergraph
                from . import modelwide_final, modelwide_sampling
                controller=self.location_boundary
                patch(DestinationInputSupergraph,"compute",lambda producer,*a,**k:controller.compute(producer,*a,**k))
                patch(modelwide_final,"device_compact_interaction_sample_simulate",controller.final)
                patch(modelwide_sampling,"_phase64_sampling_observer",controller.sample)
            yield
        finally:
            for module, name, original in reversed(patches):
                if original is sentinel:
                    vars(module).pop(name, None)
                else:
                    setattr(module, name, original)
            self.active = False
            self.events.append(dict(step=step, code_hits=self.expressions.hits-before[0],
                                    code_misses=self.expressions.misses-before[1]))

    @contextmanager
    def cpu_steps(self):
        from activitysim.core.workflow.runner import Runner
        original = Runner.by_name
        runtime = self
        def run(runner, model_name, *args, **kwargs):
            with runtime.for_step(runner._obj, str(model_name)):
                return original(runner, model_name, *args, **kwargs)
        Runner.by_name = run
        try:
            yield
        finally:
            Runner.by_name = original

    def summary(self):
        return dict(enabled=True, features=sorted(self.features), events=self.events,
                    code_hits=self.expressions.hits, code_misses=self.expressions.misses,
                    code_fallbacks=self.expressions.fallbacks, code_entries=len(self.expressions.codes),
                    code_capacity=self.expressions.capacity, mode_events=self.mode_events,
                    inputs=self.inputs.summary() if self.inputs is not None else None,
                    location_boundary=self.location_boundary.summary() if self.location_boundary is not None else None,
                    saved_answers_read=False, scope="live expression code, optional CPU reducer ablation")
