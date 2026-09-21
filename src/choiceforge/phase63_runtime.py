"""Scoped reusable preparation with live bindings and private parsed inputs."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

from .phase63_plans import ExpressionPlans


class SpecificationFiles:
    def __init__(self):
        self.frames = {}
        self.hits = self.misses = 0

    @staticmethod
    def private_copy(frame):
        result = frame.copy(deep=True)
        # pandas deep copies normally share immutable Index backing arrays.
        # Also detach those arrays so deliberate low-level mutation cannot leak.
        result.index = frame.index.copy(deep=True)
        result.columns = frame.columns.copy(deep=True)
        return result

    def read(self, original, *args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        fs = bound.arguments["filesystem"]
        name = bound.arguments.get("file_name")
        settings = bound.arguments.get("model_settings")
        kind = original.__name__
        if kind == "read_model_coefficients" and name is not None and settings is not None:
            # Preserve upstream's rejection of mutually exclusive arguments,
            # including when a matching file has already been cached.
            return original(*args, **kwargs)
        if name is None and settings is not None:
            field = "COEFFICIENT_TEMPLATE" if kind == "read_model_coefficient_template" else "COEFFICIENTS"
            name = settings.get(field) if isinstance(settings, dict) else getattr(settings, field, None)
        if name is None:
            return original(*args, **kwargs)
        if kind == "read_model_spec" and not str(name).lower().endswith(".csv"):
            name = str(name) + ".csv"
        path = Path(fs.get_config_file_path(name)).resolve()
        # Check actual bytes on EVERY request, including same-size edits with
        # preserved timestamps. Settings resolution itself is not cached.
        key = (kind, str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        if key not in self.frames:
            self.misses += 1
            frame = original(*args, **kwargs)
            if len(self.frames) >= 256:
                self.frames.clear()
            self.frames[key] = self.private_copy(frame)
            return frame
        self.hits += 1
        return self.private_copy(self.frames[key])


class Runtime:
    def __init__(self, features="plans,files"):
        self.features = set(features.split(","))
        if not self.features <= {"plans", "files", "rss"}:
            raise ValueError("Unsupported Phase 63 features")
        self.plans = ExpressionPlans()
        self.files = SpecificationFiles()
        self.events = []
        self.memory_events = []

    @contextmanager
    def for_step(self, state, step):
        from contextlib import ExitStack
        from activitysim.core import simulate
        patches = []
        before = self.plans.hits, self.files.hits
        memory_prior = None
        memory_event = None
        try:
            with ExitStack() as stack:
                if "rss" in self.features:
                    # USS page walking is expensive on this workstation. Only
                    # change diagnostic telemetry when it cannot affect adaptive
                    # chunk sizing. RSS logs remain; USS is explicitly unavailable
                    # inside this scope, not interpreted as zero memory usage.
                    if state.settings.chunk_size != 0 or state.settings.chunk_training_mode != "disabled":
                        raise ValueError("RSS-only tracing requires disabled chunking/training")
                    import psutil
                    from activitysim.core import mem
                    memory_prior = mem.USS
                    mem.USS = False
                    memory_event = {"step":step,"uss_in_step":"not sampled"}
                    self.memory_events.append(memory_event)
                    memory_event["start_rss"] = psutil.Process().memory_info().rss
                if "plans" in self.features:
                    stack.enter_context(self.plans.scope())
                if "files" in self.features:
                    for name in ("read_model_spec", "read_model_coefficients", "read_model_coefficient_template"):
                        original = getattr(simulate, name)
                        patches.append((name, original))
                        setattr(simulate, name, lambda *a, _original=original, **k:self.files.read(_original,*a,**k))
                yield
        finally:
            # Restore hooks even if diagnostic sampling itself fails.
            if memory_prior is not None:
                mem.USS = memory_prior
            for name, original in reversed(patches):
                setattr(simulate, name, original)
            self.events.append({"step":step,"plan_hits":self.plans.hits-before[0],
                                "file_hits":self.files.hits-before[1]})
            if memory_event is not None:
                memory_event["end_rss"] = psutil.Process().memory_info().rss

    def summary(self):
        return {"enabled":True,"features":sorted(self.features),"events":self.events,
                "plans":self.plans.summary(),"files":{"hits":self.files.hits,"misses":self.files.misses},
                "memory_events":self.memory_events,
                "saved_answers_read":False}

    @contextmanager
    def cpu_steps(self):
        """Give unmodified CPU steps the same preparation/diagnostic options."""
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
