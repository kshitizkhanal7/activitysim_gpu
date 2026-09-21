"""Data-only reuse of Sharrow's generated expression plans.

No Flow, DataTree, array, coefficient result or choice is persisted. The cache
stores the exact output of upstream init_sub_funcs, and every request still
runs upstream initialize_1 on its live tree. Cache identities include that
flow hash, definitions, compiler options, implementation bytes, extra function
code and the generated module's source bytes. Unknown formats fall back.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path
import threading


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ExpressionPlans:
    def __init__(self):
        import sharrow.aster
        import sharrow.flows
        import sharrow.relationships
        self.implementation = sha(b"".join(Path(m.__file__).read_bytes() for m in
            (sharrow.aster, sharrow.flows, sharrow.relationships)))
        self.owner = threading.get_ident()
        self.hits = self.misses = self.rejected = self.writes = 0
        self.events = []

    def key(self, flow, defs, options):
        if flow._hashing_level > 1 or not hasattr(flow, "flow_hash"):
            return None
        functions = []
        for function in flow.tree.extra_funcs or ():
            function = getattr(function, "py_func", function)
            if not hasattr(function, "__code__") or function.__closure__:
                return None
            # Rehash serialized defaults on every request; unsupported defaults
            # are not eligible for reuse.
            defaults = (function.__defaults__, function.__kwdefaults__)
            try:
                defaults_hash = sha(encoded(defaults))
            except (TypeError, ValueError):
                return None
            functions.append((function.__module__, function.__qualname__,
                              sha(marshal.dumps(function.__code__)), defaults_hash))
        return sha(encoded({"format":1, "implementation":self.implementation,
            "flow":flow.flow_hash, "defs":list(defs.items()), "options":options,
            "functions":functions, "arg_positions":flow.arg_name_positions,
            "dim_order":flow.dim_order, "dim_exclude":flow.dim_exclude}))

    def compile(self, original, flow, defs, *args, **kwargs):
        if threading.get_ident() != self.owner:
            return original(flow, defs, *args, **kwargs)
        bound = inspect.signature(original).bind(flow, defs, *args, **kwargs)
        bound.apply_defaults()
        options = {k:v for k,v in bound.arguments.items() if k not in {"self", "defs"}}
        key = self.key(flow, defs, options)
        if key is None:
            return original(flow, defs, *args, **kwargs)
        directory = Path(flow.cache_dir) / "choiceforge_phase63_plans"
        target = directory / (key + ".json")
        source = Path(flow.cache_dir) / ("flow_" + flow.flow_hash) / "__init__.py"
        try:
            source_digest = sha(source.read_bytes())
        except OSError:
            source_digest = None
        if source_digest is not None and target.exists():
            try:
                if target.stat().st_size > 16 * 1024**2:
                    raise ValueError("oversized plan")
                record = json.loads(target.read_bytes())
                payload = record["payload"]
                if record["sha256"] != sha(encoded(payload)) or payload["key"] != key:
                    raise ValueError("plan identity changed")
                if payload["source_sha256"] != source_digest:
                    raise ValueError("generated source changed")
                raw = payload["raw_functions"]
                if [row[0] for row in raw] != list(defs):
                    raise ValueError("plan definition order differs")
                if [row[1] for row in raw] != [str(v).lstrip() for v in defs.values()]:
                    raise ValueError("plan definitions differ")
                code, tokens = payload["code"], payload["tokens"]
                if not isinstance(code, str) or not all(isinstance(t, str) for t in tokens):
                    raise ValueError("invalid plan metadata")
                flow._raw_functions = {name:(initial, expression, set(names), arguments)
                    for name,initial,expression,names,arguments in raw}
                flow.output_name_positions = dict(payload["output_positions"])
                flow.arg_name_positions = dict(payload["arg_positions"])
                self.hits += 1
                self.events.append({"flow_hash":flow.flow_hash, "hit":True,
                                    "source_sha256":source_digest})
                return code, set(tokens)
            except (OSError, ValueError, KeyError, TypeError):
                self.rejected += 1
        self.misses += 1
        code, tokens = original(flow, defs, *args, **kwargs)
        if source_digest is not None:
            payload = {"key":key, "source_sha256":source_digest, "code":code,
                "tokens":sorted(tokens), "output_positions":list(flow.output_name_positions.items()),
                "arg_positions":list(flow.arg_name_positions.items()),
                "raw_functions":[[name,a,b,sorted(c),d] for name,(a,b,c,d) in flow._raw_functions.items()]}
            data = encoded({"payload":payload, "sha256":sha(encoded(payload))})
            if len(data) <= 16 * 1024**2:
                # Unique temporary sibling, atomic replacement; source cache is
                # disposable program metadata. No benchmark output is overwritten.
                directory.mkdir(exist_ok=True)
                import tempfile
                with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as stream:
                    stream.write(data)
                    temporary = stream.name
                os.replace(temporary, target)
                self.writes += 1
        self.events.append({"flow_hash":flow.flow_hash, "hit":False,
                            "source_sha256":source_digest})
        return code, tokens

    @contextmanager
    def scope(self):
        from sharrow.flows import Flow
        original = Flow.init_sub_funcs
        Flow.init_sub_funcs = lambda flow, defs, *a, **k:self.compile(original,flow,defs,*a,**k)
        try:
            yield
        finally:
            Flow.init_sub_funcs = original

    def summary(self):
        return {"hits":self.hits, "misses":self.misses, "writes":self.writes,
                "rejected":self.rejected, "events":self.events,
                "implementation_sha256":self.implementation,
                "saved_answers_read":False, "scope":"data-only upstream expression plan"}
