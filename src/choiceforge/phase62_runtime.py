"""Scoped preparation reuse, never chooser values or random answers.

The directory probe cache is confined to a synchronous model step. Numba's
source stamps, cache indexes, CPU/version checks and actual file writes remain
unchanged. This adapter is not a concurrent-filesystem transaction mechanism.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import copy
import os
import threading

from .phase61_inputs import SharedInputs


def node_sources(tree):
    """Same ordered traversal as the strict compiler, without typing wrappers."""
    op = tree["op"]
    if op == "name":
        yield ("name", tree["name"])
    elif op == "column":
        yield ("column", tree["name"])
    elif op == "skim":
        yield ("skim", tree["direction"], tree["key"]["value"])
    for key in ("arg", "left", "right", "value"):
        child = tree.get(key)
        if isinstance(child, Mapping):
            yield from node_sources(child)
    for key in ("args", "rights"):
        for child in tree.get(key, ()):
            yield from node_sources(child)
    for child in tree.get("keywords", {}).values():
        yield from node_sources(child)


class SpecificationCompiler:
    def __init__(self):
        self.trees = {}
        self.hits = self.misses = self.calls = 0

    def __call__(self, spec):
        from . import sharrow_ir as ir
        if "Expression" not in spec.columns or not spec.columns.is_unique:
            # Preserve upstream validation, including unsupported schemas.
            return self.original(spec)
        columns = list(spec.columns)
        alternatives = [c for c in columns if c not in {"Label", "Description", "Expression"}]
        positions = {c:i for i,c in enumerate(columns)}
        terms = []
        self.calls += 1
        for position, row in enumerate(spec.itertuples(index=False, name=None)):
            expression = str(row[positions["Expression"]]).strip()
            if not expression or expression.lower() == "nan":
                continue
            if expression not in self.trees:
                if len(self.trees) >= 8192:
                    self.trees.clear()
                self.trees[expression] = ir.expression_ir(expression)
                self.misses += 1
            else:
                self.hits += 1
            terms.append({"position":position,
                "label":str(row[positions["Label"]]) if "Label" in positions else f"expression_{position}",
                "expression":expression, "tree":copy.deepcopy(self.trees[expression]),
                "coefficients":{alt:ir._coefficient(row[positions[alt]]) for alt in alternatives}})
        document = {"ir_version":3, "numeric_policy":dict(ir._NUMERIC_POLICY),
                    "alternatives":alternatives, "terms":terms}
        document["sha256"] = ir.ir_sha256(document)
        return document


class DemandInputs(SharedInputs):
    """Defer publication until a live IR requests columns; preserve exact checks."""
    def __init__(self):
        super().__init__()
        self.states = {}
        self.requests = []

    def publish(self, state, table):
        self.states[table] = state

    def bind(self, table, frame, environment, document=None):
        if document is None or table not in self.states:
            return
        live = self.states[table].get_dataframe(table)
        names = {s[1] for t in document["terms"] for s in node_sources(t["tree"])
                 if s[0] in {"name", "column"}}
        columns = {n:live[n].to_numpy() for n in sorted(names)
                   if n in live and n in frame and live[n].dtype.kind in "biuf"}
        if columns:
            self.store.publish(table, live.index, columns)
            self.requests.append({"table":table,"columns":list(columns)})
            super().bind(table, frame, environment, document)


class DirectoryProbes:
    def __init__(self, original):
        self.original = original
        self.owner = threading.get_ident()
        self.identities = {}
        self.hits = self.misses = 0

    def ensure(self, locator):
        path = os.path.abspath(locator.get_cache_path())
        if threading.get_ident() != self.owner:
            return self.original(locator)
        def identity():
            s = os.stat(path)
            return (s.st_dev, s.st_ino, s.st_mode)
        try:
            current = identity()
        except OSError:
            current = None
        if current is not None and self.identities.get(path) == current:
            self.hits += 1
            return
        self.original(locator)  # Failed probes are never remembered.
        self.identities[path] = identity()
        self.misses += 1


class Runtime:
    def __init__(self, phase61, features="plans,trip,entities"):
        self.features = set(features.split(","))
        if not self.features <= {"plans","trip","entities"}:
            raise ValueError("Unsupported Phase 62 features")
        self.compiler = SpecificationCompiler()
        self.events = []
        self.shared = DemandInputs() if "entities" in self.features else None
        if self.shared is not None:
            phase61.shared = self.shared

    @contextmanager
    def for_step(self, state, step):
        from . import sharrow_ir, sharrow_cuda
        from numba.core.caching import _CacheLocator
        patches = []
        probes = None
        def patch(obj, name, value):
            patches.append((obj,name,getattr(obj,name)))
            setattr(obj,name,value)
        try:
            if "plans" in self.features:
                self.compiler.original = sharrow_ir.specification_ir
                patch(sharrow_ir,"specification_ir",self.compiler)
                patch(sharrow_cuda,"_node_sources",node_sources)
                probes = DirectoryProbes(_CacheLocator.ensure_cache_path)
                patch(_CacheLocator,"ensure_cache_path",lambda locator:probes.ensure(locator))
            old = os.environ.get("CHOICEFORGE_PHASE62_SKIP_UNUSED_KERNEL")
            if "trip" in self.features:
                os.environ["CHOICEFORGE_PHASE62_SKIP_UNUSED_KERNEL"] = "1"
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop("CHOICEFORGE_PHASE62_SKIP_UNUSED_KERNEL",None)
                else:
                    os.environ["CHOICEFORGE_PHASE62_SKIP_UNUSED_KERNEL"] = old
        finally:
            for obj,name,value in reversed(patches):
                setattr(obj,name,value)
            self.events.append({"step":step,"directory_probe_hits":probes.hits if probes else 0,
                                "directory_probe_misses":probes.misses if probes else 0})

    def summary(self):
        return {"enabled":True,"features":sorted(self.features),"events":self.events,
                "expression_hits":self.compiler.hits,"expression_misses":self.compiler.misses,
                "specification_calls":self.compiler.calls,"saved_answers_read":False,
                "demand_requests":self.shared.requests if self.shared else []}
