"""One process, fresh application module graph per scenario, program-only reuse.

Not a general interactive application hot-reloader: exclusively owns this
worker process and executes one model at a time. Exceptions terminate the batch.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import time


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8*1024**2):
            digest.update(block)
    return digest.hexdigest()


def verify_files(records):
    for path,expected in records.items():
        if file_digest(path) != expected:
            raise ValueError(f"Batch input or generated program changed: {path}")


class InputTables:
    """Private numeric raw-input snapshots; each model gets a deep copy.

    The caller verifies source digests at every scenario boundary. Unsupported
    object/extension dtypes are parsed normally, never shared through this cache.
    """
    def __init__(self, sources, limit=1024**3):
        self.sources = sources
        self.limit = limit
        self.frames = {}
        self.bytes = self.hits = self.misses = 0

    def read(self, original, filepath, dtypes=None):
        import numpy as np
        path = str(Path(filepath).resolve())
        if path not in self.sources:
            return original(filepath,dtypes)
        if dtypes is not None and (not isinstance(dtypes,dict) or any(not isinstance(k,str) for k in dtypes)):
            return original(filepath,dtypes)
        try:
            types = None if dtypes is None else tuple(sorted((str(k),np.dtype(v).str) for k,v in dtypes.items()))
        except (TypeError,AttributeError):
            return original(filepath,dtypes)
        key = (path,self.sources[path],types)
        if key in self.frames:
            self.hits += 1
            return self.frames[key].copy(deep=True)
        self.misses += 1
        frame = original(filepath,dtypes)
        size = int(frame.memory_usage(index=True,deep=True).sum())
        if all(dtype.kind in "biuf" for dtype in frame.dtypes) and self.bytes+size<=self.limit:
            self.frames[key] = frame.copy(deep=True)
            self.bytes += size
        return frame

    def summary(self):
        return {"hits":self.hits,"misses":self.misses,"retained_bytes":self.bytes,
                "tables":len(self.frames),"limit_bytes":self.limit,"returns_private_copies":True}


class SkimPool:
    """Bounded content-addressed skim copies; no pointer-based cross-run keys."""
    def __init__(self, limit=2 * 1024**3):
        self.limit = limit
        self.arrays = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.hashed_bytes = 0

    def upload(self, cp, values):
        import numpy as np
        host = np.ascontiguousarray(values)
        # Hash every upload request. Even an in-place change at the same address
        # cannot reuse a stale answer. Only raw immutable-network copies are kept.
        digest = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
        self.hashed_bytes += host.nbytes
        key = (int(cp.cuda.Device().id), host.shape, host.dtype.str, digest)
        if key in self.arrays:
            self.hits += 1
            self.arrays.move_to_end(key)
            return self.arrays[key]
        self.misses += 1
        result = cp.ascontiguousarray(cp.asarray(host))
        if host.nbytes <= self.limit:
            while self.bytes + host.nbytes > self.limit:
                _, old = self.arrays.popitem(last=False)
                self.bytes -= old.nbytes
            self.arrays[key] = result
            self.bytes += result.nbytes
        return result

    def summary(self):
        return {"hits":self.hits,"misses":self.misses,"hashed_bytes":self.hashed_bytes,
                "retained_bytes":self.bytes,"limit_bytes":self.limit}


def reset_application_graph(programs=None, cpu_programs=None):
    """Drop ALL application caches, registries, hooks and mutable services.

    Sharrow's content-hashed generated modules and numerical library programs
    stay loaded. No ActivitySim Flow/DataTree or ChoiceForge invocation survives
    in the next application's module graph. Old graph objects can remain owned
    by library internals; they are not installed into the new graph.
    """
    if programs is not None:
        # These two caches are explicitly keyed by generated source and IR
        # hashes. Do not retain compiled plans, coefficients or workspaces.
        import cupy as cp
        for module_name, attribute in (("choiceforge.sharrow_cuda","_KERNEL_CACHE"),
                                       ("choiceforge.native_abi_bootstrap","_NATIVE_KERNEL_CACHE")):
            module = sys.modules.get(module_name)
            if module is not None:
                cache = getattr(module,attribute)
                if not all(isinstance(v,cp.RawKernel) for v in cache.values()):
                    raise TypeError("Program-only reuse cannot retain arrays or invocation objects")
                programs[(module_name,attribute)] = dict(cache)
    if cpu_programs is not None:
        from numba.core.registry import CPUDispatcher
        # These modules' compiled functions take their live arrays/seeds as
        # arguments; they are not Flow trees, model states or output buffers.
        for module_name in ("activitysim.core.timetable", "choiceforge.phase61_normals",
                            "choiceforge.phase61_timetable"):
            module = sys.modules.get(module_name)
            if module is not None:
                for name,value in vars(module).items():
                    if isinstance(value,CPUDispatcher):
                        cpu_programs[(module_name,name)] = value
    names = [name for name in sys.modules if name == "activitysim" or name.startswith("activitysim.")
             or name == "choiceforge" or name.startswith("choiceforge.")]
    for name in names:
        del sys.modules[name]
    gc.collect()
    assert not any(n in sys.modules for n in names)
    return len(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    started = time.perf_counter()
    pool = SkimPool()
    programs = {} if manifest.get("mode") == "candidate" else None
    cpu_programs = {}
    tables = InputTables(manifest["data_sha256"])
    generated_sources = {}
    runs = []
    base_env = dict(os.environ)
    for item in manifest["runs"]:
        scenario_started = time.perf_counter()
        verify_files(manifest["data_sha256"])
        verify_files(generated_sources)
        removed = reset_application_graph(programs,cpu_programs)
        os.environ.clear()
        os.environ.update(base_env)
        os.environ.update(item["environment"])
        os.chdir(item["cwd"])
        for (module_name,attribute),program in cpu_programs.items():
            setattr(importlib.import_module(module_name),attribute,program)
        from activitysim.core import input as input_module
        original_read = input_module._read_csv_with_fallback_encoding
        input_module._read_csv_with_fallback_encoding = lambda filepath,dtypes=None,_original=original_read:tables.read(_original,filepath,dtypes)
        command = item["command"]
        if item["mode"] == "candidate":
            for (module_name,attribute),cache in programs.items():
                getattr(importlib.import_module(module_name),attribute).update(cache)
            import choiceforge.cuda_skims as skims
            skims._BATCH_SKIM_POOL = pool if manifest.get("skim_cache") == "content" else None
            sys.argv = command[1:]
            try:
                runpy.run_path(command[1],run_name="__main__")
            except SystemExit as exc:
                if exc.code:
                    raise
        else:
            import numba
            numba.set_num_threads(int(item["environment"]["NUMBA_NUM_THREADS"]))
            sys.argv = command
            from activitysim.cli.main import main as activitysim_main
            try:
                activitysim_main()
            except SystemExit as exc:
                if exc.code:
                    raise
        for name,module in tuple(sys.modules.items()):
            if name.startswith("flow_") and getattr(module,"__file__",None):
                path = str(Path(module.__file__).resolve())
                current = file_digest(path)
                if path in generated_sources and generated_sources[path] != current:
                    raise ValueError("Generated program changed during batch")
                generated_sources[path] = current
        runs.append({"name":item["name"],"scenario_wall_seconds":time.perf_counter()-scenario_started,
                     "reset_modules":removed,"skim_pool":pool.summary(),
                     "reused_source_keyed_cuda_programs":sum(len(c) for c in (programs or {}).values()),
                     "reused_cpu_programs":len(cpu_programs),"input_tables":tables.summary(),
                     "generated_program_modules":sum(n.startswith("flow_") for n in sys.modules)})
        # Close output handlers before the next model configures its own log.
        logging.shutdown()
        Path(manifest["result"]).write_text(json.dumps({"complete":False,"runs":runs},indent=2)+"\n")
    reset_application_graph()
    result = {"complete":True,"runs":runs,"batch_worker_seconds":time.perf_counter()-started,
              "skim_cache":manifest.get("skim_cache","none"),
              "input_tables":tables.summary(),
              "generated_program_sha256":generated_sources,"data_sha256":manifest["data_sha256"],
              "skim_pool":pool.summary(),"isolation":"fresh application module graph and workflow.State per scenario"}
    Path(manifest["result"]).write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
