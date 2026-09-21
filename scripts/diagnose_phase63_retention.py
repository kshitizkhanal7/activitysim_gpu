"""Instrument an isolated batch reset to locate retained GPU workspace owners.

Diagnostic ONLY: garbage-collector scans are not timing evidence. This wrapper
does not release buffers or modify model arithmetic. It emits object types,
attribute names and sizes, never input values or saved choices.
"""
import gc
import json
from pathlib import Path
import sys
import types

import phase62_batch_worker as worker


def describe_referrers(obj,ignored):
    result = []
    refs = gc.get_referrers(obj)
    for ref in refs:
        if id(ref) in ignored or ref is refs:
            continue
        if isinstance(ref,dict):
            keys = [str(k) for k,v in tuple(ref.items()) if v is obj]
            parents = []
            for parent in gc.get_referrers(ref):
                kind = type(parent)
                if isinstance(kind.__module__,str) and kind.__module__.startswith("choiceforge.") and getattr(parent,"__dict__",None) is ref:
                    upstream = []
                    for ancestor in gc.get_referrers(parent):
                        if isinstance(ancestor,dict):
                            upstream.append({"type":"dict","module":ancestor.get("__name__"),
                                "keys":[str(k) for k,v in tuple(ancestor.items()) if v is parent][:20]})
                        elif isinstance(ancestor,types.CellType):
                            upstream.append({"type":"closure_cell"})
                    parents.append({"type":kind.__module__+"."+kind.__name__,"referrers":upstream})
            result.append({"type":"dict","module":ref.get("__name__"),"keys":keys[:20],"parents":parents})
        elif isinstance(ref,types.CellType):
            functions = []
            for group in gc.get_referrers(ref):
                if isinstance(group,tuple):
                    for function in gc.get_referrers(group):
                        if isinstance(function,types.FunctionType) and function.__closure__ is group:
                            functions.append(function.__module__+"."+function.__qualname__)
            result.append({"type":"closure_cell","functions":functions[:20]})
        else:
            result.append({"type":type(ref).__module__+"."+type(ref).__name__})
    return result[:30]


def inspect_owners():
    cp = sys.modules.get("cupy")
    if cp is None:
        return []
    objects = gc.get_objects()
    owners = []
    for obj in objects:
        kind = type(obj)
        if not isinstance(kind.__module__,str) or not kind.__module__.startswith("choiceforge.") or not hasattr(obj,"__dict__"):
            continue
        arrays = {name:int(value.nbytes) for name,value in vars(obj).items() if isinstance(value,cp.ndarray)}
        if sum(arrays.values())<1024**2:
            continue
        owners.append({"type":kind.__module__+"."+kind.__name__,"id":id(obj),"arrays":arrays,
                       "referrers":describe_referrers(obj,{id(objects)})})
    return sorted(owners,key=lambda o:-sum(o["arrays"].values()))


if __name__=="__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text())
    target = Path(manifest["result"]).with_suffix(".retention.json")
    original = worker.reset_application_graph
    records = []
    def reset(*args,**kwargs):
        before = inspect_owners()
        count = original(*args,**kwargs)
        after = inspect_owners()
        records.append({"before_reset":before,"after_reset":after,"memory":worker.memory_snapshot()})
        target.write_text(json.dumps({"diagnostic_not_timing":True,"resets":records},indent=2)+"\n")
        return count
    worker.reset_application_graph = reset
    worker.main()
