"""Versioned keyed device columns; CPU publication boundaries remain explicit.

A lease must be validated immediately before a consumer uses its arrays. Raw
CUDA pointers cannot revoke themselves: retaining a pointer past mutation is
outside this API's contract. Row-layout changes invalidate every prior lease.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .cuda_backend import _cupy


@dataclass(frozen=True)
class Lease:
    store: object
    table: str
    generation: int
    columns: tuple

    def arrays(self):
        record = self.store.tables.get(self.table)
        if record is None or record["generation"] != self.generation:
            raise ValueError("Phase 59 stale device entity lease")
        return tuple(record["device"][name] for name in self.columns)


class EntityStore:
    def __init__(self):
        self.tables = {}
        self.generation = 0
        self.events = []

    def publish(self, name, index, columns):
        """Copy a live CPU snapshot; reuse equal columns, never infer mutations."""
        cp = _cupy()
        index = pd.Index(index).copy(deep=True)
        if not index.is_unique:
            raise ValueError("Phase 59 entity IDs must be unique")
        arrays = {key:np.asarray(value) for key, value in columns.items()}
        if not arrays or any(a.ndim != 1 or len(a) != len(index) or a.dtype.kind not in "biuf" for a in arrays.values()):
            raise ValueError("Phase 59 entity columns must be aligned numeric vectors")
        prior = self.tables.get(name)
        same_layout = prior is not None and prior["index"].equals(index)
        device = dict(prior["device"]) if same_layout else {}
        host = dict(prior["host"]) if same_layout else {}
        changed = []
        for key, array in arrays.items():
            if (key not in host or host[key].dtype != array.dtype or not np.array_equal(
                    host[key].view(np.uint8), np.ascontiguousarray(array).view(np.uint8))):
                host[key] = np.ascontiguousarray(array).copy()
                device[key] = cp.asarray(host[key])
                changed.append(key)
        if changed or not same_layout:
            self.generation += 1
            self.tables[name] = {"generation":self.generation, "index":index, "host":host, "device":device}
            self.events.append({"table":name, "generation":self.generation, "rows":len(index),
                "columns_uploaded":changed, "host_to_device_bytes":sum(host[k].nbytes for k in changed),
                "layout_changed":not same_layout})
        return self.lease(name, tuple(arrays))

    def invalidate(self, name):
        """Drop a deleted/replaced authoritative table; all its leases expire."""
        if name in self.tables:
            del self.tables[name]
            self.generation += 1
            self.events.append({"table":name, "generation":self.generation, "invalidated":True,
                                "host_to_device_bytes":0})

    def lease(self, name, columns):
        record = self.tables[name]
        if not set(columns) <= record["device"].keys():
            raise ValueError("Phase 59 requested missing entity columns")
        return Lease(self, name, record["generation"], tuple(columns))

    def gather(self, name, index, columns):
        cp = _cupy()
        record = self.tables[name]
        positions = record["index"].get_indexer(index)
        if (positions < 0).any():
            raise ValueError("Phase 59 entity gather contains unknown IDs")
        lease = self.lease(name, columns)
        if np.array_equal(positions, np.arange(len(record["index"]))):
            return lease.arrays()
        device_positions = cp.asarray(positions)
        return tuple(array[device_positions] for array in lease.arrays())

    def summary(self):
        return {"contract":"phase59-keyed-entity-columns-v1", "events":self.events,
                "device_bytes":sum(a.nbytes for t in self.tables.values() for a in t["device"].values()),
                "host_snapshot_bytes":sum(a.nbytes for t in self.tables.values() for a in t["host"].values()),
                "tables":{name:{"rows":len(t["index"]), "generation":t["generation"],
                                "columns":list(t["device"])} for name,t in self.tables.items()}}
