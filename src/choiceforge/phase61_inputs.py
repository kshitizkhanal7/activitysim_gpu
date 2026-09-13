"""Checked live CPU skim access and generation-checked device entity inputs."""
import numpy as np
import pandas as pd
from .phase59_entity_store import EntityStore


def chain_groups(original, frame):
    """First-appearance pair identity without materializing Python row tuples."""
    columns = [frame[n] for n in ("person_id","tour_id")]
    if not len(frame) or any(c.dtype.kind not in "iu" or c.isna().any() for c in columns):
        return original(frame)
    a, au = pd.factorize(columns[0],sort=False)
    b, bu = pd.factorize(columns[1],sort=False)
    width = max(1,len(bu))
    if len(au) > np.iinfo(np.int64).max // width:
        return original(frame)
    codes, unique = pd.factorize(a.astype(np.int64)*width+b,sort=False)
    return codes,len(unique)


def categorical_labels(original,bins,label_format):
    """Format one representative of each observed bin, preserving observed ranks."""
    if not isinstance(bins.dtype,pd.CategoricalDtype) or not len(bins) or bins.isna().any():
        return original(bins,label_format)
    codes,_ = pd.factorize(bins.cat.codes,sort=False)
    _,first = np.unique(codes,return_index=True)
    labels = original(bins.iloc[first],label_format)
    return pd.Series(labels.to_numpy()[codes],index=bins.index,name=labels.name)


def direct_lookup(original, wrapper, key, reverse=False):
    if wrapper.df is None:
        raise ValueError("Phase 61 skim wrapper has no live target")
    fixed_time = None
    if isinstance(key,tuple):
        if len(key) != 2:
            return original(wrapper,key,reverse)
        key, label = key
        fixed_time = wrapper.time_map[label]
    array = wrapper.dataset[key]
    dims = tuple(array.dims)
    supported = {wrapper.odim,wrapper.ddim,"time_period"}
    # Encoded/dask/exotic skims stay with the original decoder and indexing path.
    if (not {wrapper.odim,wrapper.ddim} <= set(dims) or not set(dims) <= supported
            or not set(wrapper.positions) <= set(dims)
            or (fixed_time is not None and "time_period" not in dims)
            or wrapper.dataset.redirection.is_blended(key)
            or "digital_encoding" in array.attrs or not isinstance(array.data,np.ndarray)):
        return original(wrapper,(key,label) if fixed_time is not None else key,reverse)
    positions = wrapper.positions
    coordinates = []
    for dim,size in zip(dims,array.shape):
        source = wrapper.ddim if reverse and dim == wrapper.odim else wrapper.odim if reverse and dim == wrapper.ddim else dim
        values = np.asarray(fixed_time if dim == "time_period" and fixed_time is not None else positions[source])
        if values.dtype.kind not in "iu" or (values < -size).any() or (values >= size).any():
            raise ValueError("Phase 61 skim positions outside the live cube")
        coordinates.append(values)
    return pd.Series(array.data[tuple(coordinates)],index=wrapper.df.index,name=key)


class SharedInputs:
    def __init__(self):
        self.store = EntityStore()
        self.events = []
        self.source_names = {}

    def publish(self, state, table):
        frame = state.get_dataframe(table)
        columns = {n:frame[n].to_numpy() for n in frame if frame[n].dtype.kind in "biuf"}
        if columns:
            self.store.publish(table,frame.index,columns)

    def bind(self, table, frame, environment, document=None):
        record = self.store.tables.get(table)
        if record is None:
            return
        positions = record["index"].get_indexer(frame.index)
        if (positions<0).any():
            raise ValueError("Phase 61 shared input has unknown entity IDs")
        selected = []
        allowed = None
        if document is not None:
            from .sharrow_cuda import _node_sources
            key = document["sha256"]
            if key not in self.source_names:
                self.source_names[key] = {s[1] for t in document["terms"] for s in _node_sources(t["tree"])
                                          if s[0] in {"name","column"}}
            allowed = self.source_names[key]
        for name,host in record["host"].items():
            if name not in frame or frame[name].dtype.kind not in "biuf" or allowed is not None and name not in allowed:
                continue
            actual = np.ascontiguousarray(frame[name].to_numpy())
            if actual.dtype == host.dtype and np.array_equal(actual.view(np.uint8),
                                                             host[positions].view(np.uint8)):
                selected.append(name)
        if not selected:
            return
        # Only exact matching live columns are borrowed. Derived/changed columns
        # remain in the original environment rather than using a stale snapshot.
        arrays = self.store.gather(table,frame.index,selected)
        for name,array in zip(selected,arrays):
            environment["df"][name] = environment[name] = array
        self.events.append({"table":table,"generation":record["generation"],"rows":len(frame),
                            "columns":selected,"matched_host_bytes":sum(record["host"][n][positions].nbytes for n in selected)})
