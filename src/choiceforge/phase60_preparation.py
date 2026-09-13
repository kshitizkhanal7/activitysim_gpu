"""Live, semantics-preserving preparation; no saved answers or persistent cache."""
from contextlib import contextmanager
import itertools
import time

import numpy as np
import pandas as pd


def pair_slots(start, end):
    """Lexicographic int16 pair uniqueness using a collision-free uint32 key."""
    start, end = np.asarray(start, dtype=np.int16), np.asarray(end, dtype=np.int16)
    keys = ((start.astype(np.int64)+32768) << 16) | (end.astype(np.int64)+32768)
    unique, inverse = np.unique(keys, return_inverse=True)
    pairs = np.column_stack(((unique >> 16)-32768, (unique & 65535)-32768)).astype(np.int16)
    return pairs, inverse


def slot_values(values, slots, count, label):
    values, slots = np.asarray(values), np.asarray(slots)
    if values.ndim != 1 or slots.shape != values.shape or slots.dtype.kind not in "iu":
        raise ValueError("Phase 60 invalid slot vectors")
    keys, first = np.unique(slots, return_index=True)
    if not np.array_equal(keys, np.arange(count)):
        raise ValueError(f"Phase 60 slot source {label} is incomplete or out of range")
    result = values[first]
    if not np.array_equal(result[slots], values):
        raise ValueError(f"Phase 60 slot source {label} is not slot-stable")
    return result


def period_positions(values):
    result = pd.Index(["EA", "AM", "MD", "PM", "EV"]).get_indexer(np.asarray(values).astype(str))
    if (result < 0).any():
        raise ValueError("Phase 60 unknown categorical skim period")
    return result.astype(np.int64)


def bin_labels(original, bins, label_format):
    if len(bins) == 0 or bins.isna().any():
        return original(bins, label_format)
    # Preserve the original rank computation and numeric conversion: it depends
    # on all observed intervals. Reduce only repeated label formatting.
    left = bins.apply(lambda x:x.left)
    mid = bins.apply(lambda x:x.mid)
    right = bins.apply(lambda x:x.right)
    rank = mid.map({x:sorted(mid.unique().tolist()).index(x)+1 if pd.notnull(x) else np.nan
                    for x in mid.unique()}, na_action="ignore")
    tuples = pd.MultiIndex.from_arrays([left, mid, right, rank])
    codes, unique = pd.factorize(tuples, sort=False)
    labels = [label_format.format(**{k:v for k,v in zip(("left","mid","right","rank"), row)
                                   if k in label_format}) for row in unique]
    result = pd.Series(np.asarray(labels, dtype=object)[codes], index=bins.index)
    return pd.to_numeric(result, errors="ignore")


def build_cdap_spec(state, coefficients, hhsize, trace_spec=False, trace_label=None,
                    cache=True, joint_tour_alt=False):
    """Construct original ordered slug matrix in arrays, then use original mapping."""
    from activitysim.abm.models.util import cdap
    from activitysim.core import simulate
    hhsize = min(hhsize, cdap.MAX_HHSIZE)
    if cache:
        cached = cdap.get_cached_spec(state, hhsize)
        if cached is not None:
            return cached
    alternatives = ["".join(t) for t in itertools.product("HMN", repeat=hhsize)]
    if joint_tour_alt:
        alternatives += ["".join(t)+"J" for t in itertools.product("HMN", repeat=hhsize)
                         if t.count("M")+t.count("N") >= 2]
    expressions, rows = [], []
    positions = {}

    def add(expression, columns, value, *, reuse=False):
        if reuse and expression in positions:
            indices = positions[expression]
        else:
            indices = [len(rows)]
            positions.setdefault(expression, []).append(len(rows))
            expressions.append(expression)
            rows.append(np.full(len(alternatives), np.nan, dtype=object))
        for i in indices:
            rows[i][columns] = value

    for p in range(1, hhsize+1):
        for activity in ("M", "N", "H"):
            add(cdap.add_pn(activity, p), [i for i,a in enumerate(alternatives) if a[p-1] == activity], 1)
    for row in coefficients[coefficients.cardinality <= hhsize].itertuples():
        if not row.interaction_ptypes:
            if row.slug in alternatives:
                add("1", [alternatives.index(row.slug)], row.slug)
            continue
        if not 0 <= row.cardinality <= cdap.MAX_INTERACTION_CARDINALITY:
            raise ValueError(f"Phase 60 invalid CDAP interaction cardinality: {row.cardinality}")
        for people in itertools.combinations(range(1, hhsize+1), row.cardinality):
            column = f"ptype_p{people[0]}" if row.cardinality == 1 else "_".join(f"p{p}" for p in people)
            expression = f"{column}=={row.interaction_ptypes}"
            selected = [i for i,a in enumerate(alternatives) if all(a[p-1] == row.activity for p in people)]
            add(expression, selected, row.slug, reuse=True)
    spec = pd.DataFrame(rows, columns=alternatives, index=pd.Index(expressions, name="Expression"))
    simulate.uniquify_spec_index(spec)
    if trace_spec:
        state.tracing.trace_df(spec, f"{trace_label}.hhsize{hhsize}_spec", transpose=False, slicer="NONE")
    mapping = coefficients.set_index("slug").coefficient.to_dict()
    for column in spec.columns:
        spec[column] = spec[column].map(lambda x:mapping.get(x, x or 0.)).fillna(0)
    if trace_spec:
        state.tracing.trace_df(spec, f"{trace_label}.hhsize{hhsize}_spec_patched", transpose=False, slicer="NONE")
    if cache:
        cdap.cache_spec(state, hhsize, spec)
    return spec


@contextmanager
def for_step(step, events, *, threads=48, frequency_controls=None):
    """Serial model-only hook scope; restore even on failure."""
    from . import raw_table_input_generation as raw, semantic_input_generation as semantic, activitysim_scheduling as scheduling
    patches = []
    def patch(obj, name, value):
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    started = time.perf_counter()
    previous_threads = None
    diagnostic = None
    try:
        patch(raw, "_PREPARATION_FAST", True)
        patch(semantic, "_PREPARATION_FAST", True)
        patch(scheduling, "_PREPARATION_FAST", True)
        if step == "cdap_simulate":
            from activitysim.abm.models.util import cdap
            patch(cdap, "build_cdap_spec", build_cdap_spec)
        if step == "summarize":
            from activitysim.abm.models import summarize
            original = summarize.construct_bin_labels
            patch(summarize, "construct_bin_labels", lambda bins, label_format:bin_labels(original, bins, label_format))
        if step == "non_mandatory_tour_frequency":
            import numba
            previous_threads = numba.get_num_threads()
            numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
            if frequency_controls is not None:
                from .phase60_frequency_control import measure
                diagnostic = measure(frequency_controls)
                diagnostic.__enter__()
        yield
    finally:
        if diagnostic is not None:
            diagnostic.__exit__(None, None, None)
        if previous_threads is not None:
            numba.set_num_threads(previous_threads)
        for obj, name, value in reversed(patches):
            setattr(obj, name, value)
        events.append({"step":step,"seconds":time.perf_counter()-started,
                       "frequency_threads":min(threads, numba.config.NUMBA_NUM_THREADS) if previous_threads is not None else None,
                       "saved_choices_read":False})
