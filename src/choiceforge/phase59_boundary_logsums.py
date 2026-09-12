"""Live CPU boundary inputs using the already-consumed standard-normal values.

The CPU must recompute the complete logsum, not merely the final scheduling
utility using rounded GPU logsums. No reference-output artifacts are read.
"""
import numpy as np
import pandas as pd


def period_slots(rows):
    """Use categorical skim keys, not the purpose-specific representative hours.

    MTC's school EV representative starts at 18, even though an actual hour-18
    departure maps to PM. Reclassifying representative hours corrupts the key.
    """
    labels = ("EA", "AM", "MD", "PM", "EV")
    out = pd.Categorical(rows.out_period, categories=labels).codes.astype(np.int16)
    inbound = pd.Categorical(rows.in_period, categories=labels).codes.astype(np.int16)
    if (out < 0).any() or (inbound < 0).any():
        raise ValueError("Phase 59 unsupported categorical skim-period key")
    return out*5+inbound


class BorrowedNormals:
    """Match assign_variables' single six-column draw, not six scalar calls."""
    def __init__(self, values):
        if not values.index.is_unique or values.shape[1] != 6:
            raise ValueError("Phase 59 boundary requires six keyed live normals")
        self.values = values
        self.cursor = 0

    def __call__(self, frame, mu=0, sigma=1, broadcast=False, size=None):
        if (not broadcast or size != 6 or self.cursor != 0
                or not np.isscalar(mu) or mu != 0 or not np.isscalar(sigma) or sigma != 1):
            raise ValueError("Phase 59 unsupported boundary normal request")
        positions = self.values.index.get_indexer(frame.index)
        if (positions < 0).any():
            raise ValueError("Phase 59 boundary normal IDs are unknown")
        self.cursor = 6
        return pd.DataFrame(self.values.to_numpy()[positions], index=frame.index)


def recompute(context, boundary_ids, compute_logsums, cpu_simple_logsums):
    from activitysim.core import simulate
    state = context["state"]
    rows = context["rows"].loc[boundary_ids].copy()
    tours = context["tours"].loc[boundary_ids]
    rng = state.get_rn_generator()
    channel = rng.get_channel_for_df(rows)
    ledger = channel.row_states.loc[boundary_ids].copy()
    normals = BorrowedNormals(context["normals"].loc[boundary_ids])
    sentinel = object()
    prior_normal = rng.__dict__.get("normal_for_df", sentinel)
    prior_simple = simulate.simple_simulate_logsums
    rng.normal_for_df = normals
    simulate.simple_simulate_logsums = cpu_simple_logsums
    try:
        logsums = compute_logsums(state, rows, tours, context["purpose"], context["settings"],
                                 context["network_los"], context["skims"], context["trace_label"]+".live_cpu_boundary")
    finally:
        if prior_normal is sentinel:
            rng.__dict__.pop("normal_for_df", None)
        else:
            rng.normal_for_df = prior_normal
        simulate.simple_simulate_logsums = prior_simple
    if normals.cursor != 6 or not ledger.equals(channel.row_states.loc[boundary_ids]):
        raise ValueError("Phase 59 boundary must reuse exactly six normals without advancing RNG")
    if not logsums.index.equals(rows.index) or not np.isfinite(logsums.to_numpy()).all():
        raise ValueError("Phase 59 CPU boundary logsums invalid or misaligned")
    return rows, logsums.to_numpy()
