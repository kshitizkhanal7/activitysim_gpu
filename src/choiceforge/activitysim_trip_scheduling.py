"""Persistent CUDA probability/choice engine for ActivitySim trip scheduling.

ActivitySim retains tour-chain ordering, its random-number ledger, failure
cohorts, and final table publication.  This module removes the repeated wide
pandas probability materialization from every trip-number/iteration call.  A
normalized probability specification and reusable workspace stay resident on
CUDA; only compact row selectors, bounds, random draws, and final choices cross
the boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np
import pandas as pd

from .cuda_backend import _cupy


_ORIGINAL_MAKE_SCHEDULING_CHOICES = None
_SERVICE = None


_CUDA_SOURCE = r'''
extern "C" __global__ void trip_schedule_choice(
 const double* spec_probs, const int* spec_rows,
 const int* earliest, const int* latest, const double* draws,
 int n_rows, int n_alts, int depart_alt_base, int normalize_clipped,
 int* choices)
{
 const int row = blockIdx.x * blockDim.x + threadIdx.x;
 if (row >= n_rows) return;
 const double* probabilities = spec_probs + (long long)spec_rows[row] * n_alts;
 double clipped_sum = 0.0;
 for (int alt=0; alt<n_alts; ++alt) {
   const int depart = depart_alt_base + alt;
   if (depart >= earliest[row] && depart <= latest[row]) {
     clipped_sum += probabilities[alt];
   }
 }
 double fail_probability;
 if (normalize_clipped) {
   fail_probability = clipped_sum > 0.0 ? 0.0 : 1.0;
 } else {
   const double bounded = clipped_sum < 0.0 ? 0.0 :
                          (clipped_sum > 1.0 ? 1.0 : clipped_sum);
   fail_probability = 1.0 - bounded;
 }
 double z = draws[row];
 double max_probability = -1.0;
 int max_choice = n_alts;
 for (int alt=0; alt<n_alts; ++alt) {
   const int depart = depart_alt_base + alt;
   double probability =
     (depart >= earliest[row] && depart <= latest[row]) ? probabilities[alt] : 0.0;
   if (normalize_clipped && clipped_sum > 0.0) probability /= clipped_sum;
   if (probability > max_probability) {
     max_probability = probability;
     max_choice = alt;
   }
   z -= probability;
   if (z <= 0.0) { choices[row] = alt; return; }
 }
 if (fail_probability > max_probability) max_choice = n_alts;
 z -= fail_probability;
 choices[row] = z <= 0.0 ? n_alts : max_choice;
}
'''


@dataclass
class TripSchedulingTelemetry:
    candidate_used: bool = False
    fallback_calls: int = 0
    calls: int = 0
    chooser_rows: int = 0
    failed_choices: int = 0
    first_trip_calls: int = 0
    specification_rows: int = 0
    alternatives: int = 0
    resident_specification_bytes: int = 0
    peak_workspace_rows: int = 0
    peak_workspace_bytes: int = 0
    host_index_seconds: float = 0.0
    random_ledger_seconds: float = 0.0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    kernel_seconds: float = 0.0
    total_service_seconds: float = 0.0


class TripSchedulingDeviceService:
    def __init__(self, probs_spec: pd.DataFrame, probs_join_cols: list[str]):
        cp = _cupy()
        self.cp = cp
        self.join_columns = tuple(probs_join_cols)
        self.probability_columns = tuple(
            column for column in probs_spec.columns if column not in self.join_columns
        )
        if not self.probability_columns:
            raise ValueError("trip scheduling probability specification is empty")
        spec_index = pd.MultiIndex.from_frame(probs_spec[list(self.join_columns)])
        if not spec_index.is_unique:
            raise ValueError("trip scheduling probability keys are not unique")
        self.spec_index = spec_index
        values = probs_spec[list(self.probability_columns)].to_numpy(
            dtype=np.float64, copy=True
        )
        row_sums = values.sum(axis=1)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise ValueError("trip scheduling specification has invalid probability rows")
        values /= row_sums[:, None]
        self.spec_probabilities = cp.ascontiguousarray(cp.asarray(values))
        self.kernel = cp.RawKernel(_CUDA_SOURCE, "trip_schedule_choice")
        self.capacity = 0
        self.spec_rows = None
        self.earliest = None
        self.latest = None
        self.draws = None
        self.choices = None
        self.telemetry = TripSchedulingTelemetry(
            candidate_used=True,
            specification_rows=int(values.shape[0]),
            alternatives=int(values.shape[1]),
            resident_specification_bytes=int(self.spec_probabilities.nbytes),
        )

    def _reserve(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        cp = self.cp
        capacity = max(rows, max(1024, self.capacity * 2))
        self.spec_rows = cp.empty(capacity, dtype=cp.int32)
        self.earliest = cp.empty(capacity, dtype=cp.int32)
        self.latest = cp.empty(capacity, dtype=cp.int32)
        self.draws = cp.empty(capacity, dtype=cp.float64)
        self.choices = cp.empty(capacity, dtype=cp.int32)
        self.capacity = capacity
        workspace_bytes = int(
            self.spec_rows.nbytes
            + self.earliest.nbytes
            + self.latest.nbytes
            + self.draws.nbytes
            + self.choices.nbytes
        )
        self.telemetry.peak_workspace_rows = capacity
        self.telemetry.peak_workspace_bytes = workspace_bytes

    def choose(
        self,
        state: Any,
        choosers_df: pd.DataFrame,
        *,
        depart_alt_base: int,
        first_trip_in_leg: bool,
    ) -> tuple[pd.Series, pd.Series, np.ndarray]:
        started = time.perf_counter()
        rows = len(choosers_df)
        self._reserve(rows)

        indexed = time.perf_counter()
        chooser_index = pd.MultiIndex.from_frame(
            choosers_df[list(self.join_columns)]
        )
        spec_rows = self.spec_index.get_indexer(chooser_index).astype(
            np.int32, copy=False
        )
        if np.any(spec_rows < 0):
            bad = np.flatnonzero(spec_rows < 0)[:5]
            raise ValueError(
                "trip scheduling probability lookup failed for chooser rows "
                f"{bad.tolist()}"
            )
        earliest = np.ascontiguousarray(choosers_df.earliest.to_numpy(), dtype=np.int32)
        latest = np.ascontiguousarray(choosers_df.latest.to_numpy(), dtype=np.int32)
        self.telemetry.host_index_seconds += time.perf_counter() - indexed

        randomized = time.perf_counter()
        random_values = np.ascontiguousarray(
            state.get_rn_generator().random_for_df(choosers_df), dtype=np.float64
        ).reshape(-1)
        self.telemetry.random_ledger_seconds += time.perf_counter() - randomized

        self.spec_rows[:rows].set(spec_rows)
        self.earliest[:rows].set(earliest)
        self.latest[:rows].set(latest)
        self.draws[:rows].set(random_values)
        self.telemetry.host_to_device_bytes += int(
            spec_rows.nbytes + earliest.nbytes + latest.nbytes + random_values.nbytes
        )

        start_event = self.cp.cuda.Event()
        end_event = self.cp.cuda.Event()
        start_event.record()
        threads = 256
        self.kernel(
            ((rows + threads - 1) // threads,),
            (threads,),
            (
                self.spec_probabilities,
                self.spec_rows,
                self.earliest,
                self.latest,
                self.draws,
                np.int32(rows),
                np.int32(len(self.probability_columns)),
                np.int32(depart_alt_base),
                np.int32(bool(first_trip_in_leg)),
                self.choices,
            ),
        )
        end_event.record()
        raw_choices = self.cp.asnumpy(self.choices[:rows])
        end_event.synchronize()
        self.telemetry.kernel_seconds += float(
            self.cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
        )
        self.telemetry.device_to_host_bytes += int(raw_choices.nbytes)
        self.telemetry.calls += 1
        self.telemetry.chooser_rows += rows
        self.telemetry.first_trip_calls += int(first_trip_in_leg)
        self.telemetry.total_service_seconds += time.perf_counter() - started

        fail_index = len(self.probability_columns)
        failed = raw_choices == fail_index
        self.telemetry.failed_choices += int(np.count_nonzero(failed))
        choice_series = pd.Series(raw_choices, index=choosers_df.index)
        random_series = pd.Series(random_values, index=choosers_df.index)
        return choice_series, random_series, failed


def install_activitysim_trip_scheduling_candidate() -> None:
    global _ORIGINAL_MAKE_SCHEDULING_CHOICES
    from activitysim.abm.models.util import probabilistic_scheduling as ps

    if _ORIGINAL_MAKE_SCHEDULING_CHOICES is not None:
        return
    _ORIGINAL_MAKE_SCHEDULING_CHOICES = ps.make_scheduling_choices
    ps.make_scheduling_choices = _make_scheduling_choices_cuda


def _make_scheduling_choices_cuda(
    state,
    choosers_df,
    scheduling_mode,
    probs_spec,
    probs_join_cols,
    depart_alt_base,
    first_trip_in_leg,
    report_failed_trips,
    trace_label,
    trace_choice_col_name="depart",
    clip_earliest_latest=True,
    *,
    chunk_sizer,
):
    global _SERVICE
    from activitysim.abm.models.util import probabilistic_scheduling as ps

    if scheduling_mode != "departure" or not clip_earliest_latest:
        raise RuntimeError(
            "Phase 35 CUDA trip scheduling supports clipped departure mode only"
        )
    if state.settings.trace_hh_id:
        raise RuntimeError("Phase 35 timed candidate does not support trace_hh_id")
    if _SERVICE is None:
        _SERVICE = TripSchedulingDeviceService(probs_spec, list(probs_join_cols))
    choices, rands, failed = _SERVICE.choose(
        state,
        choosers_df,
        depart_alt_base=depart_alt_base,
        first_trip_in_leg=first_trip_in_leg,
    )
    fail_index = len(_SERVICE.probability_columns)
    if report_failed_trips and np.any(failed):
        joined = pd.merge(
            choosers_df.reset_index(),
            probs_spec,
            on=probs_join_cols,
            how="left",
        ).set_index(choosers_df.index.name)
        ps._report_bad_choices(
            state,
            bad_row_map=failed,
            df=joined,
            filename="failed_choosers",
            trace_label=trace_label,
            trace_choosers=None,
        )
    choices = (choices + int(depart_alt_base)).where(choices != fail_index, -1)
    if np.any(failed):
        choices = choices[~failed]
    assert (choices >= choosers_df.earliest[~failed]).all()
    assert (choices <= choosers_df.latest[~failed]).all()
    return choices


def trip_scheduling_telemetry() -> dict[str, Any]:
    if _SERVICE is None:
        return asdict(TripSchedulingTelemetry())
    return asdict(_SERVICE.telemetry)
