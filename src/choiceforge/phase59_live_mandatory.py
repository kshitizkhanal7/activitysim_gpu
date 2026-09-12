"""Live mandatory scheduling: no captured population, draws, caches or answers.

The specialized timetable primitive vocabulary is checked against the live
model specification. Numerical boundary cases require live CPU adjudication.
Exact-output qualification remains independent, outside this evaluator.
"""
import ast
import time
import numpy as np
from .cuda_backend import _cupy
from .activitysim_scheduling import lower_activitysim_spec
from .gpu_scheduling_pipeline import GpuSchedulingPreparer, mode_logsum_slots
from .gpu_scheduling_integration import (IntegratedGpuMandatoryScheduler,
    IntegratedBatchTelemetry, assemble_device_logsum_cache)
from .scheduling_compiler import CompiledCudaSchedulingModel

STATEFUL = (
    "tt.previous_tour_ends(df.person_id, df.start)",
    "tt.previous_tour_begins(df.person_id, df.end)",
    "(df.tour_count>1) * (df.tour_num == 1) * _adjacent_window_before",
    "(df.tour_count>1) * (df.tour_num == 1) * _adjacent_window_after",
    "(df.tour_num > 1) * _adjacent_window_before",
    "(df.tour_num > 1) * _adjacent_window_after",
    "((df.tour_count>1) & (df.tour_num == 1)) * 1.0 / tt.remaining_periods_available(df.person_id, df.start, df.end)",
)
ASSIGNMENTS = (
    ("_adjacent_window_before", "tt.adjacent_window_before(df.person_id, df.start)>0"),
    ("_adjacent_window_after", "tt.adjacent_window_after(df.person_id, df.end)>0"),
)


def syntax(expression):
    return ast.dump(ast.parse(expression.strip(), mode="eval"), include_attributes=False)


def validate_lowering(lowered):
    if tuple(map(syntax, lowered.stateful_expressions)) != tuple(map(syntax, STATEFUL)):
        raise ValueError("Phase 59 live mandatory timetable expressions unsupported")
    if tuple((n, syntax(e)) for n, e in lowered.assignments) != tuple((n, syntax(e)) for n, e in ASSIGNMENTS):
        raise ValueError("Phase 59 live mandatory timetable assignments unsupported")
    if (lowered.row_columns != ("mode_choice_logsum", *(f"stateful_{i}" for i in range(7)))
            or lowered.alternative_columns != ("start", "end", "duration")):
        raise ValueError("Phase 59 live mandatory row/alternative layout unsupported")


def validate_feasible_cache(pending, prepared, alternative_slots):
    cp = _cupy()
    owners, slots = prepared.row_owners, alternative_slots[prepared.alternative_ids]
    if not bool(cp.all(pending.present[owners, slots] & cp.isfinite(pending.cache[owners, slots]))):
        raise ValueError("Phase 59 feasible TDD has missing or nonfinite live logsum")


class LiveMandatoryScheduler(IntegratedGpuMandatoryScheduler):
    def __init__(self):
        # Deliberately do not call the artifact-reading parent constructor.
        self.batches, self.telemetry, self.selected_batches = [], [], []
        self.cursor = 0
        self.pending = None
        self.preparer = None
        self.boundary_map_entries = 0
        self.boundary_tolerance = 2e-6
        self.finished = False
        self.models = {}
        self.batch_context = None
        self.live_logsum_inputs = None
        self.boundary_logsum_events = []

    @property
    def complete(self):
        return self.finished and self.pending is None and self.cursor == len(self.batches)

    def begin_batch(self, tours, alts, timetable, window_id_col, trace_label):
        cp = _cupy()
        if self.finished or self.pending is not None or self.cursor != len(self.batches):
            raise ValueError("Phase 59 mandatory batch lifecycle violation")
        if window_id_col != "person_id" or not tours.index.is_unique:
            raise ValueError("Phase 59 mandatory scheduling requires unique person-owned tours")
        if not np.array_equal(alts.index, np.arange(len(alts))):
            raise ValueError("Phase 59 TDD labels must be contiguous zero-based positions")
        alternative_values = alts[["start", "end", "duration"]].to_numpy(dtype=np.float32)
        if self.preparer is None:
            self.person_ids = timetable.windows_df.index.to_numpy().copy()
            self.alternative_values_host = alternative_values.copy()
            self.alternative_values = cp.asarray(alternative_values)
            self.alternative_slots = cp.asarray(mode_logsum_slots(alternative_values, np.arange(len(alts))))
            self.preparer = GpuSchedulingPreparer(len(self.person_ids), self.alternative_values)
        if not np.array_equal(alternative_values, self.alternative_values_host):
            raise ValueError("Phase 59 TDD alternatives changed mid-scheduling")
        if not np.array_equal(timetable.windows_df.index, self.person_ids):
            raise ValueError("Phase 59 person timetable identity changed mid-scheduling")
        people = timetable.windows_df.index.get_indexer(tours[window_id_col]).astype(np.int32)
        if (people < 0).any() or len(np.unique(people)) != len(people):
            raise ValueError("Phase 59 mandatory batch requires one tour per known person")
        # The live upstream timetable is authoritative at each batch boundary.
        windows = np.ascontiguousarray(timetable.windows, dtype=np.int8)
        if windows.shape != self.preparer.windows.shape:
            raise ValueError("Phase 59 live timetable shape differs from TDD periods")
        self.preparer.windows.set(windows)
        self.batches.append({"host":{"chooser_ids":tours.index.to_numpy().copy()},
            "device":{"person_rows":cp.asarray(people)}, "meta":{"trace_label":trace_label}})
        self.batch_context = (alts, timetable, window_id_col, trace_label)

    def bind_spec(self, choosers, alternatives, spec):
        lowered = lower_activitysim_spec(spec, choosers, alternatives)
        validate_lowering(lowered)
        key = (lowered.expressions, lowered.coefficients, lowered.schema)
        if key not in self.models:
            self.models[key] = CompiledCudaSchedulingModel(lowered.expressions,
                np.asarray(lowered.coefficients, dtype=np.float32), lowered.schema,
                overflow_protection=False, chooser_float64=True,
                dot_policy="sharrow65_lane4", exp_policy="libdevice_f32")
        batch = self.batches[self.cursor]
        batch["model"] = self.models[key]
        batch["meta"]["chooser_columns"] = list(lowered.chooser_columns)
        return lowered

    def accept_device_logsums(self, values, metadata):
        if self.pending is not None or self.cursor >= len(self.batches):
            raise ValueError("Phase 59 logsum handoff outside its live batch")
        self.pending = assemble_device_logsum_cache(values, metadata,
            self.batches[self.cursor]["host"]["chooser_ids"])

    def choose(self, live_chooser_ids, live_draws, live_chooser_values, boundary_resolver=None,
               *, return_device=False):
        cp = _cupy()
        started = time.perf_counter()
        batch = self.batches[self.cursor]
        if self.pending is None or not np.array_equal(live_chooser_ids, batch["host"]["chooser_ids"]):
            raise ValueError("Phase 59 live choice/cache identities differ")
        draws = cp.asarray(live_draws, dtype=cp.float64)
        values = cp.asarray(live_chooser_values, dtype=cp.float64)
        if draws.shape != (len(values),) or bool(cp.any(~cp.isfinite(draws) | (draws < 0) | (draws >= 1))):
            raise ValueError("Phase 59 invalid live mandatory draw")
        people = batch["device"]["person_rows"]
        prepared = self.preparer.prepare(people, values, self.pending.cache,
                                        **self._columns(batch["meta"]))
        validate_feasible_cache(self.pending, prepared, self.alternative_slots)
        # Use live end_previous, not a stored answer-derived previous-TDD vector.
        result = batch["model"].choose(values, prepared.row_values, self.alternative_values,
            prepared.alternative_ids, prepared.offsets, draws, return_device=True)
        selected = prepared.alternative_ids[prepared.offsets[:-1]+result.choices]
        risk = cp.flatnonzero(result.boundary_distances <= self.boundary_tolerance)
        downloaded = 0
        if len(risk):
            if boundary_resolver is None:
                raise ValueError("Phase 59 requires a live numerical boundary resolver")
            positions, cache = cp.asnumpy(risk), cp.asnumpy(self.pending.raw_cache[risk])
            resolved = np.asarray(boundary_resolver(positions, cache), dtype=np.int16)
            if resolved.shape != positions.shape or (resolved < 0).any() or (resolved >= len(self.alternative_values_host)).any():
                raise ValueError("Phase 59 live boundary resolver returned invalid TDDs")
            selected[risk] = cp.asarray(resolved)
            downloaded = positions.nbytes+cache.nbytes
        self.preparer.assign(people, selected)
        cp.cuda.Stream.null.synchronize()
        self.telemetry.append(IntegratedBatchTelemetry(self.cursor, batch["meta"]["trace_label"],
            len(values), self.pending.source_rows, self.pending.cache_build_ms,
            (time.perf_counter()-started)*1000, 0, 0., 0, 0, 0, len(risk), downloaded))
        self.selected_batches.append(selected)
        self.cursor += 1
        self.pending = None
        return selected if return_device else cp.asnumpy(selected)

    def finish(self):
        if self.pending is not None or self.cursor != len(self.batches) or not self.batches:
            raise ValueError("Phase 59 cannot finish an incomplete mandatory schedule")
        self.finished = True
