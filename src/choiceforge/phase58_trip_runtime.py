"""Live keyed RNG service shared across the three trip components.

No saved choices or scenario-specific random values are consumed. Uniforms use
the previously tested CUDA MT19937 implementation. Nonzero-variance normals
remain with ActivitySim's authoritative arithmetic, explicitly counted.
"""
from contextlib import contextmanager
import time
import numpy as np
import pandas as pd
from .activitysim_trip_scheduling import TripSchedulingDeviceService


class ResidentSchedulingService(TripSchedulingDeviceService):
    """Feed live CUDA random draws directly into the scheduling kernel."""
    def __init__(self, probs_spec, probs_join_cols, runtime):
        super().__init__(probs_spec, probs_join_cols)
        self.runtime = runtime

    def choose(self, state, choosers_df, *, depart_alt_base, first_trip_in_leg):
        started = time.perf_counter()
        rows = len(choosers_df)
        self._reserve(rows)
        indexed = time.perf_counter()
        spec_rows = self.spec_index.get_indexer(
            pd.MultiIndex.from_frame(choosers_df[list(self.join_columns)])).astype(np.int32)
        if np.any(spec_rows < 0):
            raise ValueError("Phase 58 scheduling probability keys are missing")
        earliest = np.ascontiguousarray(choosers_df.earliest, dtype=np.int32)
        latest = np.ascontiguousarray(choosers_df.latest, dtype=np.int32)
        self.telemetry.host_index_seconds += time.perf_counter()-indexed
        randomized = time.perf_counter()
        draws = self.runtime.uniform(state, choosers_df, device_only=True)
        self.telemetry.random_ledger_seconds += time.perf_counter()-randomized
        self.spec_rows[:rows].set(spec_rows)
        self.earliest[:rows].set(earliest)
        self.latest[:rows].set(latest)
        self.telemetry.host_to_device_bytes += spec_rows.nbytes+earliest.nbytes+latest.nbytes
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        self.kernel(((rows+255)//256,), (256,), (
            self.spec_probabilities, self.spec_rows, self.earliest, self.latest, draws,
            np.int32(rows), np.int32(len(self.probability_columns)), np.int32(depart_alt_base),
            np.int32(bool(first_trip_in_leg)), self.choices))
        end.record()
        choices = self.cp.asnumpy(self.choices[:rows])
        end.synchronize()
        self.telemetry.kernel_seconds += self.cp.cuda.get_elapsed_time(start, end)/1000
        self.telemetry.device_to_host_bytes += choices.nbytes
        self.telemetry.calls += 1
        self.telemetry.chooser_rows += rows
        self.telemetry.first_trip_calls += int(first_trip_in_leg)
        failed = choices == len(self.probability_columns)
        self.telemetry.failed_choices += int(failed.sum())
        self.telemetry.total_service_seconds += time.perf_counter()-started
        # The existing adapter rejects tracing and never reads the rand Series.
        return pd.Series(choices, index=choosers_df.index), None, failed


class TripRuntime:
    def __init__(self, *, device_retries=False):
        self.device_retries = device_retries
        self.entity_store = None
        if device_retries:
            from .phase59_entity_store import EntityStore
            self.entity_store = EntityStore()
        self.service = None
        self.events = []
        self.mode_events = []
        self.chain_events = []
        self.expected_mode_rows = None
        self.step = None
        self.epoch = 0
        self.mode_capture_directory = None  # Explicit diagnostic runs only.

    def publish_entities(self, state):
        if self.entity_store is None or not hasattr(state, "get_dataframe"):
            return
        for table, names in {
            "persons":("household_id", "age", "ptype"),
            "tours":("person_id", "household_id", "start", "end", "destination"),
            "trips":("person_id", "tour_id", "household_id", "origin", "destination",
                      "outbound", "trip_num", "trip_count", "depart"),
        }.items():
            frame = state.get_dataframe(table)
            columns = {name:frame[name].to_numpy() for name in names if name in frame
                       and frame[name].dtype.kind in "biuf"}
            if table == "trips" and "trip_mode" in frame:
                from .nested_logit import MTC21_ALTERNATIVES
                columns["trip_mode_code"] = pd.Categorical(frame.trip_mode, categories=MTC21_ALTERNATIVES).codes
            if columns:
                self.entity_store.publish(table, frame.index, columns)

    def uniform(self, state, frame, n=1, *, device_only=False):
        started = time.perf_counter()
        if not frame.index.is_unique or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Phase 58 requires unique keyed rows and a positive draw count")
        if state.get_rn_generator().step_name != self.step:
            raise ValueError("Phase 58 cannot use a stale step epoch")
        if self.service is None:
            from .modelwide_service import Phase46DestinationService
            self.service = Phase46DestinationService()
        host, device = self.service.random_for_df(state, frame, int(n), device_only=device_only)
        self.events.append({"step": self.step, "epoch": self.epoch, "kind": "cuda_uniform",
                            "rows": len(frame), "draws_per_row": int(n),
                            "device_only": device_only, "seconds": time.perf_counter()-started})
        return device if device_only else host

    @contextmanager
    def for_step(self, state, step):
        if self.step is not None:
            raise ValueError("Phase 58 contexts cannot overlap")
        self.publish_entities(state)
        upstream_scheduling = mode_module = None
        if step == "trip_scheduling":
            from . import activitysim_trip_scheduling as scheduling_module
            if scheduling_module._SERVICE is not None:
                raise ValueError("Phase 58 scheduling service must start in a fresh process")
            from activitysim.abm.models import trip_scheduling as upstream_scheduling
            from .phase58_schedule_chain import run_trip_scheduling
            if self.device_retries:
                from .phase59_retry import run_trip_scheduling
        if step == "trip_mode_choice":
            from activitysim.abm.models import trip_mode_choice as mode_module
            from .phase58_mode_reduction import mode_choice_simulate
            if hasattr(state, "get_dataframe"):
                self.expected_mode_rows = len(state.get_dataframe("trips"))
        self.step = step
        self.epoch += 1
        rng = state.get_rn_generator()
        original_uniform = rng.random_for_df
        original_normal = rng.normal_for_df
        sentinel = object()
        previous_uniform = rng.__dict__.get("random_for_df", sentinel)
        previous_normal = rng.__dict__.get("normal_for_df", sentinel)
        previous_runtime = rng.__dict__.get("_choiceforge_phase58_runtime", sentinel)

        def uniform(frame, n=1):
            if len(frame) == 0:
                return original_uniform(frame, n)
            return self.uniform(state, frame, n)

        def normal(frame, mu=0, sigma=1, broadcast=False, size=None):
            started = time.perf_counter()
            # Preserve the entire authoritative transform for ordinary normals.
            # With zero variance and finite nonzero mu, z*0+mu == mu for every
            # finite z. Avoid generating irrelevant z, but advance the SAME ledger.
            zero_variance = (broadcast and size is None and len(frame) > 0
                             and np.all(np.asarray(sigma) == 0)
                             and np.all(np.isfinite(mu)) and np.all(np.asarray(mu) != 0))
            if zero_variance:
                if rng.step_name != self.step:
                    raise ValueError("Phase 58 normal shortcut outside its live step")
                unique = frame.index.unique().to_series()
                channel = rng.get_channel_for_df(unique)
                if channel.step_name != rng.step_name:
                    raise ValueError("Phase 58 normal channel epoch differs")
                # Use Series alignment exactly as the upstream broadcast result.
                result = pd.Series(0.0, index=frame.index) * sigma + mu
                channel.row_states.loc[unique.index, "offset"] += 1
                kind = "zero_variance_normal_identity"
            else:
                result = original_normal(frame, mu=mu, sigma=sigma, broadcast=broadcast, size=size)
                kind = "authoritative_cpu_normal"
            self.events.append({"step": step, "epoch": self.epoch, "kind": kind,
                                "rows": len(frame), "draws_per_row":1 if size is None else size,
                                "seconds": time.perf_counter()-started})
            return result

        rng.random_for_df = uniform
        rng.normal_for_df = normal
        rng._choiceforge_phase58_runtime = self
        original_chain = original_mode_simulate = None
        if step == "trip_scheduling":
            original_chain = upstream_scheduling.run_trip_scheduling
            upstream_scheduling.run_trip_scheduling = lambda *a, **k: run_trip_scheduling(self, *a, **k)
        if step == "trip_mode_choice":
            original_mode_simulate = mode_module.mode_choice_simulate
            mode_module.mode_choice_simulate = lambda *a, **k: mode_choice_simulate(self, *a, **k)
        completed = False
        try:
            yield self
            completed = True
        finally:
            if original_chain is not None:
                upstream_scheduling.run_trip_scheduling = original_chain
            if original_mode_simulate is not None:
                mode_module.mode_choice_simulate = original_mode_simulate
            for name, previous in (("random_for_df", previous_uniform),
                                   ("normal_for_df", previous_normal),
                                   ("_choiceforge_phase58_runtime", previous_runtime)):
                if previous is sentinel:
                    delattr(rng, name)
                else:
                    setattr(rng, name, previous)
            self.step = None
            if completed:
                self.publish_entities(state)

    def summary(self):
        return {"contract": "phase58-live-trip-rng-v1", "events": self.events,
                "device_retry_controller": self.device_retries,
                "entity_store": self.entity_store.summary() if self.entity_store is not None else None,
                "mode_events": self.mode_events,
                "chain_events": self.chain_events,
                "expected_mode_rows": self.expected_mode_rows,
                "epochs": self.epoch,
                "uniform_draws": sum(e["rows"]*e["draws_per_row"] for e in self.events if e["kind"] == "cuda_uniform"),
                "device_only_draws": sum(e["rows"]*e["draws_per_row"] for e in self.events
                                          if e["kind"] == "cuda_uniform" and e["device_only"]),
                "zero_variance_rows": sum(e["rows"] for e in self.events if e["kind"] == "zero_variance_normal_identity"),
                "authoritative_cpu_normal_calls": sum(e["kind"] == "authoritative_cpu_normal" for e in self.events),
                "result_replay_enabled": False,
                "scope": ("live random channels, complete device departure retries and mode utility-to-choice; CPU annotations, ordinary normals and final table publication retained"
                          if self.device_retries else "live random channels, device scheduling chains and mode utility-to-choice; CPU retry cohorts, ordinary normals and final table publication retained")}
