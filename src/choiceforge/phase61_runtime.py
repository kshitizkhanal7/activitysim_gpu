"""Scoped Phase 61 consumers; every hook is restored after the model step."""
from contextlib import contextmanager
import time
from pathlib import Path
import numpy as np
from .phase61_inputs import SharedInputs, direct_lookup, categorical_labels, chain_groups
from .phase61_timetable import Availability
from .phase58_trip_runtime import TripRuntime


class Runtime:
    def __init__(self, features="skims,timetable,tour_modes,entities,normals,uniforms,labels,packing", *, timetable_backend="cpu", capture_directory=None):
        self.features = set(features.split(","))
        if not self.features or not self.features <= {"skims","timetable","tour_modes","entities","normals","uniforms","labels","packing"}:
            raise ValueError("Phase 61 unsupported feature selection")
        self.shared = SharedInputs()
        self.availability = Availability(backend=timetable_backend)
        self.tour_modes = TripRuntime()
        self.tour_modes.mode_component = "tour_mode_choice"
        self.events = []
        self.skim_calls = 0
        self.normal_events = []
        self.uniforms = TripRuntime()
        self.capture_directory = Path(capture_directory) if capture_directory else None
        self.capture_counts = {}
        if self.capture_directory is not None:
            self.capture_directory.mkdir(parents=True, exist_ok=False)
            self.tour_modes.mode_capture_directory = self.capture_directory / "tour_modes"
            self.availability.capture = lambda *a:self.capture("timetable",dict(zip(
                ("windows","footprints","packed","masks","owners","alternatives"),a)))

    def capture(self, kind, arrays):
        if self.capture_directory is None:
            return
        directory = self.capture_directory / kind
        directory.mkdir(exist_ok=True)
        number = self.capture_counts.get(kind,0)
        np.savez(directory / f"batch-{number:03d}.npz", **arrays)
        self.capture_counts[kind] = number + 1

    @contextmanager
    def for_step(self, state, step):
        patches = []
        def patch(obj,name,value):
            patches.append((obj,name,getattr(obj,name)))
            setattr(obj,name,value)
        started = time.perf_counter()
        completed = False
        try:
            if "packing" in self.features and step == "trip_scheduling":
                from . import phase58_schedule_chain
                original_groups = phase58_schedule_chain.chain_groups
                patch(phase58_schedule_chain,"chain_groups",lambda frame:chain_groups(original_groups,frame))
            if "uniforms" in self.features:
                from activitysim.core.random import Random
                original_uniform = Random.random_for_df
                self.uniforms.step = step
                self.uniforms.epoch += 1
                def uniform(rng,frame,n=1):
                    if rng is not state.get_rn_generator() or not len(frame) or not frame.index.is_unique:
                        return original_uniform(rng,frame,n)
                    return self.uniforms.uniform(state,frame,n)
                patch(Random,"random_for_df",uniform)
            if "labels" in self.features and step == "summarize":
                from . import phase60_preparation
                original_labels = phase60_preparation.bin_labels
                patch(phase60_preparation,"bin_labels",lambda original,bins,fmt:
                      categorical_labels(lambda b,f:original_labels(original,b,f),bins,fmt))
            if "normals" in self.features:
                from activitysim.core.random import SimpleChannel
                from .phase61_normals import normal
                original_normal = SimpleChannel.normal_for_df
                capture = (lambda seeds,offsets,values,ids,step:self.capture("normals",
                    dict(seeds=seeds,offsets=offsets,values=values,ids=ids,step=step))) if self.capture_directory else None
                patch(SimpleChannel,"normal_for_df",lambda *a,**k:normal(original_normal,self.normal_events,*a,capture=capture,**k))
            if "skims" in self.features:
                from activitysim.core.skim_dataset import DatasetWrapper
                original = DatasetWrapper.lookup
                def lookup(wrapper,key,reverse=False):
                    self.skim_calls += 1
                    return direct_lookup(original,wrapper,key,reverse)
                patch(DatasetWrapper,"lookup",lookup)
            if "timetable" in self.features and step.endswith("tour_scheduling"):
                from activitysim.core.timetable import TimeTable
                patch(TimeTable,"tour_available",lambda table,ids,alts:self.availability(table,ids,alts))
            if "entities" in self.features and step in {"tour_mode_choice_simulate","trip_mode_choice"}:
                from . import activitysim_mode_choice
                table = "trips" if step == "trip_mode_choice" else "tours"
                self.shared.publish(state,table)
                original_inputs = activitysim_mode_choice._strict_inputs
                def inputs(state,spec,frame,locals_d):
                    result = original_inputs(state,spec,frame,locals_d)
                    self.shared.bind(table,frame,result[1],result[0])
                    return result
                patch(activitysim_mode_choice,"_strict_inputs",inputs)
            if "tour_modes" in self.features and step == "tour_mode_choice_simulate":
                from activitysim.abm.models.util import mode
                from .phase58_mode_reduction import mode_choice_simulate
                self.tour_modes.step = step
                self.tour_modes.epoch += 1
                patch(mode,"mode_choice_simulate",lambda *a,**k:mode_choice_simulate(self.tour_modes,*a,**k))
            yield
            completed = True
        finally:
            for obj,name,value in reversed(patches):
                setattr(obj,name,value)
            if completed and "entities" in self.features and step in {"trip_destination","trip_scheduling"}:
                self.shared.publish(state,"trips")
            self.events.append({"step":step,"seconds":time.perf_counter()-started,"saved_answers_read":False})

    def summary(self):
        return {"enabled":True,"features":sorted(self.features),"events":self.events,
                "diagnostic_capture":self.capture_directory is not None,"capture_counts":self.capture_counts,
                "skim_calls":self.skim_calls,"availability_events":self.availability.events,
                "timetable_entity_store":self.availability.store.summary(),
                "shared_input_events":self.shared.events,"entity_store":self.shared.store.summary(),
                "tour_mode_events":self.tour_modes.mode_events,"tour_rng_events":self.tour_modes.events,
                "normal_events":self.normal_events,"uniform_events":self.uniforms.events}
