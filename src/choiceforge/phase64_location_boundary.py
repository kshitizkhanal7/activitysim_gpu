"""Recompute borderline destination choices from live CPU inputs, not GPU logsums.

One producer/consumer context per sequential segment. Keep the already-used
normal values on device until final choice; download only guarded owners.
Never read a modeled-answer file and never draw a second random number.
"""
import inspect
import time

import numpy as np
import pandas as pd

from .phase59_boundary_logsums import BorrowedNormals


class LocationBoundary:
    def __init__(self):
        from activitysim.abm.models.util import logsums
        from activitysim.core import simulate
        from .destination_input_supergraph import DestinationInputSupergraph
        from .modelwide_final import device_compact_interaction_sample_simulate
        self.cpu_logsums=logsums.compute_location_choice_logsums
        self.cpu_simple=simulate.simple_simulate_logsums
        self.compute_original=DestinationInputSupergraph.compute
        self.final_original=device_compact_interaction_sample_simulate
        self.compute_signature=inspect.signature(self.compute_original)
        self.final_signature=inspect.signature(self.final_original)
        self.context=None
        self.sample_context=None
        self.events=[]

    def sample(self, context):
        if self.sample_context is not None:
            raise ValueError("Previous destination sample context was not consumed")
        self.sample_context=context

    def compute(self, producer, *args, **kwargs):
        if self.context is not None:
            raise ValueError("Previous destination boundary context was not consumed")
        if self.sample_context is None:
            raise ValueError("Destination boundary requires the live sampling context")
        bound=self.compute_signature.bind(producer,*args,**kwargs)
        bound.apply_defaults()
        context=dict(bound.arguments)
        service=producer.sample_service
        original=service.normal_for_df_device
        sentinel=object()
        prior=vars(service).get("normal_for_df_device",sentinel)
        captured=[]
        def normal(state, frame, size):
            result=original(state,frame,size)
            if size!=6 or not frame.index.is_unique or result.shape!=(len(frame),6):
                raise ValueError("Destination boundary needs six unique-owner normals")
            captured.append((frame.index.copy(),result.copy()))
            return result
        service.normal_for_df_device=normal
        try:
            result=self.compute_original(producer,*args,**kwargs)
        finally:
            # The method normally lives on the class; do not retain bound aliases.
            if prior is sentinel:
                del service.normal_for_df_device
            else:
                service.normal_for_df_device=prior
        if len(captured)!=1 or context["owner_choosers"] is None:
            raise ValueError("Missing compact live destination boundary inputs")
        context["normal_ids"],context["normal_values"]=captured[0]
        self.context=context
        return result

    def final(self,*args,**kwargs):
        from .modelwide_final import _cpu_reference_utility
        from activitysim.core import simulate
        from activitysim.core import interaction_simulate, logit
        bound=self.final_signature.bind(*args,**kwargs)
        bound.apply_defaults()
        values=bound.arguments
        context=self.context
        if context is None or context["state"] is not values["state"]:
            raise ValueError("Final choice has no matching live boundary producer")
        choosers=values["choosers"]
        alternatives=values["alternatives"]
        if not alternatives.index.equals(context["choosers"].index):
            raise ValueError("Destination boundary producer/consumer order changed")
        service=values["service"]
        event=dict(trace_label=str(values["trace_label"]),owners=len(choosers),guarded_owners=0,
                   normal_values_downloaded=0,utility_cells_recomputed=0,seconds=0.,rng_unchanged=True,
                   retained_normal_bytes=int(context["normal_values"].nbytes),sample_probability_cells_recomputed=0,
                   saved_answers_read=False,complete=False)
        def recompute(positions,width):
            started=time.perf_counter()
            ids=choosers.index.take(positions)
            sample=alternatives.loc[ids].copy()
            sampling=self.sample_context
            if sampling is not None:
                exact_choosers=sampling["choosers"].loc[ids]
                exact,_=interaction_simulate.eval_interaction_utilities(
                    state=values["state"],spec=sampling["spec"],df=exact_choosers,
                    locals_d=sampling["locals_d"],trace_label=str(sampling["trace_label"])+".phase64_sampling_boundary",
                    trace_rows=None,estimator=None,log_alt_losers=False,extra_data=sampling["alternatives"],
                    zone_layer=sampling["zone_layer"],compute_settings=sampling["compute_settings"])
                sampling_values=np.asarray(exact.utility,dtype=np.float32).reshape(len(ids),len(sampling["alternatives"]))
                probabilities=logit.utils_to_probs(values["state"],pd.DataFrame(sampling_values,index=ids),
                    allow_zero_probs=False,overflow_protection=True,
                    trace_label=str(sampling["trace_label"])+".phase64_sampling_boundary",trace_choosers=exact_choosers).to_numpy()
                alt_positions=sampling["alternatives"].index.get_indexer(sample[values["choice_column"]])
                owner_positions=ids.get_indexer(sample.index)
                if (alt_positions<0).any() or (owner_positions<0).any():
                    raise ValueError("CPU sample-correction identity changed")
                sample["prob"]=probabilities[owner_positions,alt_positions]
                event["sample_probability_cells_recomputed"]=int(probabilities.size)
                if not np.isfinite(sample.prob).all() or (sample.prob<=0).any():
                    raise ValueError("Invalid CPU sample-correction probability")
            owners=context["owner_choosers"]
            columns=[c for c in owners if c not in sample]
            frame=sample.join(owners[columns],how="left")
            normal_positions=context["normal_ids"].get_indexer(ids)
            if (normal_positions<0).any():
                raise ValueError("Boundary normal owner missing")
            normals=BorrowedNormals(pd.DataFrame(service.cp.asnumpy(
                context["normal_values"][service.cp.asarray(normal_positions)]),index=ids))
            state=values["state"]
            rng=state.get_rn_generator()
            channel=rng.get_channel_for_df(frame)
            ledger=channel.row_states.loc[ids].copy()
            sentinel=object()
            prior_normal=vars(rng).get("normal_for_df",sentinel)
            prior_simple=simulate.simple_simulate_logsums
            rng.normal_for_df=normals
            simulate.simple_simulate_logsums=self.cpu_simple
            try:
                logs=self.cpu_logsums(state,frame,context["tour_purpose"],context["logsum_settings"],
                    context["model_settings"],context["network_los"],0,context["chunk_tag"],
                    str(context["trace_label"])+".phase64_live_boundary",
                    in_period_col=context["in_period_col"],out_period_col=context["out_period_col"],
                    duration_col=context["duration_col"])
            finally:
                if prior_normal is sentinel:
                    vars(rng).pop("normal_for_df",None)
                else:
                    rng.normal_for_df=prior_normal
                simulate.simple_simulate_logsums=prior_simple
            event["rng_unchanged"]=ledger.equals(channel.row_states.loc[ids])
            if normals.cursor!=6 or not event["rng_unchanged"]:
                raise ValueError("Boundary recomputation advanced RNG or changed borrowed normals")
            if not logs.index.equals(sample.index) or not np.isfinite(np.asarray(logs)).all():
                raise ValueError("CPU boundary logsums invalid or reordered")
            sample["mode_choice_logsum"]=logs
            counts=sample.groupby(level=0,sort=False).size().to_numpy()
            raw=_cpu_reference_utility(state,choosers.loc[ids],sample,counts,values["spec"],values["skims"],
                values["locals_d"] or {},str(values["trace_label"])+".phase64_live_boundary",values["compute_settings"])
            if raw.shape!=(len(sample),) or not np.isfinite(raw).all() or np.any(counts>width):
                raise ValueError("CPU boundary utility layout invalid")
            # Keep the ORIGINAL segment width: subset padding changes summation.
            padded=np.full((len(ids),width),-999.,dtype=np.float32)
            cursor=0
            for i,count in enumerate(counts):
                padded[i,:count]=raw[cursor:cursor+count]
                cursor+=count
            event.update(guarded_owners=len(ids),normal_values_downloaded=len(ids)*6,
                         utility_cells_recomputed=len(raw),seconds=time.perf_counter()-started)
            return padded
        if hasattr(service,"phase64_boundary_utilities"):
            raise ValueError("Overlapping destination boundary callback")
        service.phase64_boundary_utilities=recompute
        try:
            result=self.final_original(*args,**kwargs)
            event["complete"]=True
            return result
        finally:
            del service.phase64_boundary_utilities
            self.context=None
            self.sample_context=None
            self.events.append(event)

    def summary(self):
        return dict(events=self.events,contexts_outstanding=int(self.context is not None or self.sample_context is not None),
                    saved_answers_read=False,scope="Live upstream CPU logsums and utilities for guarded destination owners; borrowed normals and unchanged RNG ledger")
