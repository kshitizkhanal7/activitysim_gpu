"""Live per-tour departure chains; ActivitySim retains retry/cohort ownership.

One GPU thread executes the short dependent chain for a person-tour. All tours
run in parallel. Bounds, outbound-to-inbound constraints and intermediate
choices remain on device for the iteration. No reference outputs are read.
"""
import time
import numpy as np
import pandas as pd
from .activitysim_trip_scheduling import TripSchedulingDeviceService

SOURCE = r'''
__device__ int choose(const double* p,int alts,int base,int lo,int hi,double z,int norm) {
 double total=0.; for(int a=0;a<alts;a++)if(a+base>=lo&&a+base<=hi)total+=p[a];
 double fail=norm?(total>0.?0.:1.):1.-fmin(1.,fmax(0.,total));
 double largest=-1.; int best=alts;
 for(int a=0;a<alts;a++) {
  double v=(a+base>=lo&&a+base<=hi)?p[a]:0.; if(norm&&total>0.)v/=total;
  if(v>largest){largest=v;best=a;} z-=v; if(z<=0.)return a+base;
 }
 if(fail>largest)best=alts; z-=fail;
 int a=z<=0.?alts:best; return a==alts?-1:a+base;
}
extern "C" __global__ void chains(const int* ptr,int groups,const int* outbound,
 const int* num,const int* count,const int* fixed,const int* hour,const int* lo,
 const int* hi,const int* specrow,const int* drawrow,const double* draws,
 const double* probs,int alts,int base,int firstout,int firstin,int force,
 int* result,unsigned char* failed) {
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=groups)return;
 int begin=ptr[g],end=ptr[g+1],maxout=-1,previous=-1;
 // Rows sorted by trip id within person-tour; validated increasing trip_num.
 for(int r=begin;r<end;r++)if(outbound[r]) {
  int lower=previous>=0?previous:lo[r]; int v;
  if(fixed[r])v=hour[r];
  else {
   v=choose(probs+(long long)specrow[r]*alts,alts,base,lower,hi[r],draws[drawrow[r]],num[r]==firstout);
   if(v<0){failed[r]=1;if(force)v=lower;}
   previous=v>=0?v:lower;
  }
  result[r]=v;if(v>maxout)maxout=v;
 }
 previous=-1;
 for(int r=end-1;r>=begin;r--)if(!outbound[r]) {
  int upper=previous>=0?previous:hi[r],lower=maxout>=0?maxout:lo[r]; int v;
  if(fixed[r])v=hour[r];
  else {
   v=choose(probs+(long long)specrow[r]*alts,alts,base,lower,upper,draws[drawrow[r]],count[r]-num[r]==firstin);
   if(v<0){failed[r]=1;if(force)v=upper;}
   previous=v>=0?v:upper;
  }
  result[r]=v;
 }
}
'''


class ChainSchedulingService(TripSchedulingDeviceService):
    def __init__(self, spec, columns, runtime):
        values = spec.drop(columns=columns).to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Phase 58 chain probabilities must be finite and nonnegative")
        super().__init__(spec, columns)
        self.runtime = runtime
        self.chain_kernel = self.cp.RawKernel(SOURCE, "chains", options=("--fmad=false",))

    def pack(self, trips, tours, settings):
        from activitysim.abm.models import trip_scheduling as upstream
        if (settings.scheduling_mode != "departure" or settings.preprocessor is not None
                or upstream._logic_version(settings) != 2
                or settings.FAILFIX != "choose_most_initial"):
            raise ValueError("Phase 58 chains require departure/logic-2/no-preprocessor/choose-most-initial")
        if not trips.index.is_unique or trips.empty:
            raise ValueError("Phase 58 chains require nonempty unique trip IDs")
        frame = trips.copy()
        upstream.set_tour_hour(frame, tours)
        upstream.set_stop_num(frame)
        numeric = frame[["earliest", "latest", "tour_hour", "trip_num", "trip_count"]].to_numpy()
        if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Phase 58 chain times and ordinals must be nonnegative finite integers")
        group, uniques = pd.factorize(pd.MultiIndex.from_frame(frame[["person_id", "tour_id"]]), sort=False)
        order = np.lexsort((frame.index.to_numpy(), group))
        frame = frame.iloc[order]
        group = group[order]
        ptr = np.r_[0, np.flatnonzero(group[1:] != group[:-1])+1, len(frame)].astype(np.int32)
        out = frame.outbound.to_numpy(bool)
        num, count = frame.trip_num.to_numpy(), frame.trip_count.to_numpy()
        # Reject exotic/noncontiguous cohorts rather than invent chain semantics.
        leg = pd.DataFrame({"group":group, "out":out, "num":num, "count":count})
        grouped = leg.groupby(["group", "out"], sort=False)
        checks = grouped.agg(minimum=("num","min"), maximum=("num","max"),
                             size=("num","size"), count=("count","first"), distinct=("count","nunique"))
        if not ((checks.minimum == 1) & (checks.maximum == checks["count"])
                & (checks["size"] == checks["count"]) & (checks.distinct == 1)).all():
            raise ValueError("Phase 58 chains require complete contiguous trip legs")
        previous_num = grouped["num"].shift()
        if not (leg.num[previous_num.notna()].to_numpy() == previous_num.dropna().to_numpy()+1).all():
            raise ValueError("Phase 58 trip IDs must preserve within-leg trip-number order")
        fixed = np.where(out, num == 1, num == count) | (frame.primary_purpose == "atwork").to_numpy()
        active = frame.loc[~fixed]
        firstout = int(num[out & ~fixed].min()) if (out & ~fixed).any() else -1
        firstin = int(num[~out & ~fixed].min()) if (~out & ~fixed).any() else -1
        specrow = np.full(len(frame), -1, np.int32)
        specrow[~fixed] = self.spec_index.get_indexer(pd.MultiIndex.from_frame(active[list(self.join_columns)]))
        if (specrow[~fixed] < 0).any():
            raise ValueError("Phase 58 chain probability keys missing")
        drawrow = np.full(len(frame), -1, np.int32)
        drawrow[~fixed] = np.arange(len(active), dtype=np.int32)
        arrays = [ptr, out, num, count, fixed, frame.tour_hour, frame.earliest, frame.latest, specrow, drawrow]
        host = [np.ascontiguousarray(a, dtype=np.int32) for a in arrays]
        return frame, active, len(uniques), host, firstout, firstin

    def run(self, state, trips, tours, settings, is_last_iteration):
        started = time.perf_counter()
        frame, active, groups, host, firstout, firstin = self.pack(trips, tours, settings)
        draws = self.runtime.uniform(state, active, device_only=True) if len(active) else self.cp.empty(1)
        device = [self.cp.asarray(a) for a in host]
        result = self.cp.empty(len(frame), self.cp.int32)
        failed = self.cp.zeros(len(frame), self.cp.uint8)
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        self.chain_kernel(((groups+127)//128,), (128,),
            (device[0], np.int32(groups), *device[1:], draws, self.spec_probabilities,
             np.int32(len(self.probability_columns)), np.int32(settings.DEPART_ALT_BASE),
             np.int32(firstout), np.int32(firstin), np.int32(is_last_iteration), result, failed))
        end.record()
        choices = self.cp.asnumpy(result)
        failures = int(self.cp.count_nonzero(failed).get())
        end.synchronize()
        self.telemetry.calls += 1
        self.telemetry.chooser_rows += len(active)
        self.telemetry.failed_choices += failures
        self.telemetry.host_to_device_bytes += sum(a.nbytes for a in host)
        self.telemetry.device_to_host_bytes += choices.nbytes+8
        self.telemetry.kernel_seconds += self.cp.cuda.get_elapsed_time(start,end)/1000
        self.telemetry.total_service_seconds += time.perf_counter()-started
        self.runtime.chain_events.append({"rows":len(frame), "choosers":len(active),
            "groups":groups, "failures_before_final_coercion":failures,
            "last_iteration":bool(is_last_iteration), "seconds":time.perf_counter()-started,
            "intermediate_choice_download_bytes":0})
        return pd.Series(choices[choices >= 0], index=frame.index[choices >= 0])


def run_trip_scheduling(runtime, state, trips_chunk, tours, probs_spec, model_settings,
                        estimator, is_last_iteration, trace_label, *, chunk_sizer):
    from . import activitysim_trip_scheduling as adapter
    from activitysim.abm.models import trip_scheduling as upstream
    if estimator is not None or state.settings.trace_hh_id:
        raise ValueError("Phase 58 chain execution excludes estimation and household tracing")
    if adapter._SERVICE is None:
        columns = model_settings.probs_join_cols or upstream.PROBS_JOIN_COLUMNS_DEPARTURE_BASED
        adapter._SERVICE = ChainSchedulingService(probs_spec, columns, runtime)
    if not isinstance(adapter._SERVICE, ChainSchedulingService):
        raise ValueError("Phase 58 chain service ownership mismatch")
    return adapter._SERVICE.run(state, trips_chunk, tours, model_settings, is_last_iteration)
