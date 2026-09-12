"""Complete live departure retries with one final publication and ledger update.

The rectangular random buffer is speculative computation, not speculative RNG
consumption. Only attempts actually used advance the authoritative row ledger.
Successful legs are absent on later retries, including from inbound constraints.
"""
import time
import numpy as np
import pandas as pd
from .phase58_schedule_chain import ChainSchedulingService, SOURCE as CHAIN_SOURCE

SOURCE = CHAIN_SOURCE.split('extern "C"')[0] + r'''
extern "C" __global__ void retries(const int* ptr,int groups,const int* outbound,
 const int* num,const int* count,const int* fixed,const int* hour,const int* lo,
 const int* hi,const int* specrow,const int* drawrow,const double* draws,
 const double* probs,int alts,int base,int firstout,int firstin,int iterations,
 int* result,int* consumed,int* stats) {
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=groups)return;
 int begin=ptr[g],end=ptr[g+1]; bool activeout=true,activein=true;
 for(int i=0;i<iterations && (activeout||activein);i++) {
  int maxout=-1,previous=-1, failures=0, attempts=0, rows=0;
  bool badout=false,badin=false;
  if(activeout)for(int r=begin;r<end;r++)if(outbound[r]) {
   rows++; int lower=previous>=0?previous:lo[r],v;
   if(fixed[r])v=hour[r];
   else {
    attempts++;consumed[r]++;
    v=choose(probs+(long long)specrow[r]*alts,alts,base,lower,hi[r],
      draws[(long long)drawrow[r]*iterations+i],num[r]==firstout);
    if(v<0){failures++;badout=true;if(i==iterations-1)v=lower;}
    previous=v>=0?v:lower;
   }
   result[r]=v;if(v>maxout)maxout=v;
  }
  previous=-1;
  if(activein)for(int r=end-1;r>=begin;r--)if(!outbound[r]) {
   rows++;int upper=previous>=0?previous:hi[r],lower=maxout>=0?maxout:lo[r],v;
   if(fixed[r])v=hour[r];
   else {
    attempts++;consumed[r]++;
    v=choose(probs+(long long)specrow[r]*alts,alts,base,lower,upper,
      draws[(long long)drawrow[r]*iterations+i],count[r]-num[r]==firstin);
    if(v<0){failures++;badin=true;if(i==iterations-1)v=upper;}
    previous=v>=0?v:upper;
   }
   result[r]=v;
  }
  // One update per active warp instead of four contended atomics per tour.
  // All participating lanes are at the same retry index; completed lanes have
  // left this loop and are excluded by the current active mask.
  #if __CUDA_ARCH__ >= 800
  unsigned mask=__activemask();
  int warp_rows=__reduce_add_sync(mask,rows);
  int warp_attempts=__reduce_add_sync(mask,attempts);
  int warp_failures=__reduce_add_sync(mask,failures);
  if((threadIdx.x&31)==__ffs(mask)-1) {
   atomicAdd(stats+i*4,warp_rows);atomicAdd(stats+i*4+1,warp_attempts);
   atomicAdd(stats+i*4+2,warp_failures);atomicAdd(stats+i*4+3,__popc(mask));
  }
  #else
  atomicAdd(stats+i*4,rows);atomicAdd(stats+i*4+1,attempts);
  atomicAdd(stats+i*4+2,failures);atomicAdd(stats+i*4+3,1);
  #endif
  activeout=badout;activein=badin;
 }
}
'''


class RetrySchedulingService(ChainSchedulingService):
    def __init__(self, spec, columns, runtime):
        super().__init__(spec, columns, runtime)
        self.retry_kernel = self.cp.RawKernel(SOURCE, "retries", options=("--fmad=false",))

    def run(self, state, trips, tours, settings, is_last_iteration):
        from .modelwide_service import Phase46DestinationService
        started = time.perf_counter()
        frame, active, groups, host, firstout, firstin = self.pack(trips, tours, settings)
        iterations = int(settings.MAX_ITERATIONS)
        if not 1 <= iterations <= 100:
            raise ValueError("Phase 59 supports 1..100 retry attempts")
        # Cohorts are keyed by tour_id upstream. Reject ambiguous ownership.
        if frame.groupby("tour_id", sort=False).person_id.nunique().max() != 1:
            raise ValueError("Phase 59 requires one person per tour")
        rng = state.get_rn_generator()
        if rng.step_name != self.runtime.step:
            raise ValueError("Phase 59 retry RNG outside its live step")
        channel = rng.get_channel_for_df(frame)
        if channel.step_name != rng.step_name:
            raise ValueError("Phase 59 retry channel epoch differs")
        ledger = channel.row_states.loc[active.index, ["row_seed", "offset"]].copy()
        if self.runtime.service is None:
            self.runtime.service = Phase46DestinationService()
        draws = (self.runtime.service.generate_from_seeds(ledger.row_seed, ledger.offset, iterations)
                 if len(active) else self.cp.empty(1, self.cp.float64))
        if getattr(self.runtime, "entity_store", None) is not None:
            names = ("outbound", "trip_num", "trip_count", "fixed", "tour_hour", "earliest",
                     "latest", "specrow", "drawrow")
            lease = self.runtime.entity_store.publish("trips", frame.index, dict(zip(names, host[1:])))
            device = [self.cp.asarray(host[0]), *lease.arrays()]
        else:
            device = [self.cp.asarray(a) for a in host]
        result = self.cp.full(len(frame), -1, self.cp.int32)
        consumed = self.cp.zeros(len(frame), self.cp.int32)
        stats = self.cp.zeros((iterations, 4), self.cp.int32)
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        self.retry_kernel(((groups+127)//128,), (128,),
            (device[0], np.int32(groups), *device[1:], draws, self.spec_probabilities,
             np.int32(len(self.probability_columns)), np.int32(settings.DEPART_ALT_BASE),
             np.int32(firstout), np.int32(firstin), np.int32(iterations), result, consumed, stats))
        end.record()
        choices, used, counts = map(self.cp.asnumpy, (result, consumed, stats))
        if (choices < 0).any() or (used < 0).any() or (used > iterations).any():
            raise ValueError("Phase 59 incomplete retry result; ledger not committed")
        # No other consumer may mutate offsets between snapshot and commit.
        if not channel.row_states.loc[active.index, ["row_seed", "offset"]].equals(ledger):
            raise ValueError("Phase 59 RNG ledger changed during retry execution")
        channel.row_states.loc[frame.index, "offset"] += used
        end.synchronize()
        elapsed = time.perf_counter()-started
        self.telemetry.calls += 1
        self.telemetry.chooser_rows += int(used.sum())
        self.telemetry.failed_choices += int(counts[:, 2].sum())
        self.telemetry.host_to_device_bytes += sum(a.nbytes for a in host)
        self.telemetry.device_to_host_bytes += choices.nbytes+used.nbytes+counts.nbytes
        self.telemetry.kernel_seconds += self.cp.cuda.get_elapsed_time(start, end)/1000
        self.telemetry.total_service_seconds += elapsed
        for i, (rows, attempts, failures, group_count) in enumerate(counts):
            if not rows:
                continue
            self.runtime.chain_events.append({"rows":int(rows), "choosers":int(attempts),
                "groups":int(group_count), "failures_before_final_coercion":int(failures),
                "last_iteration":i == iterations-1, "seconds":elapsed if i == 0 else 0.,
                "controller":"phase59-device-retries", "intermediate_choice_download_bytes":0})
        self.runtime.events.append({"step":self.runtime.step, "epoch":self.runtime.epoch,
            "kind":"cuda_uniform", "rows":int(used.sum()), "draws_per_row":1,
            "device_only":True, "seconds":elapsed, "speculative_draws_generated":len(active)*iterations,
            "speculative_draw_buffer_bytes":int(draws.nbytes),
            "rng_state_buffer_bytes":int(self.runtime.service._rng_states.nbytes) if len(active) else 0,
            "retry_working_device_bytes":int(sum(a.nbytes for a in device)+result.nbytes+consumed.nbytes+stats.nbytes),
            "unused_draws_consumed":0})
        return pd.Series(choices, index=frame.index)


def run_trip_scheduling(runtime, state, trips_chunk, tours, probs_spec, model_settings,
                        estimator, is_last_iteration, trace_label, *, chunk_sizer):
    from . import activitysim_trip_scheduling as adapter
    from activitysim.abm.models import trip_scheduling as upstream
    if estimator is not None or state.settings.trace_hh_id:
        raise ValueError("Phase 59 retry execution excludes estimation and household tracing")
    if adapter._SERVICE is None:
        columns = model_settings.probs_join_cols or upstream.PROBS_JOIN_COLUMNS_DEPARTURE_BASED
        adapter._SERVICE = RetrySchedulingService(probs_spec, columns, runtime)
    if not isinstance(adapter._SERVICE, RetrySchedulingService):
        raise ValueError("Phase 59 retry service ownership mismatch")
    return adapter._SERVICE.run(state, trips_chunk, tours, model_settings, is_last_iteration)
