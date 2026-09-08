"""Device-resident MTC nested probabilities and choice with live CPU edge checks.

The 1e-9 CDF guard is an engineering safety envelope, not a universal arithmetic
theorem. Qualified runs additionally require independent exact-output checks.
No stored choices or reference-population identities are used by this module.
"""
from functools import lru_cache
import time
import numpy as np
import pandas as pd
from .cuda_backend import _cupy
from .nested_logit import MTC21_ALTERNATIVES, mtc21_coefficients

SOURCE = r'''
__device__ double sum(const double* u, int a, int b) {
 double s=0.; for(int i=a;i<b;i++) s+=u[i]; return s;
}
__device__ double clipu(double v) { return v<=1e-300 ? 0. : v; }
__device__ double ratio(double v, double den) { return den>0. ? fmin(1.,clipu(v)/den) : 0.; }
__device__ double den(const double* u, int a, int b) {
 double s=0.; for(int i=a;i<b;i++) s+=clipu(u[i]); return s;
}
extern "C" __global__ void mode_choice(
 const float* raw, const double* draws, int rows, const double* c,
 int* choices, double* logsums, unsigned char* guards, unsigned char* bad,
 double* probabilities, int capture) {
 int row=blockDim.x*blockIdx.x+threadIdx.x; if(row>=rows)return;
 double u[31]={0.}, p[21];
 for(int a=0;a<21;a++) {
   double scale=a<6?c[0]*c[1]:(a<8?c[2]:(a<18?c[3]*c[4]:c[5]));
   double value=(double)raw[row*21+a]/scale;
   if(!isfinite(value) || value>690.) {bad[row]=1; return;}
   u[a]=exp(value);
 }
 u[21]=exp(c[1]*log(sum(u,0,2))); u[22]=exp(c[1]*log(sum(u,2,4)));
 u[23]=exp(c[1]*log(sum(u,4,6))); u[24]=exp(c[0]*log(sum(u,21,24)));
 u[25]=exp(c[2]*log(sum(u,6,8))); u[26]=exp(c[4]*log(sum(u,8,13)));
 u[27]=exp(c[4]*log(sum(u,13,18))); u[28]=exp(c[3]*log(sum(u,26,28)));
 u[29]=exp(c[5]*log(sum(u,18,21)));
 double root_raw=((u[24]+u[25])+u[28])+u[29];
 double root_den=((clipu(u[24])+clipu(u[25]))+clipu(u[28]))+clipu(u[29]);
 logsums[row]=log(exp(log(root_raw)));
 double au=ratio(u[24],root_den), nm=ratio(u[25],root_den);
 double tr=ratio(u[28],root_den), rh=ratio(u[29],root_den);
 for(int a=0;a<6;a++) {
   int begin=(a/2)*2, parent=21+a/2;
   p[a]=(au*ratio(u[parent],den(u,21,24)))*ratio(u[a],den(u,begin,begin+2));
 }
 for(int a=6;a<8;a++)p[a]=nm*ratio(u[a],den(u,6,8));
 for(int a=8;a<18;a++) {
   int begin=a<13?8:13, parent=a<13?26:27;
   p[a]=(tr*ratio(u[parent],den(u,26,28)))*ratio(u[a],den(u,begin,begin+5));
 }
 for(int a=18;a<21;a++)p[a]=rh*ratio(u[a],den(u,18,21));
 double total=0., z=draws[row], maximum=-1.; int selected=-1, maxpos=0;
 for(int a=0;a<21;a++) {
   if(capture) probabilities[row*21+a]=p[a];
   total+=p[a]; if(p[a]>maximum){maximum=p[a];maxpos=a;}
   z-=p[a]; if(fabs(z)<=1e-9)guards[row]=1;
   if(selected<0 && z<=0.)selected=a;
 }
 if(!isfinite(total)||fabs(total-1.)>1e-7||!isfinite(logsums[row]))bad[row]=1;
 choices[row]=selected<0?maxpos:selected;
}
'''


@lru_cache(maxsize=1)
def kernel():
    return _cupy().RawKernel(SOURCE, "mode_choice", options=("--fmad=false", "--std=c++11"))


def validate(nest, alternatives):
    from activitysim.core import logit
    if tuple(alternatives) != MTC21_ALTERNATIVES:
        raise ValueError("Phase 58 requires canonical MTC mode ordering")
    topology = {n.name:tuple(n.alternatives) for n in logit.each_nest(nest, type="node")}
    expected = {"root": ("AUTO", "NONMOTORIZED", "TRANSIT", "RIDEHAIL"),
        "AUTO": ("DRIVEALONE", "SHAREDRIDE2", "SHAREDRIDE3"),
        "DRIVEALONE": MTC21_ALTERNATIVES[:2], "SHAREDRIDE2": MTC21_ALTERNATIVES[2:4],
        "SHAREDRIDE3": MTC21_ALTERNATIVES[4:6], "NONMOTORIZED": MTC21_ALTERNATIVES[6:8],
        "TRANSIT": ("WALKACCESS", "DRIVEACCESS"), "WALKACCESS": MTC21_ALTERNATIVES[8:13],
        "DRIVEACCESS": MTC21_ALTERNATIVES[13:18], "RIDEHAIL": MTC21_ALTERNATIVES[18:21]}
    if topology != expected:
        raise ValueError("Phase 58 nested topology differs from its reviewed order")
    coefficients = np.asarray(mtc21_coefficients(nest), dtype=np.float64)
    if np.any(~np.isfinite(coefficients)) or np.any((coefficients < .05) | (coefficients > 1)):
        raise ValueError("Phase 58 nesting coefficients outside the qualified domain")
    return coefficients


def reduce_modes(utilities, draws, nest, *, capture=False):
    cp = _cupy()
    coefficients = validate(nest, MTC21_ALTERNATIVES)
    values = cp.ascontiguousarray(utilities, dtype=cp.float32)
    draws = cp.ascontiguousarray(draws, dtype=cp.float64)
    rows = len(values)
    if values.shape != (rows, 21) or draws.size != rows or not rows:
        raise ValueError("Phase 58 requires nonempty aligned utilities and draws")
    choices = cp.empty(rows, dtype=cp.int32)
    logsums = cp.empty(rows, dtype=cp.float64)
    guards, bad = cp.zeros(rows, dtype=cp.uint8), cp.zeros(rows, dtype=cp.uint8)
    probabilities = cp.empty((rows,21) if capture else (1,), dtype=cp.float64)
    kernel()(((rows+127)//128,), (128,), (values, draws, np.int32(rows), cp.asarray(coefficients),
                                        choices, logsums, guards, bad, probabilities, np.int32(capture)))
    if int(cp.count_nonzero(bad).get()):
        raise ValueError("Phase 58 mode probabilities are outside the finite qualified domain")
    return choices, logsums, guards, probabilities


def mode_choice_simulate(runtime, state, choosers, spec, nest_spec, skims, locals_d,
                         mode_column_name, logsum_column_name, trace_label,
                         trace_choice_name, trace_column_names=None, estimator=None,
                         compute_settings=None, explicit_chunk_size=0):
    from activitysim.core import simulate, logit
    from .activitysim_mode_choice import _strict_inputs, _write_report
    from .sharrow_cuda import evaluate_strict_cuda
    if estimator is not None or explicit_chunk_size or state.settings.trace_hh_id or state.settings.use_explicit_error_terms:
        raise ValueError("Phase 58 mode runtime requires unchunked inverse-CDF non-estimation execution")
    validate(nest_spec, spec.columns)
    # Binding the current frame is normally done by simple_simulate.
    simulate.set_skim_wrapper_targets(choosers, skims)
    started = time.perf_counter()
    document, environment, ir_hit, ir_ms = _strict_inputs(state, spec, choosers, locals_d)
    generated = evaluate_strict_cuda(document, environment, rows=len(choosers), return_device=True,
        capture_features=False, locality_tile_rows=1, locality_optimized=False,
        compact_inputs=True, group_skim_indices=True, sparse_zero_coefficients=False,
        expression_float32=True, persistent_plan=True, reuse_buffers=False)
    cp = _cupy()
    draws = runtime.uniform(state, choosers, device_only=True)
    choices, logsums, guards, _ = reduce_modes(generated.utilities, draws, nest_spec)
    risk_rows = cp.asnumpy(cp.flatnonzero(guards))
    host_choices, host_logsums = cp.asnumpy(choices), cp.asnumpy(logsums)
    # Recompute only numerical-boundary rows live, using the SAME GPU-produced
    # utility values and already-consumed random draw. Never advance RNG twice.
    if len(risk_rows):
        utilities = pd.DataFrame(cp.asnumpy(generated.utilities[risk_rows]),
                                 index=choosers.index.take(risk_rows), columns=spec.columns)
        nested = simulate.compute_nested_exp_utilities(utilities, nest_spec)
        conditional = simulate.compute_nested_probabilities(state, nested, nest_spec, trace_label)
        probs = simulate.compute_base_probabilities(conditional, nest_spec, spec)
        host_choices[risk_rows] = logit.choice_maker(probs.to_numpy(), cp.asnumpy(draws[risk_rows]))
        host_logsums[risk_rows] = np.log(nested.root.to_numpy())
    result = pd.DataFrame(index=choosers.index)
    result[mode_column_name] = pd.Categorical.from_codes(host_choices+1, categories=[""]+list(spec.columns))
    if logsum_column_name is not None:
        result[logsum_column_name] = host_logsums
    t = generated.telemetry
    _write_report({"phase": 58, "component": "trip_mode_choice", "trace_label": trace_label,
        "rows": len(choosers), "terms": t.terms, "alternatives": t.alternatives,
        "candidate_used": True, "fallback_used": False, "expression_dtype": t.expression_dtype,
        "persistent_plan": t.persistent_plan, "plan_cache_hit": t.plan_cache_hit,
        "plan_build_ms": t.plan_build_ms, "ir_cache_hit": ir_hit, "ir_compile_ms": ir_ms,
        "binding_resolve_ms": t.binding_resolve_ms, "host_pack_ms": t.host_pack_ms,
        "input_upload_ms": t.input_upload_ms, "kernel_ms": t.kernel_ms,
        "utility_download_ms": 0., "elapsed_ms": (time.perf_counter()-started)*1000,
        "cache_key": t.cache_key, "source_sha256": t.source_sha256,
        "live_boundary_guard_rows": len(risk_rows)})
    runtime.mode_events.append({"rows": len(choosers), "guard_rows": len(risk_rows),
        "dense_utility_download_bytes_avoided": int(generated.utilities.nbytes),
        "seconds": time.perf_counter()-started})
    return result
