# Phase 7 results: batched destination setup and CUDA nested logit

Date: 2026-08-11  
Hardware: NVIDIA RTX A4000 16 GB  
Framework: ActivitySim 1.4.0, cached Sharrow required, CuPy 13.6.0

## Outcome

Phase 7 improves the strongest Phase 6 destination backend in two ways. It
runs the shared OD+DP preprocessor once per trip number rather than once per
purpose, and it replaces ActivitySim's canonical MTC 21-mode pandas nest
reduction with one fused FP64 CUDA kernel.

In the interleaved six-run experiment, median `trip_destination` time falls
from 12.179 to 10.281 seconds, a 1.185x speedup. Median time for all 34 models
falls from 54.459 to 52.166 seconds, a 1.044x speedup. Every optimized run is
faster than every baseline run at both boundaries. All seven substantive final
CSV files are byte-identical in all six runs.

## Why this boundary

Profiling showed that 30 per-purpose destination-logsum calls repeatedly paid
for a 70-expression preprocessor. The Phase 7 batch executes it three times:
once for each trip number. In the batching-only probe, the three passes totaled
449.6 ms and `trip_destination` completed in 10.775 seconds.

The remaining mode-choice logsum path evaluates a 404-row specification into
21 alternative utilities and then reduces a fixed nested-logit tree. Captured
Sharrow utility evaluation totaled 4.327 seconds; first-use compilation/loading
made three purpose calls especially expensive. The smaller reducer was still a
clean GPU target because its topology is fixed and every row is independent.

## Real nested-logit capture

The observer runs the model normally and records inputs immediately after
ActivitySim evaluates mode utilities. It captured:

- 30 purpose/trip-number batches;
- 107,854 rows by 21 alternatives;
- 18.119 MB of FP64 utility values;
- each numeric purpose-specific nest and ActivitySim result.

The capture script is `scripts/capture_phase7_nested_logsums.py`; artifacts are
under `benchmark-results/phase7-nested-logsum-capture`.

## Kernel benchmark

Thirty-one warmed repetitions alternate CPU-first and GPU-first order. CPU is
ActivitySim's `compute_nested_exp_utilities` pandas implementation. GPU timing
includes host-to-device utility transfer, all kernel work, synchronization, and
device-to-host logsum transfer. Compilation and CUDA context creation occur
before measured trials.

| Metric | ActivitySim CPU | CUDA, transfers included |
|---|---:|---:|
| Median, complete 30-batch sequence | 486.626 ms | 120.737 ms |
| Speedup | 1.00x | 4.030x |
| Median saved | | 365.889 ms |
| Observed range | 479.843-513.036 ms | 9.276-164.314 ms |

GPU timing was variable under Windows WDDM, but even the slowest GPU sequence
beat the fastest CPU sequence by more than 3x. Maximum absolute error against
the captured ActivitySim logsums was 3.553e-15.

## Live A/B protocol and results

The baseline is Phase 5 scheduling plus the Phase 6 combined-direction
destination backend. The optimized side uses Phase 5 scheduling plus Phase 7
trip-number batching and CUDA nested logit. Runs are fresh processes in fixed
interleaved order A1/B1/A2/B2/A3/B3. ActivitySim's own timers are used and no
setup time is removed.

| Boundary | A1 | A2 | A3 | A median | B1 | B2 | B3 | B median | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trip_destination` | 12.125 | 12.179 | 12.187 | 12.179 | 10.254 | 10.324 | 10.281 | 10.281 | 1.185x |
| All 34 models | 54.459 | 54.349 | 54.859 | 54.459 | 52.166 | 52.290 | 52.008 | 52.166 | 1.044x |

Paired component savings are 1.871, 1.855, and 1.906 seconds. Paired whole-
model savings are 2.293, 2.059, and 2.851 seconds. The worst optimized versus
best baseline speedups are 1.174x for the component and 1.039x for the model.

## Correctness and safety

SHA-256 checks cover `final_accessibility`, `final_households`,
`final_joint_tour_participants`, `final_land_use`, `final_persons`,
`final_tours`, and `final_trips`. All are byte-identical to the Phase 5
reference in every A and B run. `final_checkpoints.csv` is excluded because it
contains timing metadata.

Destination batching rejects unsupported estimation, three-zone LOS, multiple
preprocessors, and preprocessors that reference purpose-varying coefficients
before sampling consumes random draws. The CUDA reducer validates the exact
alternative order and nest shape. A CUDA failure falls back to ActivitySim over
the already-evaluated utility matrix, so expressions and random draws are not
repeated. The complete suite passes 32 tests.

## Interpretation

The 4.030x result belongs only to the nested-logit reduction boundary. That
boundary saves roughly half a second, while trip-number batching removes much
more repeated setup. The component improves 1.185x and the whole model 1.044x.
These narrower numbers are the appropriate practical claims.

The next evidence priority is external validity: reproduce on a larger public
ActivitySim model, at least one additional NVIDIA GPU, and preferably an
independent machine. Within this prototype, Phase 7 meets its performance,
correctness, fallback, and reproducibility criteria.
