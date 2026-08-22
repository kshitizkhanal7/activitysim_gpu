# Phase 6: destination choice, batching, and a whole-model win

## Outcome

Phase 6 found two optimizations at two different boundaries.

1. A segmented CUDA kernel packs all 30 captured trip-destination simulation
   segments into one launch. On the real 53,927-row replay, it is **1.464x
   faster than a batched Numba CPU implementation including transfers** and
   **2.988x faster with device-resident inputs**. All 3,971 selected
   alternatives match exactly.
2. An explicit ActivitySim backend combines the two directional mode-choice
   logsum calculations while preserving ActivitySim's keyed random-number
   sequence. In an interleaved six-run experiment, the complete
   `trip_destination` component improves from a **16.307-second median to
   14.238 seconds: 1.145x faster**. The 34-step model improves from **59.209 to
   56.264 seconds: 1.052x faster**. All seven substantive final CSV files are
   byte-identical in all six trials.

These are separate claims. The integrated component gain currently comes from
eliminating duplicated directional preprocessing and Sharrow utility setup.
The segmented CUDA sampler is proven at its replay boundary but is not enabled
inside the component because ActivitySim requests one purpose segment at a
time, which is below the measured GPU crossover for most calls.

## Phase 6A - capture the real boundary

`scripts/capture_phase6_trip_destination.py` records ActivitySim 1.4's real
trip-destination simulation inputs after expression evaluation. The public
prototype MTC run produced 30 batches, 3,971 choosers, 53,927 ragged rows, 4 to
20 alternatives per chooser, 14 utility terms, and 3.020 MB of expanded terms.

Replay reconstructs utilities, stable logsums, probabilities, random draws,
and chosen positions. It reported zero mismatches for ActivitySim probability
sampling, the CPU implementation, the original per-batch CUDA kernel, and the
new segmented CUDA kernel. Maximum GPU logsum error is `8.73e-6`.

## Phase 6B - make the kernel large enough

The original path launched one grid for each of 30 small segments. Its
transfer-inclusive median was 15.032 ms versus 2.646 ms for the per-batch CPU
suite: only 0.176x CPU speed. Even device-resident, 30 launches took 7.104 ms.

`interaction_batched_terms_choice_f32` stores one coefficient row per segment,
one segment ID per chooser, and CSR-style offsets for ragged choice sets. It
performs utility accumulation, stable logsum-exp, and inverse-CDF sampling in
one launch without a padded utility matrix.

With 21 repetitions on an NVIDIA RTX A4000:

| Real replay boundary | Strong CPU | CUDA | Speedup |
|---|---:|---:|---:|
| 30 separate segments, transfers included | 2.646 ms | 15.032 ms | 0.176x |
| One packed launch, transfers included | 1.241 ms | 0.847 ms | 1.464x |
| One packed launch, device resident | 1.241 ms | 0.416 ms | 2.988x |

The CPU comparator is a fused Numba loop over the same packed terms, segment
coefficients, ragged offsets, and draws.

## Phase 6C - establish the crossover

| Batches | Interaction rows | CUDA/CPU speedup, transfers included |
|---:|---:|---:|
| 1 | 16,016 | 0.502x |
| 5 | 28,570 | 0.836x |
| 10 | 39,706 | 1.119x |
| 15 | 47,794 | 1.319x |
| 30 | 53,927 | 1.469x |

The practical rule on this machine is to batch roughly 35,000 or more rows
before dispatching this shape to CUDA. A single purpose remains on CPU. This is
a measured policy for this hardware and model, not a universal constant.

## Phase 6D - integrate safely

The patch adds an opt-in setting:

```yaml
inherit_settings: true
DESTINATION_LOGSUM_BACKEND: choiceforge_combined
```

The default is `activitysim`; unknown values raise an error. Three-zone models
fall back to the original implementation. The combined backend draws the OD
and DP keyed random blocks in the original order, evaluates the stacked
preprocessor once, evaluates the nested utility/logsum model once for both
directions, and splits results back into original row order.

An early attempt ignored the stochastic-preprocessor draw order. It was faster
but changed downstream results and was rejected. The released path advances
ActivitySim's random channels exactly as before.

`integration/activitysim-1.4-choiceforge.patch` passes `git apply --check`
against the pristine ActivitySim 1.4 wheel. The overlay is
`benchmark-data/configs_phase6_choiceforge`.

## Phase 6E - interleaved whole-model proof

Fresh processes ran in the order A1/B1/A2/B2/A3/B3. A is Phase 5 with
ActivitySim destination logsums. B adds only the Phase 6 overlay.

| Boundary | A seconds | B seconds | Median speedup |
|---|---|---|---:|
| `trip_destination` | 16.483, 16.307, 16.293 | 14.238, 14.276, 14.097 | **1.145x** |
| All 34 model steps | 59.209, 59.210, 58.602 | 56.963, 56.168, 56.264 | **1.052x** |

Every B component trial beat every A trial. The worst B result was still
1.141x faster than the best A result. Paired component savings were 2.245,
2.031, and 2.196 seconds.

SHA-256 comparisons covered seven final files in all six trials: 42 file
comparisons, zero differences. `final_checkpoints.csv` is excluded because it
contains run timing metadata.

## Reproduce

```powershell
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase6_destination_replay.py --repeats 21
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase6_activitysim_component.py
```

One integrated trial uses the Phase 6 and Phase 5 overlays, in that order,
before the Sharrow and base configuration directories. Machine-readable
evidence is in `benchmark-results/phase6-replay-summary.json` and
`benchmark-results/phase6-activitysim-summary.json`.

## Limits and next work

This proves one public 25-zone model on one RTX A4000. Next, restructure each
trip-number iteration so its purpose segments can be prepared together, making
the proven CUDA sampler available live; then repeat on a larger public region,
a second GPU architecture, and Linux. Speed does not show that the behavioral
assumptions are more accurate.
