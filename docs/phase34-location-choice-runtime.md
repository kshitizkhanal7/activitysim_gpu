# Phase 34: location-choice family and full-model proof

## Result

Phase 34 expands the qualified model-wide CUDA boundary to school location,
workplace location, joint-tour destination, at-work subtour destination, and
at-work subtour mode choice. The retained Phase 17, 32, and 33 consumers stay
enabled. On the public Prototype MTC extended model at 50,000 households, three
fresh matched pairs produced this result:

| Pair | Regular ActivitySim | Phase 34 | Saved | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 270.2 s | 185.1 s | 85.1 s | 31.50% | 1.460x |
| 2 | 210.2 s | 199.6 s | 10.6 s | 5.04% | 1.053x |
| 3 | 269.1 s | 207.2 s | 61.9 s | 23.00% | 1.299x |
| Median | **269.1 s** | **199.6 s** | **69.5 s** | **25.83%** | **1.348x** |

Every candidate won. All three independent complete-output comparisons found
zero changed decision cells and zero changed decision rows. This is a
whole-runtime result against regular pinned ActivitySim with Sharrow required,
not a claim that Phase 34's new kernels alone saved 69.5 seconds.

## What changed

ActivitySim still owns configuration, sampling, nesting, random streams,
shadow-pricing iteration, table mutation, and output. Phase 34 intercepts the
repeated `simple_simulate_logsums` arithmetic inside four location-choice
families and evaluates their reviewed expression IR using the generated CUDA
backend. It also applies the same generated utility bridge to at-work subtour
mode choice. The bridge is fail-closed: a missing expected program, a CPU
fallback, an incorrect workload shape, or a changed decision fails the run.

Each qualified candidate executed exactly:

| New Phase 34 group | CUDA calls | Rows |
|---|---:|---:|
| school location logsums | 3 | 685,915 |
| workplace location logsums | 4 | 1,859,082 |
| joint-tour destination logsums | 5 | 76,559 |
| at-work subtour destination logsums | 1 | 310,968 |
| **Location subtotal** | **13** | **2,932,524** |
| at-work subtour mode utility | 1 | 15,100 |

The proof gate checks both the 13-call/2,932,524-row total and every group above,
not merely whether at least one GPU report exists. All three candidates had
zero fallback calls. The at-work component imports its utility function into a
module-local binding, so Phase 34 patches that binding explicitly while keeping
the original ActivitySim implementation as the exact fallback.

## Component evidence

The median component table is the best way to separate the new boundary from
retained acceleration and ordinary runtime noise.

| Component | Regular | Phase 34 | Saved | Speedup | Role |
|---|---:|---:|---:|---:|---|
| school location | 11.0 | 8.4 | 2.6 | 1.310x | new Phase 34 GPU logsums |
| workplace location | 16.8 | 12.3 | 4.5 | 1.366x | new Phase 34 GPU logsums |
| joint-tour destination | 3.7 | 3.5 | 0.2 | 1.057x | new Phase 34 GPU logsums |
| at-work destination | 4.9 | 2.8 | 2.1 | 1.750x | new Phase 34 GPU logsums |
| at-work mode | 1.3 | 1.6 | -0.3 | 0.813x | new Phase 34 GPU utility |
| mandatory scheduling | 29.7 | 15.0 | 14.7 | 1.980x | retained Phase 32 GPU |
| non-mandatory destination | 17.1 | 11.9 | 5.2 | 1.437x | retained Phase 33 GPU |
| non-mandatory scheduling | 12.0 | 7.1 | 4.9 | 1.690x | retained Phase 33 GPU |
| trip destination | 54.1 | 32.8 | 21.3 | 1.649x | retained Phase 17 GPU |
| trip mode | 12.5 | 10.7 | 1.8 | 1.168x | retained Phase 17 GPU |
| **All 34 model steps** | **269.1** | **199.6** | **69.5** | **1.348x** | integrated runtime |

The five new component medians sum to about 9.1 seconds saved. That is the
most defensible estimate of Phase 34's newly targeted contribution in this
experiment. It is not a causal decomposition: component medians are computed
independently, and untargeted CPU steps also varied. At-work mode is an honest
negative result in the median even though a separate smoke run improved its
displaced first-use compilation from 2.7 to 1.2 seconds. The bridge remains
qualified because it expands coverage and can help cold ordering, but it is
not advertised as a standalone speedup.

## Correctness contract

Every published final CSV is compared to its matched control. The verifier
requires identical table schemas, row identities, missing values, and every
substantive modeled decision. Only four named diagnostic columns may differ
within explicit limits:

| Diagnostic | Largest observed absolute difference | Gate |
|---|---:|---:|
| destination logsum | 0.0000100000000032 | 0.0001 |
| mode-choice logsum | 0.0000038135214631 | 0.00001 |
| school-location logsum | 0.0000038146972621 | 0.00001 |
| workplace-location logsum | 0.0000019073486328 | 0.00001 |

All three comparisons passed with zero changed choices. Four output tables are
byte-identical; other tables contain only the declared bounded diagnostic
rounding. The benchmark also proves all 34 steps completed, the controlled
random stream stayed exact, the earlier scheduling boundaries stayed exact,
the CUDA skim allocation was released once after the final consumer, and no
silent fallback occurred.

## Method and limits

The benchmark uses the public Prototype MTC extended example, all 1,454 zones,
50,000 sampled households, pinned ActivitySim source, and an RTX A4000 16 GB.
Each pair starts a regular ActivitySim control and then a Phase 34 candidate in
fresh processes. The primary statistic is the median of three full 34-step
times. The spread is large: pairwise savings range from 10.6 to 85.1 seconds.
That means the direction is replicated - all three candidates win - but the
exact 25.83% magnitude should not be treated as a narrow confidence interval.
More repetitions on a quiet, dedicated machine are needed for publication-
grade uncertainty estimates.

This remains a hybrid runtime. CPUs still perform sampling, joins, pandas table
work, untargeted components, and file output. The GPU acceleration applies to
this configuration and hardware; another region, specification, scale, or
device must rerun the same gates.

## Reproduction

```powershell
.\scripts\run_phase34_location_choice_ab.ps1 `
  -Repetitions 3 `
  -Households 50000 `
  -RunTag p34model1 `
  -Baseline activitysim
```

Primary evidence is
[`phase34-p34model1-summary.json`](../benchmark-results/phase34-p34model1-summary.json),
the three `phase34-p34model1-gpu-*.json` proof reports, and the three
`phase34-p34model1-exact-*.json` complete-output comparisons. The fail-closed
[`phase34-p34model1-audit.json`](../benchmark-results/phase34-p34model1-audit.json)
rechecks those gates and records SHA-256 hashes for every primary evidence and
implementation file.

## Next major opportunity

The next phase should not add another isolated logsum kernel. The largest
credible opportunity is a resident tour-trip state engine that eliminates
repeated pandas packing across location choice and trip construction, then a
qualified GPU trip scheduler. It should retain ActivitySim's public inputs,
random streams, and outputs; run changed-scenario tests; and require a direct
incremental A/B against Phase 34 as well as the regular-ActivitySim comparison.
That design attacks orchestration and data movement, now the main barriers to
larger gains.
