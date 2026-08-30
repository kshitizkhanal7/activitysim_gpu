# Phase 40: device-resident trip-destination sampling

## Outcome

Phase 40 moves the complete full-zone trip-destination sampler onto CUDA. The
GPU keeps the 91,524-by-1,454 utility surface resident while it performs
float32 exponential normalization, ActivitySim-compatible preserved-order
inverse-CDF selection, and duplicate counting. ActivitySim remains the owner of
its keyed random-number ledger and uploads only the 2,745,720 authoritative
draws. The host receives compact choices, selected probabilities, first-pick
flags, counts, and proof flags - never the dense utility or error-bound tables.

The final clean 50,000-household public run completed all 34 ActivitySim model
steps and passed every integration gate. It evaluated 133,075,896 utility
cells, returned zero utility bytes, avoided 1,064,607,168 bytes of dense
device-to-host traffic, and used zero fallback calls.

## Qualified boundary

1. ActivitySim advances its keyed random stream for the exact chooser index and
   returns 30 float64 draws per chooser.
2. The Phase 39 CUDA expression kernel writes the complete utility and
   conservative error-bound matrices to device memory.
3. A resident CUDA kernel computes float32 weights and probabilities, preserves
   ActivitySim's draw ordering, performs inverse-CDF selection, and proves each
   selected CDF interval against the error envelope.
4. The same kernel calculates a conservative selected-probability log-error
   risk. Rows above `0.00009` are exact-adjudicated because the sampling
   probability becomes part of `destination_logsum`.
5. A second CUDA kernel identifies first occurrences and duplicate pick counts.
6. Only compact results and row proof flags return to the host.
7. Precision-ambiguous rows use the cached strict Sharrow `idotter` arithmetic
   ABI. Phase 39 shadow-proved this evaluator array-identical to the live
   Sharrow expression path. It uses the original ActivitySim draws.

Unsupported estimation mode, alternative universes, equations, sample sizes,
or data layouts stop the run. There is no quiet CPU fallback.

## Why a second probability guard was necessary

The first implementation guarded only whether a small arithmetic difference
could change the chosen zone. The complete 442,682-trip output had zero changed
destinations, but one `destination_logsum` differed by `0.000238`, above the
declared `0.0001` diagnostic gate. A stable choice was therefore insufficient:
ActivitySim carries `log(probability / pick_count)` into later calculations.

An overly conservative probability bound fixed the output but exact-routed
91,518 of 91,524 rows and made `trip_destination` take 75.2 seconds. That design
was rejected. The selected-probability risk guard at `0.00009` exact-routes
13,607 rows in total (14.87%), catches the diagnostic outlier, and remains under
the 25% sparsity gate.

| Variant | Guard rows | Sampling time | Result |
|---|---:|---:|---|
| probability bound at `0.000025` | 91,518 | 45.65 s | exact but non-sparse |
| tight CDF plus probability risk | 5,951 | 7.05 s | one `0.000213` logsum failure |
| tight CDF plus same-zone work guard | 23,185 | 15.45 s | exact but over 25% and slower |
| final conservative CDF plus `0.00009` risk | 13,607 | 10.48 s | exact decisions and bounded diagnostics |

These rejected results matter: the final guard was selected by complete-model
evidence, not by a kernel-only microbenchmark.

## Final full-workload evidence

Source artifacts are
[`phase40-p40final1-gpu.json`](../benchmark-results/phase40-p40final1-gpu.json)
and
[`phase40-p40final1-exact.json`](../benchmark-results/phase40-p40final1-exact.json).

| Measurement | Final value |
|---|---:|
| sampling programs | 30 |
| chooser rows | 91,524 |
| alternatives per chooser | 1,454 |
| utility cells | 133,075,896 |
| controlled random draws | 2,745,720 |
| retained sample rows | 2,094,156 |
| utility bytes downloaded | 0 |
| compact host-result bytes | 36,243,504 |
| random-draw upload bytes | 21,965,760 |
| dense utility/bound bytes avoided | 1,064,607,168 |
| exact guard rows | 13,607 (14.87%) |
| CUDA utility time | 0.445 s |
| resident choice kernel time | 1.963 s |
| duplicate kernel time | 0.027 s |
| cached exact guard time | 7.329 s |
| compact download time | 0.028 s |
| host result packing | 0.402 s |
| complete sampling boundary | 10.480 s |
| complete `trip_destination` step | 23.0 s |
| all 34 model steps | 165.0 s |

The independent verifier found zero changed decision cells. Six non-trip CSVs
were byte-identical. `destination_logsum` differed by at most `0.000012`, 8.3
times inside its `0.0001` gate; `mode_choice_logsum` differed by zero.

## Performance interpretation

Phase 40 is a major residency and transfer result, but it is not an incremental
speed promotion over Phase 38. In one fresh matched Phase 38/Phase 40 pair, all
model steps increased from 162.8 to 169.0 seconds and `trip_destination`
increased from 19.0 to 24.4 seconds. The candidate failed the "win every pair"
promotion rule. The final cached-oracle production run improved to 165.0 and
23.0 seconds, but it was not another alternating pair, so it does not replace
that formal result.

Against the existing three-pair regular pinned ActivitySim baseline, the latest
full GPU run is descriptively 1.25x faster overall (206.6 versus 165.0 seconds)
and 1.78x faster for `trip_destination` (41.0 versus 23.0 seconds). This is the
cumulative Phase 17-40 project advantage, not a causal Phase 40-only speedup.
Phase 38 remains the promoted configuration when minimum elapsed time is the
sole criterion.

## What comes next

Phase 40 proves that ActivitySim's full-zone sampler can be device resident
without changing modeled decisions and without returning gigabyte-scale dense
tables. It also shows that transfer elimination does not guarantee a stopwatch
win when 14.87% of rows still require exact CPU arithmetic.

The next major opportunity is a shared Sharrow/CUDA arithmetic compiler, or an
upstream Sharrow GPU backend, that emits the same 15-term reduction semantics
on both devices. If the ordinary CUDA utility is authoritative or bit-identical,
the 7.329-second sparse oracle disappears. A successful next phase must retain
the keyed RNG contract, rerun mutation and complete-output tests, and win at
least three fresh alternating Phase 38/candidate pairs before promotion.
