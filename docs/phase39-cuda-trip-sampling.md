# Phase 39: exact CUDA trip-destination sampling utility

## Result

Phase 39 successfully moves the largest trip-destination sampling arithmetic
surface to CUDA, preserves ActivitySim's modeled decisions, and proves why this
particular hybrid boundary must **not** replace Phase 38. On the public
50,000-household, 1,454-zone Prototype MTC model, 30 CUDA launches calculated
133,075,896 utility cells for 91,524 trip choosers. They avoided constructing a
133-million-row pandas Cartesian table and used no fallback.

The candidate passed exact output verification in all three fresh matched
Phase 38/39 pairs. It nevertheless won only one pair. Median all-model-step time
rose from 160.8 to 165.6 seconds (0.971x, or 2.99% slower), and median
`trip_destination` time rose from 18.8 to 23.5 seconds (0.800x, or 25.0%
slower). The strict performance promotion gate therefore failed. Phase 39 is an
opt-in research path, not the new default.

## What was moved

ActivitySim's public sampler evaluates every destination zone before retaining
a smaller sample. Phase 39 recognizes the reviewed 15-row sampling expression
program, validates its coefficients and data layout, and generates a flat CUDA
kernel over every chooser/zone pair. The kernel reads three resident skim cubes,
size terms, trip flags, and the complete zone universe. It writes a float32
utility and an arithmetic-error envelope per cell.

The following semantics deliberately remain ActivitySim-owned on the CPU:

- probability normalization;
- its keyed random-number ledger;
- inverse-CDF selection in preserved draw order;
- duplicate collapse and `pick_count`;
- retry and workflow orchestration.

This division made the implementation independently testable, but it also
created the performance wall measured below.

## Exact arithmetic guard

The CUDA accumulation and Sharrow's Numba/NumPy dot product can differ by a few
float32 bits even when both evaluate the same 15 terms. A small utility change
usually cannot change a random choice, but occasionally a draw is close to a
CDF boundary.

Phase 39 computes a conservative per-cell error envelope. From it, the CPU
derives lower and upper CDF limits for every retained draw. A chooser row is
accepted only when all draws choose the same alternative throughout that
interval. Otherwise the complete 1,454-zone row is recomputed by the live
Sharrow evaluator and the original random draws are reused. The production
benchmark guarded 10,721 of 91,524 chooser rows (11.71%), or 15,588,334 utility
cells.

A separate cached Numba evaluator reconstructs Sharrow's 15-element float32
intermediate vector and exact `np.dot(..., out=...)` call. In a full guard-shadow
run, all 15,588,334 guarded utilities were bit-for-bit identical to live
Sharrow. It is retained as an independent oracle; live Sharrow remains the
production adjudicator because it was faster on this machine. Any unreviewed
expression, coefficient shape, zone universe, skim layout, or size-term
contract fails closed.

## Public-benchmark evidence

| Measure | Qualified result |
|---|---:|
| households | 50,000 |
| zones | 1,454 |
| ActivitySim steps | 34 of 34 |
| sampling programs | 30 of 30 |
| chooser rows | 91,524 |
| CUDA utility cells | 133,075,896 |
| dense host cross-join rows avoided | 133,075,896 |
| ActivitySim random draws retained | 2,745,720 |
| sampled output rows | 2,094,156 |
| guarded chooser rows | 10,721 (11.71%) |
| guarded utility cells | 15,588,334 |
| downloaded dense utility bytes | 532,303,584 |
| CUDA utility kernel, across three pairs | 0.706–0.813 s |
| complete Phase 39 sampling boundary | 9.503–10.088 s |
| exact-guard time inside that boundary | 5.663–6.184 s |
| CUDA fallbacks | 0 |
| matched pairs with exact modeled decisions | 3 of 3 |
| matched pairs won | 1 of 3 |
| median whole-model Phase 38 / Phase 39 | 160.8 / 165.6 s |
| median whole-model speedup | 0.971x (2.99% slower) |
| median trip-destination Phase 38 / Phase 39 | 18.8 / 23.5 s |
| median trip-destination speedup | 0.800x (25.0% slower) |

The three all-model pairs were 167.7/166.5, 160.3/163.4, and 160.8/165.6
seconds in Phase 38/39 order. The first candidate won by 1.2 seconds; the next
two lost by 3.1 and 4.8 seconds. Reporting the median and every pair prevents
the one favorable run from being mistaken for a reproducible speedup.

## Rejected tighter envelope

An experimental tighter envelope was exhaustively compared with live Sharrow
over all 133,075,896 utility cells and had zero cell-level bound violations. It
reduced the guarded population to 2,037 rows. End-to-end verification still
found a maximum `destination_logsum` difference of 0.0002125, above the
project's 0.0001 diagnostic gate, even though every published decision column
remained exact. That experiment was rejected and the conservative production
envelope restored. This is an important distinction: a utility-error proof is
not automatically a proof for every downstream diagnostic that consumes
sample probabilities.

## Why the kernel win does not become a model win

The GPU arithmetic is not the bottleneck. It needs less than one second across
all 30 programs. The hybrid design then downloads 532 MB of utilities, performs
CPU softmax and inverse-CDF work, and invokes exact Sharrow on ambiguous rows.
Those costs dominate the saved arithmetic. Phase 38 already evaluates the
smaller post-sampling destination/logsum workload efficiently, so adding this
pre-sampling hybrid stage increases total time.

This result does not say GPUs are unsuitable for sampling. It says the useful
boundary must be larger and resident. Moving only utility generation creates a
dense device-to-host handoff and preserves the serial orchestration that needs
to be removed.

## Reproduction

Run three fresh Phase 38/39 pairs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase39_cuda_sampling_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p39proof
```

The wrapper runs the complete 34-step model, verifies outputs after every pair,
writes component timings, and fails if the candidate does not win every pair.
The checked-in evidence intentionally records that performance failure.

To compare the dedicated guard oracle with live Sharrow, set
`CHOICEFORGE_PHASE39_GUARD_SHADOW=1`. To compare every CUDA utility with
Sharrow during qualification, set `CHOICEFORGE_PHASE39_UTILITY_SHADOW=1`.
These modes add substantial work and are not timing modes.

## Claim boundary and Phase 40

Phase 39 proves full-zone CUDA utility coverage, exact modeled decisions in
three matched public runs, and a fail-closed arithmetic bridge. It does not
prove a speedup, a GPU-only model, universal validity for changed expressions
or datasets, or exact equality of every intermediate float. Its performance
gate failed, so Phase 38 remains the promoted runtime.

Phase 40 should be one ambitious resident-sampler phase, not another small
kernel tweak. A shared expression IR must define one canonical arithmetic ABI
for Sharrow CPU and CUDA; utilities must stay on device through softmax,
controlled draws, inverse CDF, duplicate collapse, and sample correction; and
only the compact sampled rows should cross back to ActivitySim. Qualification
must include changed-input mutation tests, exhaustive arithmetic shadowing,
exact random-ledger and published-decision checks, GPU memory/occupancy
counters, and at least three fresh Phase 38/candidate pairs. This attacks the
measured 8+ seconds around the sub-second kernel and removes the 532 MB
download—the only Phase 39 successor with a credible path to a material model
gain.
