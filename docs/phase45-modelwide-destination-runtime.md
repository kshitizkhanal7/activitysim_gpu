# Phase 45: model-wide resident destination runtime

## Outcome

Phase 45 extends the resident CUDA sampling system from trip destination to
five other ActivitySim model families: school location, workplace location,
joint-tour destination, non-mandatory-tour destination, and at-work subtour
destination. Nineteen public-model sampling programs now evaluate 274,223,637
dense chooser-by-zone utility cells on the GPU. The same five families use one
compact final sampled-choice adapter for 201,390 choosers and 4,696,676 sampled
alternative rows.

Three fresh matched Phase 44/45 pairs all favored Phase 45 end to end. Median
time across all 34 model steps fell from 152.1 to 149.7 seconds: 2.4 seconds
saved, 1.58% lower, and 1.016x faster. The five targeted components together
fell from 37.1 to 34.5 seconds: 2.6 seconds saved and 1.075x faster.

Every pair passed independent output verification with zero changed modeled
decision cells. Maximum diagnostic differences were 1.91e-6 for school and
workplace location logsums, 3.0e-6 for destination logsums, and zero for mode
logsums, all inside their declared tolerances.

## What moved to the GPU

The destination-sampling specifications reduce to a reviewed contract:

- distance splines over the 1,454-zone `DIST` skim;
- a destination size term transformed with `log1p`;
- shadow-price size and utility adjustments for school and workplace choice;
- two income/distance interactions for workplace choice; and
- a zero-attraction availability term.

The CUDA kernel evaluates this contract over every chooser and legal zone.
It uses the shared grouped-left float32 dot-product policy that reproduces
Sharrow's OpenBLAS association. `log1p(size * shadow)` depends only on the
1,454 destinations, so NumPy calculates those values once on their original
column dtypes and uploads the float32 result. Exhaustive shadow checks on the
largest 38,525,184-cell program found zero utility-bit mismatches.

The resident inverse-CDF kernel uses the previously qualified NumPy-compatible
float32 exponential and pairwise reduction. It consumes ActivitySim's exact
keyed random draws, preserves alternative order, selects 30 draws per chooser,
and performs duplicate counting. Dense utilities do not return to the host.

## Why a small CPU boundary remains

Even with bit-identical utilities, shifted CUDA exponential/division can differ
from NumPy by one float32 unit in the last place. A random draw extremely close
to a cumulative-probability boundary can therefore select a neighboring zone.
The qualification shadow measured the largest relevant boundary displacement
as 2.17e-7. Production uses a conservative 5e-7 gate.

Only 7,313 of 201,390 chooser rows (3.63%) enter this gate. Their already-exact
utility rows are downloaded and NumPy repeats only final normalization and
selection with the same draws. Sharrow is not recompiled and the dense surface
is not reevaluated. The other 96.37% remain on the resident CUDA path.

## Compact final sampled choice

After sampling and destination-specific logsums, ActivitySim chooses one item
from each chooser's narrow sample. The shared Phase 45 adapter validates
contiguous chooser ordering, prunes unused columns, pads ragged samples without
a generic group-by/join path, and then calls Sharrow's authoritative final
utility evaluator and ActivitySim's probability/choice semantics.

This final sampled-choice evaluator is still CPU code. The Phase 45 GPU claim
is specifically the 274.2-million-cell dense sampling utility, normalization,
inverse-CDF, and duplicate-count workload. The compact adapter is a separate
systems optimization around that GPU work.

## Replicated performance

| Pair | Phase 44 all-model | Phase 45 all-model | Saved | Speedup |
|---|---:|---:|---:|---:|
| 1 | 151.5 s | 150.3 s | 1.2 s | 1.008x |
| 2 | 152.5 s | 149.2 s | 3.3 s | 1.022x |
| 3 | 152.1 s | 149.7 s | 2.4 s | 1.016x |
| median | **152.1 s** | **149.7 s** | **2.4 s** | **1.016x** |

| Target component | Phase 44 median | Phase 45 median | Speedup |
|---|---:|---:|---:|
| school location | 8.0 s | 6.7 s | 1.194x |
| workplace location | 11.7 s | 11.3 s | 1.035x |
| joint-tour destination | 3.3 s | 3.5 s | 0.943x |
| non-mandatory-tour destination | 11.3 s | 10.2 s | 1.108x |
| at-work subtour destination | 2.8 s | 2.6 s | 1.077x |
| five-component aggregate | **37.1 s** | **34.5 s** | **1.075x** |

Joint-tour destination is too small to amortize its first-program compilation
and regressed by 0.2 seconds. It remains in the shared runtime because the
five-component aggregate improved in every matched pair, as did the complete
model. This limitation is reported rather than hidden.

The older established regular ActivitySim/Sharrow control has a 206.6-second
median. Compared descriptively, not as a fresh Phase 45 pair, 149.7 seconds is
1.380x faster and 27.5% lower. The causal Phase 45 claim is the fresh matched
152.1-to-149.7 comparison above.

## Reproduction

```powershell
.\scripts\run_phase45_modelwide_destination_ab.ps1 -Repetitions 3 -Households 50000 -RunTag p45final
.\.venv-phase8\Scripts\python.exe scripts\summarize_phase45_qualification.py
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

Authoritative evidence:

- `benchmark-results/phase45-p45final-summary.json`
- `benchmark-results/phase45-p45final-qualification.json`
- `benchmark-results/phase45-p45final-exact-{1,2,3}.json`
- `benchmark-results/phase45-p45final-gpu-{1,2,3}.json`

The final repository-wide regression run passed **186 tests**. Pytest emitted
only a Windows warning that its optional `.pytest_cache` directory was not
writable; test execution and results were unaffected.

## Next major opportunity

Phase 45 demonstrates that one reviewed arithmetic compiler can cover several
destination families. The next large gain should not optimize another small
boundary in isolation. It should build a persistent model-wide destination
service that precompiles the 7-, 9-, and 11-term programs before timed model
steps, retains destination feature vectors and scratch buffers across all 19
calls, batches the small joint-tour segments, and moves the final sampled-choice
utility/probability path onto the same strict GPU arithmetic ABI. Qualification
must retain the current full-output verifier and fresh matched-pair design.
