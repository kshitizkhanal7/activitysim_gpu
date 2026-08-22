# Phase 17: persistent execution and trip-mode continuation

Phase 17 addresses the most important lifecycle cost left after the Phase 16
kernel win. It makes generated CUDA work reusable across calls, carries that
backend into trip mode choice, and tests the result with exact output gates and
five fresh interleaved public-benchmark pairs.

## Outcome

On the public Prototype MTC Extended benchmark, sampled at 50,000 households
with all 1,454 zones:

| Measure | Baseline | Phase 17 | Result |
|---|---:|---:|---:|
| Trip destination, median of 5 | 28.4 s | 27.3 s | 1.040x; component gate passes |
| Paired destination savings | - | 1.8, 0.8, 0.9, 0.8, 1.9 s | all five positive |
| Bootstrap median destination saving, 95% | - | 0.8 to 1.9 s | excludes zero |
| Whole model, median of 5 | 191.474 s | 190.307 s | 1.006x |
| Paired whole-model savings | - | 1.845, -0.042, 1.360, -0.038, 2.595 s | 3 of 5 positive |
| Bootstrap median whole-model saving, 95% | - | -0.042 to 2.595 s | includes zero; strict gate fails |
| Modeled decision differences | - | 0 in all five runs | pass |

The result is stronger than Phase 16 at the target-component boundary and is
the first repeated series in which the Phase 17 whole-model median is also
faster. The repository still does not label whole-model superiority proven:
the predeclared strict gate requires every candidate whole-model time to beat
every baseline, and the bootstrap interval must not be presented as excluding
zero when it does not.

## What changed

### Persistent compiled plans

`CompiledStrictCudaPlan` retains one generated source/kernel pair, the device
coefficient matrix, typed bindings, hashes, and an optional reusable workspace.
On the 30-batch destination run, ten distinct plans are built and the remaining
20 calls are hits. Trip mode choice then records ten more plan hits and no
builds.

Plan reuse is fail closed. The key includes the IR, coefficient policy,
generated-source policy, and device. Each call also validates its binding ABI.
Row arrays and scalar values may change; semantic kind, scalar-versus-row role,
compact alias layout, storage slot, and skim rank may not. A mismatch selects
or builds another valid plan instead of running the wrong kernel.

Persistent mode uses stable scalar slots. Two equal scalar values no longer
share a slot merely because they happen to be equal in one call, so a later
value change cannot corrupt the reused ABI.

### Trip-mode continuation

The destination candidate deliberately bypasses Sharrow utility evaluation.
An early Phase 17 experiment improved destination but made `trip_mode_choice`
about 2.4 seconds slower: it had displaced Sharrow's cold flow compilation into
that later component.

The opt-in trip-mode bridge solves the lifecycle problem. It uses the same
strict IR and generated FP32 CUDA plans for the trip-mode utility matrix. Only
utility evaluation changes. ActivitySim remains authoritative for its nested
logit, probabilities, random-number stream, choices, and output assembly. Any
unsupported input or runtime error falls back to the original Sharrow call and
is counted; the reported proof has zero fallbacks.

### Exact decision policy

Exact byte equality is required for all modeled trip decisions and six
substantive non-trip CSV files. Two printed diagnostic columns have explicit
numeric gates because they are not later used for a reported decision:

| Diagnostic | Observed maximum | Gate |
|---|---:|---:|
| `destination_logsum` | 0.0000100000000032 | 0.0001 |
| `mode_choice_logsum` | 0.0000028958556211 | 0.00001 |

The verifier excludes only those named diagnostics. Every other trip column
must match exactly. This prevents a broad floating-point tolerance from hiding
a changed destination, mode, or other modeled result.

## Qualification evidence

The final 1,001-household qualification (`p17deviceq`) checks the shared FP32
CPU oracle against generated CUDA before relying on model outputs:

| Qualification item | Result |
|---|---:|
| Real destination batches | 30 / 30 exact |
| Rows | 85,126 |
| Feature cells | 32,262,754 / 32,262,754 exact |
| Utility cells | 1,787,646 / 1,787,646 exact |
| CPU/GPU maximum feature or utility difference | 0.0 |
| Destination plan hits | 20 of 30 calls |
| Trip-mode plan hits | 10 of 10 calls |
| Candidate fallbacks | 0 |
| Final decision differences | 0 |
| Maximum destination / mode logsum difference | 0.000008 / 0.000001906741 |

The five-pair 50,000-household proof contains 150 destination reports covering
20,941,560 rows and 50 mode reports covering 2,213,410 rows. It records 100
repeat destination plan hits, 50 trip-mode plan hits, and zero fallbacks.

## Reusable workspace experiment

`CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS=1` reuses plan-local device input and
output allocations. The first call grows a workspace geometrically; a later
call can borrow it only when its schema-selected plan and capacity match.
Returned device arrays are borrowed until the next sequential call on that
plan, which matches the ActivitySim bridge's immediate-consumption contract.

Two implementations were measured. A pinned-host version reduced upload time
but made column packing slower and was rejected. The retained device-only
version keeps NumPy's vectorized host packing and reuses CUDA allocations. Its
single 50,000-household diagnostic reduced destination host-pack plus upload
telemetry from 1,471.5 ms in the earlier plan-only run to 1,364.1 ms, about
107 ms. That cross-run telemetry is useful mechanism evidence, not a repeated
whole-model proof. The switch therefore remains off in the primary proof.

## Stability finding

The first attempted five-pair series used 24 BLAS threads and encountered a
native Windows heap failure during a baseline run, accompanied by OpenBLAS
thread metadata warnings. No candidate code was active in that failed process.
The final proof fixes both `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` at 16,
records the setting in the manifest, and completes all ten processes. The
partial 24-thread series is not used as evidence.

## Reproduce

Run exact shared-IR and final-output qualification:

```powershell
.\scripts\run_phase17_candidate.ps1 `
  -Households 1001 -RunTag reproduce-p17-exact -EnableModeChoice
```

Run the five-pair proof:

```powershell
.\scripts\run_phase17_incremental_ab.ps1 `
  -Households 50000 -Repetitions 5 -RunTag reproduce-p17-proof `
  -BlasThreads 16 -EnableModeChoice -RequireComponentPromotion
```

Add `-EnableReusableBuffers` only to reproduce the opt-in workspace experiment.
Use a fresh run tag because runners refuse to overwrite evidence.

Primary machine-readable evidence:

- `benchmark-results/phase17-p17deviceq-qualification.json`
- `benchmark-results/phase17-p17deviceq-output-verification.json`
- `benchmark-results/phase9-mtc-full-p17modeproof16-runs.json`
- `benchmark-results/phase17-p17modeproof16-summary.json`
- `benchmark-results/phase17-p17device50-summary.json`

## Claim boundary and next phase

Phase 17 proves generated-GPU superiority for the trip-destination component
on this machine with exact modeled decisions and a positive repeated saving.
It also supplies encouraging five-pair whole-model evidence: a 1.006x median
speedup. It does not prove whole-model superiority under the strict predeclared
gate or a confidence interval excluding zero.

The next phase should first obtain more independent controlled repetitions to
narrow the whole-model interval. After that, the highest-value replication is
the same locked qualification and A/B protocol on a second NVIDIA GPU and a
second public ActivitySim model. An upstream proposal should expose the strict
IR, FP32 policy, hashes, ABI validator, fallback telemetry, and claim boundary,
not only a benchmark headline.
