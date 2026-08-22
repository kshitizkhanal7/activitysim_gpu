# Phase 15: device-resident strict CUDA candidate

## Outcome

Phase 15 moved strict IR version 3 utilities from shadow mode into the real
ActivitySim trip-destination logsum path. Generated utilities remain on the GPU
through MTC-21 nested-logsum reduction, with no utility download or reducer
re-upload. The backend is exact and repeatably faster for the destination
component at the 1,001-household qualification scale. It is not promoted for
the 50,000-household benchmark because that scale is slower than Phase 11.

The machine-enforced promotion gate requires at least three fresh interleaved
pairs, exact modeled decisions, positive destination and whole-model medians,
and complete timing separation. The final trial passes the destination
conditions but fails the whole-model conditions.

## Implementation

- An explicit `CHOICEFORGE_STRICT_CUDA_CANDIDATE=1` switch intercepts the
  Sharrow flow; the default remains off.
- The same hashed IR drives the strict CPU oracle and generated CUDA source.
- A compact ABI passes float32 skim cubes and ActivitySim's mapped origin,
  destination, and time positions to the kernel. It avoids a 379-column host
  feature table.
- Non-skim float, integer, and Boolean leaves retain separate typed storage.
- Coefficients, kernels, and immutable skim cubes are cached by semantic
  content or underlying allocation, not ephemeral wrapper identity.
- Generated utilities feed the nested reducer while device-resident.
- The skim cache is released after trip destination so later steps do not
  inherit allocator pressure.
- Generation or reduction errors fall back to authoritative Sharrow.
- A row policy can reject an ineligible batch before large allocation.

## Exactness qualification

The final compact qualification used public Prototype MTC Extended data, 1,001
households, all 1,454 zones, 379 terms, and 21 alternatives.

| Gate | Result |
|---|---:|
| Real batches | 30 / 30 exact |
| Rows | 85,126 |
| Feature cells | 32,262,754 / 32,262,754 exact |
| Utility cells | 1,787,646 / 1,787,646 exact |
| Maximum CPU/GPU feature difference | 0.0 |
| Maximum CPU/GPU utility difference | 0.0 |
| Utility device-to-host bytes | 0 |
| Reducer host-to-device bytes | 0 |

The complete output gate requires every modeled `final_trips.csv` field to
match exactly and every non-trip substantive final CSV to be byte-identical.
`destination_logsum` is a reported diagnostic, not a decision. Its maximum
difference was 8e-6 at 1,001 households and 1e-5 at 50,000 households, below
the published 1e-4 bound. This is the expected difference between strict IR
and current Sharrow semantics, not a strict CPU/GPU mismatch.

## Direct performance evidence

The clean attribution experiment gives both conditions Phase 11 batching and
the CUDA nested reducer. The only intended difference is generated strict
utilities versus Sharrow utilities.

Three fresh A1/B1/A2/B2/A3/B3 pairs at 1,001 households produced:

| Boundary | Phase 11 median | Compact Phase 15 median | Result |
|---|---:|---:|---:|
| Trip destination | 11.3 s | 10.3 s | 1.097x; 1.0 s saved |
| All model steps | 84.631 s | 85.445 s | 0.990x; not promoted |

All Phase 15 destination times (10.0, 10.3, 10.4 s) were below all Phase 11
times (11.3, 11.3, 11.4 s). All 90 utility batches handed device utilities to
the reducer, and all three output comparisons preserved every decision.
Whole-model times included unrelated scheduling variation, so the strict gate
failed instead of overclaiming the component result.

## Scale rejection

One fresh diagnostic pair at 50,000 households and 1,454 zones evaluated
4,188,312 utility rows. It is enough to reject, not claim superiority:

| Boundary | Phase 11 | Compact Phase 15 | Result |
|---|---:|---:|---:|
| Trip destination | 28.4 s | 33.3 s | 0.853x |
| All model steps | 192.519 s | 197.871 s | 0.973x |

Decisions remained exact and peak GPU memory was 9,196 MiB. The compact ABI is
a large improvement over the first unbounded prototype (58.2 s destination,
223.385 s whole model), but remains slower than Sharrow at scale. A repeated
50,000-household campaign was not run after this clear rejection.

## Reproduction

```powershell
.\scripts\run_phase15_candidate.ps1 -Households 1001 `
  -RunTag compact1 -MaxCandidateRows 2000000

.\scripts\run_phase15_incremental_ab.ps1 -Households 1001 `
  -Repetitions 3 -RunTag p15finalr3 -MaxCandidateRows 2000000
```

Primary evidence is in `benchmark-results/phase15-candidate-summary.json`,
`benchmark-results/phase15-p15finalr3-summary.json`, and
`benchmark-results/phase15-p15compact50-summary.json`. Manifests record the
ActivitySim commit, patch, environment lock, GPU, configuration hashes,
candidate switch, and row policy.

## Decision and next architecture

Phase 15 is complete as a correctness, integration, and component-performance
phase. It is opt-in and fail-safe, not the production default. Phase 11 remains
the supported 50,000-household result.

The next compiler should tile rows, coalesce repeated OD/time skim gathers,
reuse gathered values across terms, and fuse ordered utility accumulation
without a global feature matrix. It must consume strict IR version 3 and pass
the same exact CPU/GPU and complete-output gates before another promotion.
