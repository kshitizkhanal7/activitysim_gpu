# Phase 16: locality, numeric-policy, and component promotion

Phase 16 turns the Phase 15 scale failure into a measured compiler study. It
does not hide unsuccessful ideas. Cooperative row tiles, sparse-coefficient
lowerings, strict FP64 locality changes, and an FP32 expression policy were
implemented and tested separately. Only the configuration that passed its
predeclared correctness and performance boundary is promoted.

## Outcome

The FP32 generated-utility candidate passes the trip-destination component
promotion gate on the public Prototype MTC Extended benchmark at 50,000
households and all 1,454 zones:

| Measure | Phase 11 baseline | Phase 16 FP32 | Result |
|---|---:|---:|---:|
| Trip destination, median of 3 | 28.5 s | 27.8 s | 1.025x |
| Individual trip-destination runs | 28.5, 28.5, 28.8 s | 28.4, 27.7, 27.8 s | every candidate beats every baseline |
| Whole model, median of 3 | 193.014 s | 194.312 s | 0.993x; not promoted |
| Modeled decision differences | - | 0 in all 3 candidates | pass |
| Maximum diagnostic logsum difference | - | 0.000010 | below 0.0001 gate |

The component gate passes. The stricter whole-model gate fails because the
untargeted model steps varied enough to outweigh the median 0.7-second
destination saving. Phase 16 therefore establishes repeatable GPU kernel and
component superiority, not whole-application superiority.

## What changed

The retained compiler path has four independent changes:

1. Scalar constants use scalar device arrays instead of being expanded to one
   value per row.
2. Identical dense input allocations share one ABI slot.
3. Immutable CUDA skim bindings, strict IR documents, and coefficient arrays
   are cached with explicit telemetry.
4. The optional FP32 policy performs expression arithmetic in FP32 before the
   existing ordered FP32 coefficient multiply/add reduction. This aligns with
   Sharrow's FP32 intermediate storage and is substantially better suited to
   the RTX A4000 than forced FP64 expression arithmetic.

The FP64 IR version 3 policy remains available and default. The FP32 policy is
explicitly selected with `CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32=1`; its
source hash, manifest setting, and telemetry label prevent the two policies
from being confused.

## Exactness evidence

The real 1,001-household qualification evaluates the shared IR using both the
FP32 CPU oracle and generated FP32 CUDA:

| Qualification item | Result |
|---|---:|
| Real batches | 30 / 30 exact |
| Rows | 85,126 |
| Feature cells | 32,262,754 / 32,262,754 exact |
| Utility cells | 1,787,646 / 1,787,646 exact |
| Maximum CPU/CUDA feature difference | 0.0 |
| Maximum CPU/CUDA utility difference | 0.0 |
| Candidate fallbacks | 0 |
| Utility downloads before nested logsum | 0 bytes |
| Nested-logsum utility re-uploads | 0 bytes |

The separate FP32 CPU oracle is deliberate. Comparing FP32 CUDA to the FP64
oracle would compare different declared arithmetic policies. Unit coverage
includes an expression where FP32 and FP64 produce different results and
requires CPU-FP32 and CUDA-FP32 to agree exactly.

Final-output replication is a second gate. In every large candidate run all
modeled trip decisions and six non-trip substantive CSV files match. The
printed `destination_logsum` diagnostic differs by at most 0.000010 and is not
used by a later decision in this output comparison.

## Performance mechanism

The 50,000-household workload contains 4,188,312 generated utility rows in 30
batches. Relative to the rejected Phase 15 FP64 candidate, Phase 16 reduced:

| Candidate stage | Phase 15 FP64 | Phase 16 strict-locality FP64 | Phase 16 FP32 |
|---|---:|---:|---:|
| Host packing | 2.407 s | 1.080 s | about 1.1-1.3 s |
| Input upload | 0.642 s | 0.287 s | about 0.3-0.4 s |
| Generated utility kernel | 4.529 s | 4.107 s | about 1.2-1.6 s |

The FP32 scale run measured a 1.228-second utility total; the repeated run's
retained telemetry sample measured 1.559 seconds. The manifest timings, rather
than either internal sample alone, establish the component result.

## Rejected variants

### Cooperative row tiles

Tiles of 2, 4, and 8 rows were exact in GPU tests, including a partial final
tile. On the public model, eagerly staging all 149 skim bindings added more
synchronization and data loading than it saved. It remains experimental and is
not the default.

### Sparse coefficient lowering

Only 419-465 of 7,959 term-alternative coefficients are nonzero in each real
batch. Phase 16 proved that 639,938,136 zero operations could be skipped on the
small public run. Two exact implementations still lost:

- an alternative-specific straight-line lowering enlarged control flow and
  instruction footprint;
- a compact sparse-array lowering replaced coalesced dense work with indirect
  shared-memory gathers.

The RTX A4000 executes the dense alternative warp efficiently, so arithmetic
count alone was a misleading predictor. Sparse lowering is opt-in research
code and disabled by default.

### Strict-locality FP64

Compaction and grouping improved Phase 15's internal costs but not enough. At
50,000 households it produced 32.5 seconds for trip destination versus 30.1
seconds for its paired baseline. That candidate is rejected.

## Reproduce

Run the FP32 CPU/CUDA exactness and output gate:

```powershell
.\scripts\run_phase16_candidate.ps1 `
  -Households 1001 -RunTag reproduce-p16-exact `
  -TileRows 1 -EnableFloat32Expressions
```

Run the component promotion proof:

```powershell
.\scripts\run_phase16_incremental_ab.ps1 `
  -Households 50000 -Repetitions 3 -RunTag reproduce-p16-proof `
  -TileRows 1 -EnableFloat32Expressions -RequireComponentPromotion
```

Use a new run tag because the scripts refuse to overwrite an existing output.
The whole-model gate is intentionally separate and can be enforced with
`-RequirePromotion`; it does not pass in the reported Phase 16 series.

Primary machine-readable evidence:

- `benchmark-results/phase16-p16fp32exact-qualification.json`
- `benchmark-results/phase16-p16fp32exact-output-verification.json`
- `benchmark-results/phase9-mtc-full-p16fp32proof-runs.json`
- `benchmark-results/phase16-p16fp32proof-component-summary.json`

## Claim boundary and next work

Phase 16 supports this statement: on the recorded RTX A4000 workstation, the
generated FP32 GPU utility path is exactly reproducible against its published
FP32 CPU oracle, preserves all modeled decisions in the reported public runs,
and provides repeated trip-destination component superiority at 50,000
households.

It does not support a whole-model superiority claim. The next phase should
reduce about one second of per-run binding/setup overhead, capture telemetry
for every repetition, and repeat the whole-model proof on a quieter controlled
host. Cross-hardware replication and a second public ActivitySim model remain
required before an upstream production proposal.
