# Phase 51: fused compact destination utility

## Result

Phase 51 removes the last dense row-shaped device ABI between compact
destination inputs and strict utility evaluation. One generated CUDA kernel
now reconstructs the reviewed owner, destination, land-use, wait-time, and
availability values in shared memory and evaluates the 315-term, 21-mode
utility immediately. It does this for all 19 public destination-logsum calls
and 4,696,676 sampled alternatives with no dense-input generator and no
fallback.

The strongest result is capacity, not an incremental speedup. Each qualified
run avoids 1,953,817,216 bytes of dense device allocation across the 19
sequential calls, while their row-owner maps total 18,786,704 bytes. These are
allocation-volume sums, not simultaneous peak memory. The largest call
replaces 295,698,208 dense bytes with a 2,843,252-byte owner map, reducing its
live row state by 292,854,956 bytes. The compatibility bootstrap is 416 bytes
per call (7,904 bytes summed across the calls). Compact host uploads fall from
58,259,296 to 47,787,016 bytes because reviewed integer
owner and land-use state is range-checked and stored as int32.

Three fresh Phase 50/51 pairs preserved all seven published files byte for
byte. The fused kernel pipeline was approximately neutral, while first-use
compilation made the complete instrumented service modestly slower:

| Incremental measurement | Phase 50 | Phase 51 | Interpretation |
|---|---:|---:|---|
| generator + utility / fused kernel pipeline, median | 1.723 s | 1.739 s | 0.9% slower; effectively neutral at this scale |
| complete instrumented destination-logsum service, median | 8.570 s | 8.943 s | 4.35% slower, mainly new first-use compilation |
| five destination components, median | 16.4 s | 17.0 s | 3.66% slower; only 1 of 3 pairs won |
| complete 34-step model, median | 136.300 s | 135.018 s | 0.94% observed improvement; only 2 of 3 pairs won |
| changed modeled decision cells | 0, 0, 0 | 0, 0, 0 | exact |

Accordingly, Phase 51 does **not** claim incremental speed superiority over
Phase 50. It qualifies a deterministic memory-capacity improvement with an
approximately neutral fused kernel.

The separate cumulative comparison against regular pinned ActivitySim with
Sharrow required is strong. Phase 51 won all three complete-model pairs and
all three destination-component pairs:

| Cumulative measurement | Regular ActivitySim | Phase 51 | Result |
|---|---:|---:|---:|
| complete 34-step model, median | 202.600 s | 135.318 s | 67.282 s saved; 33.21% lower; **1.497x** |
| five destination components, median | 42.8 s | 16.9 s | 25.9 s saved; 60.51% lower; **2.533x** |
| complete-model pairs won | - | 3 of 3 | replicated |
| changed modeled decision cells | - | 0, 0, 0 | exact |

That 1.497x result belongs to the accumulated system through Phases 1-51. It
must not be credited to Phase 51 alone.

## Why this phase exists

Phase 50 stopped pandas from constructing and uploading 192,563,716 repeated
row values. It still created the corresponding arrays on the GPU, then had a
second kernel read them immediately. That was fast but required almost two
gigabytes of transient device memory. A larger population, more alternatives,
or a smaller GPU could fail even though the compact source data itself fit.

Phase 51 changes this pipeline:

1. One int32 owner selector is generated on the GPU for every sampled row.
2. A row block reads the compact chooser and destination state.
3. The 10 floating and 31 integer row values are reconstructed cooperatively
   into a small shared-memory workspace.
4. The strict IR evaluates all 315 features from that workspace.
5. The 21 utilities are accumulated in the same source order and arithmetic
   policy as the qualified Phase 50 kernel.
6. Existing resident nested-logit and destination-choice stages consume the
   utilities without a host round trip.

The dense arrays never exist. This is a genuine fused compiler path, not an
allocation that is merely hidden from telemetry.

## Arithmetic and ABI guarantees

The fused backend is generated from the same hashed strict IR as the CPU
reference and earlier CUDA evaluator. It accepts only the reviewed public ABI:

- 10 unique floating row sources;
- 31 unique integer or availability sources;
- six grouped skim coordinate directions;
- 315 ordered feature terms; and
- 21 alternatives.

Aliases such as `column:age` and `name:age` resolve to the same canonical
storage slot. Direct availability expressions read the same resident skim
cubes and use the same car-ownership source. Destination/origin direction and
period selection are explicit in generated source.

Integer compaction is guarded. Before any int64 owner or land-use value is
stored as int32, the host checks the complete compact array against int32
limits. The generated kernel widens shared values back to `long long` at every
strict-IR reference. An out-of-range public input raises an error rather than
overflowing or falling back.

The native strict ABI compiler can now prepare bindings, scalars,
coefficients, resident skim arguments, output storage, and a manifest without
compiling the obsolete dense-input kernel. Its manifest records that the
kernel was intentionally not compiled.

## Execution design and rejected variants

The final design came from measured iteration, not assumption:

- Fully inlining every compact expression in every feature thread passed
  correctness but repeated too many source and skim reads.
- One block per owner reused owner state well but serialized that owner's
  sampled alternatives and was slower on small owner groups.
- One block per row with cooperative shared reconstruction exposed enough
  parallelism and avoided repeated feature-level reconstruction.
- A 128-thread block preserved arithmetic order but was slower than 256
  threads because each feature thread performed more work.
- A second block-wide barrier was removed because the grouped-skim setup's
  existing barrier already makes reconstructed values visible.
- `__restrict__` hints were measured and rejected because they did not improve
  this device.
- Range-checked int32 compact owner/land packets were retained because they
  reduce upload bytes and future capacity pressure while keeping values exact.

Rejected variants are not used to support the qualification claim.

## Replication and exactness

The production experiments use the public Prototype MTC Extended model,
50,000 households, 1,454 zones, all 34 ActivitySim model steps, fresh
control-then-candidate processes, and three matched pairs per comparison.
Each pair runs an independent verifier across accessibility, households,
joint-tour participants, land use, persons, tours, and trips.

All six output comparisons report zero changed modeled decision cells. In the
incremental Phase 50/51 experiment, all seven published CSV files are
byte-identical in all three pairs. The integrated candidate reports also
require all inherited Phase 1-50 gates plus the Phase 51 gates to be true.

The new gates require:

- exactly 19 fused calls and 4,696,676 rows;
- exact 10/31/six-group source coverage;
- more than 1.9 GB of dense device ABI eliminated;
- less than 20 KB of minimal bootstrap storage;
- exactly four row-owner bytes per sampled row;
- positive fused and owner-map kernel timing; and
- zero generator, dense host pack, binding-resolution, and fallback calls.

## Assumptions and claim boundary

The fused fast path is deliberately specific to the reviewed public model. It
assumes the dense zero-based 1,454-zone universe, contiguous alternatives for
each stable owner, current five-period and skim-direction contracts, unchunked
location-logsum calls, stable owner fields, and no separate transit subzone
access-distance column. Land-use content remains fingerprinted.

A changed source count, unsupported skim direction, reordered owner group,
unknown period, out-of-range compact integer, changed land-use table, or
unsupported chunking contract stops the run. There is no silent generic or CPU
fallback behind the Phase 51 flag.

The promoted claim is therefore:

> On the reviewed public 50,000-household benchmark, Phase 51 avoids
> 1.935 GB of net device allocation volume across all 19 destination-logsum
> calls; the largest call uses 292.85 MB less live row state. It
> preserves exact published decisions. Its incremental kernel performance is
> approximately neutral and its full instrumented service is 4.35% slower
> than Phase 50; no incremental speedup is claimed. The accumulated Phase 51
> system is 1.497x faster than regular pinned ActivitySim by the three-pair
> median.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path src)

.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p51final `
  -Baseline phase50 -CandidatePhase 51

.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p51cpu `
  -Baseline activitysim -CandidatePhase 51

.\.venv-phase8\Scripts\python.exe scripts\build_phase51_qualification.py `
  --incremental-summary benchmark-results\phase51-p51final-summary.json `
  --cpu-summary benchmark-results\phase51-p51cpu-summary.json `
  --output benchmark-results\phase51-qualification.json

.\.venv-phase8\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Primary evidence:

- `benchmark-results/phase51-qualification.json`
- `benchmark-results/phase51-p51final-summary.json`
- `benchmark-results/phase51-p51cpu-summary.json`
- `benchmark-results/phase51-p51final-{base,gpu,exact}-{1,2,3}.json`
- `benchmark-results/phase51-p51cpu-exact-{1,2,3}.json`
- `src/choiceforge/destination_input_supergraph.py`
- `src/choiceforge/sharrow_cuda.py`
- `src/choiceforge/native_abi_bootstrap.py`
- `tests/test_destination_input_supergraph.py`
- `tests/test_sharrow_cuda.py`
- `tests/test_native_abi_bootstrap.py`

## Next major phase

Phase 52 should turn the capacity result into a replicated incremental speed
win without restoring the dense ABI. The largest actionable gap is first-use
compilation: Phase 51's median compile portion is about 0.30 seconds above
Phase 50. The next runtime should precompile the shared fused program before
component timing, cache it by IR/source ABI across processes, and retain
compact owner packets across related purpose calls.

The kernel scheduler should also test a generated multi-row owner tile: two or
four rows share cached owner state while independent warps retain row
parallelism and ordered 21-alternative accumulation. Qualification should
require zero changed decisions, no dense ABI, no fallback, and a three-pair
win in both synchronized kernel/service time and the five destination
components. The evidence here shows that further gains must come from
amortization and owner locality, not merely deleting another array.
