# Phase 12: device-side destination-utility foundation

## Outcome

Phase 12 establishes and tests the missing device-side boundary between an
ActivitySim trip-destination utility evaluator and the existing CUDA MTC-21
nested-logsum kernel.  It is a verified kernel pipeline, but it is **not yet
enabled for the Prototype MTC Extended ActivitySim configuration**.  That
distinction is intentional: a correct full deployment needs a model-specific
compiler for the real trip-mode expressions and skim semantics.

## Why this is the next opportunity

The Phase 11 profiler measured 12,564,936 utility rows (2.111 GB) sent from
the CPU into the nested-logsum kernel.  The final reduction itself took only
603 ms in aggregate; its host-to-device transfers took 1,623 ms.  So merely
tuning that reduction cannot produce a major end-to-end result.  The remaining
high-value target is to calculate the 21 utilities on the GPU and pass that
device matrix directly to the reducer.

ActivitySim currently calls Sharrow's CPU `flow.dot(...)` for this work.  The
real MTC `trip_mode_choice.csv` includes many preprocessed fields, availability
conditions, directional skims, clipping/piecewise expressions, and 21
alternatives.  Reimplementing the whole expression language without an
equivalence gate would risk changing travel choices.  Phase 12 therefore takes
the smallest safe cut.

## Implemented ABI

[`choiceforge.destination_utility.LoweredDestinationUtility`](../src/choiceforge/destination_utility.py)
represents a utility specification only after an upstream compiler has lowered
it to this explicit numeric form:

```
utilities[row, alternative] = sum(features[row, feature] * coefficient[feature, alternative]) + constant[alternative]
```

The object requires unique, ordered feature and alternative names, float64
finite coefficient data, and exact matrix dimensions.  This prevents a silent
column-order mismatch from producing plausible but incorrect choices.

`mtc21_logsums_from_lowered_cuda` performs the following device pipeline:

```mermaid
flowchart LR
  A["Lowered numeric features"] --> B["float64 GPU utility GEMM"]
  B --> C["21 utilities remain on GPU"]
  C --> D["CUDA MTC-21 nested logsum"]
  D --> E["One host logsum per row"]
```

The intermediate 21-column utility matrix is not downloaded and re-uploaded.
The existing nested reducer reports zero host-to-device milliseconds whenever
it receives this device array.

## Verification

The new tests cover:

- exact CPU evaluation of the declared linear ABI;
- GPU utility values against that CPU reference to `1e-12` relative and
  absolute tolerance;
- end-to-end lowered utility plus nested logsum values against a CPU-utility
  reference; and
- zero intermediate device-to-host transfer and zero reducer host-to-device
  transfer.

All repository tests pass: **41 passed** (the only warning is the workspace's
non-writable pytest cache).

## Measured kernel-pipeline result

Run:

```powershell
.\.venv-phase8\Scripts\python.exe benchmarks\benchmark_phase12_lowered_pipeline.py `
  --rows 250000 --features 64 --repetitions 3 `
  --output benchmark-results\phase12-lowered-utility-250k.json
```

On this workstation (RTX A4000), fixed-seed 250,000-row, 64-feature,
21-alternative lowered workload results were:

| Measurement | Median |
|---|---:|
| CPU reference utility + canonical nested logsum | 235.215 ms |
| GPU lowered utility + device-resident nested logsum | 28.988 ms |
| Pipeline speedup | 8.114x |
| Maximum permitted difference | `1e-11` (passed) |

The 128 MB feature upload was 12.402 ms in the final measured call; GPU utility
GEMM was 4.618 ms; the nested reducer kernel was 2.120 ms; and its input upload
was exactly 0 ms.  This is a controlled microbenchmark, not an ActivitySim
model result.  The CPU reference is a vectorized NumPy implementation of the
same lowered utility and canonical nesting.

The machine-readable record is
[`phase12-lowered-utility-250k.json`](../benchmark-results/phase12-lowered-utility-250k.json).

## Expression-lowering progress

The Phase 12 foundation now includes
[`activitysim_expression.py`](../src/choiceforge/activitysim_expression.py), a
reviewed AST interpreter that runs against either NumPy or CuPy.  It does not
use Python `eval`.  It supports the precise syntax currently used by the public
MTC trip-mode spec: arithmetic, comparisons, availability masks, `df` columns,
`od_skims`, `odt_skims`, `dot_skims`, `clip`, and `np.maximum`.

It parsed and evaluated all **253 unique expressions** in the public
`trip_mode_choice.csv` with both NumPy and CuPy test data.  It can lower a
coefficient-resolved ActivitySim spec into a named feature matrix and the
`LoweredDestinationUtility` ABI.  New syntax, missing names, unresolved
coefficient strings, inconsistent row counts, or ambiguous labels raise an
error; they never select a GPU approximation.

This completes the expression side of the first compiler slice.

## Directional skim adapter

[`cuda_skims.py`](../src/choiceforge/cuda_skims.py) now provides the matching
data side for standard dense ActivitySim `SkimDict` objects.  It uploads the
invariant row-major skim cube once, then lazily gathers 2D and time-stacked 3D
skims by their existing block offsets.  It deliberately uses ActivitySim's own
`OffsetMapper` for origin/destination mapping, so noncontiguous zone IDs keep
the framework's exact mapping semantics.  The adapter also preserves the
permitted `-1` “not in skim” behavior (returns NaN) and rejects other invalid
zones.  Sparse `MazSkimDict` overlays are explicitly unsupported and fail
closed.

The `CudaSkimWrapper` and `activitysim_cuda_environment` helpers turn real
chooser column names plus ActivitySim-style `od_skims`, `odt_skims`, and
`dot_skims` wrappers into lazy CuPy inputs for the expression compiler.  Tests
verify 2D lookup, period-specific 3D lookup, noncontiguous zone mapping,
missing-zone NaN behavior, invalid-zone rejection, and lazy wrapper behavior.

This is an adapter, not an enabled model switch: it needs captured-batch
comparison against the public `Network_LOS` configuration before the
ActivitySim destination path can invoke it.

## What remains before an end-to-end claim

1. Capture real public-MTC `Network_LOS` batches and compare every GPU skim
   gather with its `SkimDict` lookup, including all periods, zone offsets,
   missing values, and data types.  Unsupported data layouts must report a
   reason and use Sharrow/ActivitySim unchanged.
2. For each supported expression, compare its lowered feature value and all 21
   utilities with ActivitySim on captured production batches.  Require a
   predeclared tolerance and verify the final logsums.
3. Enable the backend only after those checks, behind an explicit configuration
   flag and CPU fallback.
4. Rerun the Phase 11 interleaved 50,000-household A/B protocol.  A performance
   claim requires byte-identical substantive outputs in every pair and a
   predeclared median/worst-pair improvement.  The Phase 12 microbenchmark must
   never be substituted for this gate.

The next logical target is a captured-batch equivalence harness and an opt-in
ActivitySim call-site.  Only then can this device pipeline replace Sharrow for
the supported MTC trip-mode logsums and face the Phase 11 byte-identical A/B
benchmark gate.

## Phase 13 update

The first target described above is complete as a strict CPU reference and
real Sharrow comparison gate. See
[`phase13-strict-cpu-reference.md`](phase13-strict-cpu-reference.md). Current
Sharrow remains authoritative. Phase 14 has also generated CUDA from the same
revised IR and matched the strict CPU oracle exactly on all 30 real public
batches. See
[`phase14-strict-cuda-generator.md`](phase14-strict-cuda-generator.md). A
production replacement still requires the repeated byte-identical A/B gate.
