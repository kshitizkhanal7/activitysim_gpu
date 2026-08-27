# Phase 30: native strict-ABI bootstrap

## Outcome

Phase 30 removes ActivitySim's dense chooser-alternative logsum preprocessor
from the live production bootstrap. The strict CUDA invocation is compiled
directly from reviewed hashed utility IR, an explicit raw-source type
contract, scalar settings, controlled random draws, and immutable raw skim
metadata. The legacy dense path now runs only in a separate oracle process.

This closes the main boundary left by Phase 29. It does not claim that all of
ActivitySim is CPU-free. ActivitySim still restores the public checkpoint,
owns upstream tables and workflow configuration, loads the skim dataset, and
publishes the result.

## Native bootstrap contract

`NativeStrictAbiCompiler` accepts no dense preprocessor frame and no captured
strict invocation. For each purpose it validates and records:

- one 315-term utility IR and 21 alternatives;
- 10 floating-point and 31 integer per-row sources;
- 48 scalar sources;
- 209 logical skim sources;
- six grouped coordinate sets; and
- the generated source, IR, and complete ABI schema SHA-256.

Unknown source names, unsupported dtypes, or wrong-rank skim arrays fail
compilation. Work, school, and university produce three stable purpose-specific
IR and schema hashes; each repeats exactly for the first-tour and later-tour
program.

The shared code generator retains the qualified arithmetic contract:
float32 expressions and features, Sharrow-compatible fused float32 utility
accumulation, grouped skim indices, and direct utility output without a
captured feature table.

## Live execution sequence

The native branch in `run_phase22_integrated_scheduling.py` bypasses the
original dense `_compute_logsums` call. It:

1. loads the reviewed tour-mode settings, specification, coefficients, and
   nested-logit tree;
2. creates and hashes the shared strict IR;
3. obtains one controlled six-column random draw and retains the first
   per-tour draw, preserving ActivitySim's stream advancement;
4. constructs the Phase 29 one-row-per-tour and land-use raw source;
5. uploads immutable xarray skim cubes without installing dense target rows;
6. compiles the native ABI and raw-table expansion with oracle validation
   disabled;
7. executes strict utility and nested logsum on CUDA; and
8. feeds the result directly into the resident cache, timetable, and choice
   graph.

The placeholder returned to ActivitySim is index-aligned but carries no
modeled result. The resident GPU cache is authoritative.

## Public benchmark proof

The benchmark is the public Prototype MTC Extended 50,000-household checkpoint:

| Quantity | Value |
|---|---:|
| Mode-logsum rows | 1,210,124 |
| Scheduled mandatory tours | 81,983 |
| Real utility programs | 6 |
| Terms per program | 315 |
| Mode alternatives | 21 |
| Independent native processes | 3 |
| Measured complete resident replays | 15 |

Every process passes 27 resident proof gates and 10 live ActivitySim gates.
All 15 replays have zero logsum-bit mismatches and zero final TDD mismatches.
The production bootstrap reads zero dense preprocessor rows and values and
avoids all 1,210,124 dense rows.

The compact persistent input state remains 24,849,394 bytes for 503,411,584
bytes of removed repeated arrays, a 20.258505x reduction. Post-seal modeled
H2D, intermediate modeled D2H, modeled CPU fallback, and retained captured
row pointers are all zero.

## Independent byte-level replication gate

The resident replay compares against the result captured in its own process,
so Phase 30 adds an intentionally separate oracle test. One native process and
one Phase 29 legacy-dense process each download their six generated logsum
vectors after all timed work and hash the exact bytes.

All six per-program hashes match. The aggregate SHA-256 on both sides is:

`41ea4ab90d0b47595a6ad59b1598a050a09a01db5d775dd3d1ad9f5be79e1322`

These deliberate qualification downloads are explicitly excluded from
performance runs and reports.

## Timing result and interpretation

| Metric | Process 1 | Process 2 | Process 3 | Median |
|---|---:|---:|---:|---:|
| Complete resident graph | 0.226712 s | 0.221389 s | 0.231066 s | 0.226712 s |
| Six-program native bootstrap | 4.540633 s | 4.488558 s | 4.867005 s | 4.540633 s |
| Cold checkpoint-to-result | 30.222467 s | 30.779023 s | 30.739373 s | 30.739373 s |

Phase 29's medians were 0.225311 seconds resident and 30.759445 seconds cold.
Phase 30 is therefore 0.622% slower resident and 0.065% faster cold. Those are
noise-scale changes, not material speedups.

The architectural improvement is nevertheless important: the native path no
longer requires the dense preprocessor to exist. Cold time remains dominated
by Python/ActivitySim initialization and especially fresh loading of the
6.452 GB Sharrow skim dataset, observed at roughly 12 seconds per process.

## Arithmetic investigation

The dedicated scheduling compiler now exposes two exponential lowerings while
holding the qualified dot product, pairwise float32 sum, and source-order
inverse-CDF search fixed:

| Policy | Frozen-reference mismatches | Detected ambiguities |
|---|---:|---:|
| CUDA `expf` | 0 | 58 |
| CUDA float64 `exp`, rounded to float32 | 0 | 57 |

Both pass the frozen reference, and every mismatch would be contained in the
detected ambiguity set. A live float64 experiment nevertheless changed the
boundary population and still required one correction. Phase 30 therefore
retains the qualified 57-entry Sharrow decision map on CUDA. It is sparse,
resident, transfers zero boundary bytes, and is honest about its public-data
scope. It must not be removed until a common cross-library exponential and
reduction implementation passes changed-world and boundary-fuzz qualification.

## Assumptions and limits

- The public 50,000-household checkpoint and configuration are the qualified
  model world.
- The utility IR and scalar configuration are reviewed inputs and are hashed.
- Purpose-specific raw-source types are explicit and fail closed.
- Controlled random stream advancement is part of the compatibility contract.
- Immutable skim cubes have the same ActivitySim/xarray meaning as in the
  public run.
- Byte-identical native/legacy logsums prove exact replication for this
  qualified workload, not for every future unqualified model configuration.
- Synthetic changed-world Phase 29 tests remain relevant because Phase 30
  reuses that raw source compiler.
- Complete-model or universal CPU-free speedup is not claimed.

## Next major phase

Phase 31 should attack the real cold bottleneck with a versioned persistent
native skim store. It should prepack only the 149 required physical cubes in a
GPU-ready layout, hash every artifact, support reuse across processes and
scenarios, and fail closed on IR/schema/source mismatch. Qualification must
report cold file-to-GPU time, warm reuse, resident execution, final
publication, memory budget, and exact output separately.

In parallel, a single specified exponential/reduction/search implementation
should be used by both the reference evaluator and CUDA. Boundary-fuzz and
changed-scenario suites must prove it before deleting the 57-entry map.

## Reproduction

Run `scripts/run_phase22_integrated_scheduling.py` three times in fresh
processes with `--native-abi-bootstrap-report` and five resident replays. Run
one proof-only native process and one `--resident-raw-table-input-report`
legacy process with `--qualification-logsum-hash-report`. Then run:

```text
python scripts/qualify_phase30_arithmetic_contract.py ...
python scripts/summarize_phase30_native_bootstrap.py ...
```

The aggregate artifact is
`benchmark-results/phase30-native-bootstrap-summary.json`. Its source-hash map
chains the three native reports, three live reports, independent native and
legacy logsum hashes, legacy resident proof, arithmetic qualification, and
Phase 29 summary.
