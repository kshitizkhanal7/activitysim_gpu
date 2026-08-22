# Phase 3 results: compact compiled ActivitySim scheduling

Date: August 11, 2026

## Outcome

Phase 3 closes the transfer bottleneck found in Phase 2. Instead of sending a
151.6 MB matrix containing every evaluated utility term, ChoiceForge now sends
a compact scheduling ABI and compiles the supported ActivitySim expressions
directly into CUDA source.

On the largest native real-data batch, the compact input is 22.046 MB, an
85.46% reduction. The transfer-inclusive GPU takes 2.478 ms versus 9.481 ms for
a generated 48-thread Numba CPU implementation: a 3.83x speedup. The resident
GPU kernel takes 0.181 ms, a 52.27x speedup. All 4,477 choices in all six real
mandatory-scheduling batches match ActivitySim exactly.

This satisfies the Phase 3 acceptance test at the captured scheduling-kernel
boundary. It does not yet prove that ActivitySim's complete mandatory tour
scheduling component is 3.83x faster, because construction of compact inputs,
mode-choice logsum calculation, timetable updates, and pandas integration are
outside the timed replay boundary.

## Compact input ABI

The largest work batch contains 3,381 choosers, 642,390 feasible interaction
rows, 190 time alternatives, and 59 utility expressions. Its inputs are:

- 11 chooser columns stored once per chooser;
- `start`, `end`, and `duration` stored once for each of 190 alternatives;
- a 16-bit alternative ID for each feasible interaction row;
- mode-choice logsum plus seven stateful timetable-derived primitives per row;
- CSR-style offsets describing each ragged alternative set; and
- ActivitySim's original uniform random draw per chooser.

Pure arithmetic, comparison, and Boolean expressions are not pre-evaluated.
The compiler parses their Python abstract syntax tree, validates every name and
operation, and emits CUDA scalar expressions inside the fused kernel. Unknown
columns, function calls, and unsupported syntax fail clearly during compile.

Stateful timetable functions remain a documented boundary: ActivitySim
evaluates seven such primitives before replay. This is materially smaller than
the full term matrix and keeps Phase 3 achievable without reimplementing the
entire timetable state machine.

## Fused execution

One CUDA block owns one chooser and its ragged alternative tranche. Each lane:

1. loads chooser values once from the chooser table;
2. finds the alternative's `start`, `end`, and `duration` through its compact ID;
3. loads the row-varying logsum and timetable primitives;
4. evaluates the generated utility expressions with embedded coefficients;
5. participates in stable maximum and exponential-sum reductions; and
6. uses ActivitySim's draw to select the first cumulative alternative at the
   probability boundary.

Only choice position and logsum are returned. Utilities, terms, probabilities,
and cumulative probabilities are never written as global GPU tables.

## Correctness evidence

| Check | Result |
|---|---:|
| Mandatory scheduling batches | 6 |
| Real ActivitySim choosers | 4,477 |
| GPU vs ActivitySim choice mismatches | 0 |
| Compiled CPU vs ActivitySim choice mismatches | 0 |
| GPU vs compiled CPU choice mismatches | 0 |
| Max compact-vs-Phase-2 utility difference | 2.29e-5 |
| Max GPU-vs-CPU logsum difference | 2.93e-6 |
| Full-run `tdd`, `start`, `end` mismatches vs warm Sharrow | 0 |
| Automated tests | 21 passed |

The utility difference reflects float32 accumulation order between the
generated scalar expression path and the Phase 2 term-matrix dot product. No
rows or near-boundary draws were removed, and every final choice matches.

## Performance evidence

Hardware: AMD Ryzen Threadripper PRO 5965WX (24 physical cores, 48 Numba
threads) and NVIDIA RTX A4000 16 GB. Values are medians of 15 warm repetitions.
CUDA synchronization is inside every timed call. Compilation and the first
warm call across all six batches took 6.203 seconds and are reported separately.

| Real-batch copies | Choosers | Compact input | Phase 2 input | CPU | GPU incl. transfers | GPU resident | Incl. speedup | Resident speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,381 | 22.046 MB | 151.604 MB | 9.481 ms | 2.478 ms | 0.181 ms | 3.83x | 52.27x |
| 2 | 6,762 | 44.091 MB | 303.208 MB | 18.443 ms | 4.681 ms | 0.271 ms | 3.94x | 68.10x |
| 4 | 13,524 | 88.179 MB | 606.416 MB | 37.132 ms | 9.348 ms | 0.410 ms | 3.97x | 90.65x |
| 8 | 27,048 | 176.355 MB | 1,212.832 MB | 69.735 ms | 17.787 ms | 0.709 ms | 3.92x | 98.33x |

Scale factors repeat captured real tours and their ActivitySim-owned draws.
They are throughput experiments, not synthetic behavioral scenarios.

The CPU comparator is not pandas or a readable NumPy oracle. It is generated
from the same validated expression AST, compiled by Numba, parallelized across
48 threads, and followed by compiled stable ragged choice. Phase 3 therefore
beats a strong purpose-built CPU boundary, not merely ActivitySim's legacy
expression loop.

## Reproduction

```powershell
# Capture compact and Phase 2 validation representations from canonical MTC.
.venv-asim\Scripts\python.exe scripts\capture_phase2_activitysim.py `
  --project benchmark-data\prototype_mtc\prototype_mtc `
  --output benchmark-data\prototype_mtc\prototype_mtc\output_phase3_capture `
  --capture benchmark-results\phase3-replay

# Compile, validate all six batches, and benchmark the largest batch.
$env:PYTHONPATH = "src"
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase3_compact_scheduling.py `
  --repeats 15 --scales 1 2 4 8

# Complete test suite.
.venv-asim\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Raw samples, per-batch errors, schemas, expressions, and environment-relevant
counts are in `benchmark-results/phase3-summary.json`. The six captured replay
files and manifest are under `benchmark-results/phase3-replay/`.

## What Phase 4 should prove

Phase 3 has made the isolated kernel boundary convincingly faster, including
transfers. The next proof should integrate compact packing and result mapping
into ActivitySim as an explicit backend, then measure the complete warm
mandatory scheduling component against warm Sharrow. Mode-choice logsums and
timetable primitives should be cached or generated on device where practical.
Acceptance should continue to require zero choice mismatches for all six
batches and exact final tour time labels.
