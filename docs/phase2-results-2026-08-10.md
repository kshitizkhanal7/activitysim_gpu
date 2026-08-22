# Phase 2 results: deterministic ActivitySim scheduling replay

Date: August 10, 2026

## Outcome

Phase 2 replaces the scheduling-shaped synthetic input with a deterministic
replay captured from ActivitySim 1.4's public `prototype_mtc` example. The new
ragged CUDA kernel accepts ActivitySim's evaluated utility terms, coefficient
vector, feasible-alternative offsets, and random draws. It fuses term
aggregation, stable logsum-exp, and inverse-CDF choice without materializing a
probability table.

All 4,477 mandatory-tour choices across six work, school, and university
batches match ActivitySim exactly. The full captured run also selects the same
`tdd`, `start`, and `end` values as the previously measured warm Sharrow run for
every tour. Floating logsum columns are not byte-identical because the regular
and Sharrow paths use different floating-point evaluation orders.

The performance result is conditional but useful:

- GPU-resident execution is 8.57x faster than the strongest lowered CPU
  boundary on the native largest batch and 19.20x faster at four repeated
  copies of the same real tours.
- transfer-inclusive GPU execution is slower at every measured size. At the
  native size it takes 15.545 ms versus 8.241 ms for the CPU, or 0.53x.
- therefore Phase 2 proves the CUDA arithmetic/choice kernel is superior when
  its inputs are resident, but it does not prove that an isolated ActivitySim
  component is faster end to end. Phase 3 must eliminate or amortize the large
  host-to-device term transfer.

## What is real and what is repeated

The capture script runs the unmodified MTC configuration with ActivitySim's
own random-number manager. At the interaction-simulation boundary it records:

- the numeric value of each active scheduling expression;
- evaluated coefficients;
- the ragged chooser/alternative row grouping;
- ActivitySim's float64 utilities and probabilities;
- ActivitySim's random draws and selected positions; and
- trace labels identifying tour number and segment.

The canonical model contains 4,477 mandatory tours in six batches. The largest
batch is first work tours: 3,381 choosers, 642,390 feasible interaction rows,
190 alternatives per chooser, and 59 active utility terms. Scales 2 and 4 are
made only by repeating this captured batch and its draws. No synthetic feature
values or alternatives are introduced. These repeated scales measure
throughput, not a new behavioral scenario.

## Correctness evidence

| Check | Result |
|---|---:|
| Mandatory batches | 6 |
| Mandatory choosers | 4,477 |
| Exact probability/draw replay mismatches | 0 |
| Lowered fused-GPU choice mismatches | 0 |
| Largest-batch max utility error | 4.37e-7 |
| Largest-batch mean utility error | 8.08e-8 |
| Full-run `tdd`, `start`, `end` mismatches vs warm Sharrow | 0 |
| Automated tests | 17 passed |

The exact probability replay uses ActivitySim's float64 probability matrix and
draws with ChoiceForge's existing float64 sampling kernel. The lowered fused
path uses float32 term values and coefficients. It still produces zero choice
mismatches in all six batches; per-batch utility error is recorded in the raw
summary rather than hidden.

## Performance evidence

Hardware is the Phase 1 machine: AMD Ryzen Threadripper PRO 5965WX (24 physical
cores) and NVIDIA RTX A4000 16 GB. Numbers are medians of nine warm repetitions.
The CPU comparator uses NumPy BLAS to aggregate the term matrix once and a
compiled Numba loop for stable ragged choice. This is substantially stronger
than timing pandas expression evaluation as the CPU baseline.

| Real-batch copies | Choosers | Term input | CPU | GPU incl. transfers | GPU resident | Incl. speedup | Resident speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,381 | 151.6 MB | 8.241 ms | 15.545 ms | 0.961 ms | 0.53x | 8.57x |
| 2 | 6,762 | 303.2 MB | 20.346 ms | 32.298 ms | 1.326 ms | 0.63x | 15.34x |
| 4 | 13,524 | 606.4 MB | 43.035 ms | 62.427 ms | 2.242 ms | 0.69x | 19.20x |

"GPU including transfers" moves the full term matrix, coefficients, offsets,
and draws to the GPU and returns choices and logsums on every repetition.
"GPU resident" preloads inputs, times the kernel, and synchronizes CUDA before
stopping the timer.

## What the lowerer supports

The capture front end lowers the ActivitySim scheduling specification into a
numeric term matrix. The real work specification exercises 59 active terms,
including arithmetic, comparisons, Boolean interactions, person/tour fields,
start/end/duration fields, mode-choice logsums, and timetable-derived values.
Temporary rows used only to feed later expressions are evaluated but omitted
from the coefficient matrix.

This is a deliberate intermediate representation. Stateful timetable
functions and general pandas/Sharrow expressions are still evaluated by the
ActivitySim front end before the fused CUDA kernel runs. Calling this a full
GPU expression compiler would be incorrect. Moving those primitive columns
and timetable operations onto the GPU is the highest-value next step because
the expanded term matrix is 151.6 MB for only 3,381 choosers.

## Reproduction

Use the pinned `.venv-asim` environment described in the integration guide.

```powershell
# Canonical capture. Runs prototype_mtc and writes replay artifacts.
.venv-asim\Scripts\python.exe scripts\capture_phase2_activitysim.py `
  --project benchmark-data\prototype_mtc\prototype_mtc `
  --output benchmark-data\prototype_mtc\prototype_mtc\output_phase2_capture `
  --capture benchmark-results\phase2-replay

# Warm replay benchmark and complete six-batch correctness validation.
$env:PYTHONPATH = "src"
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase2_activitysim_replay.py `
  --repeats 9 --scales 1 2 4

# Test suite.
.venv-asim\Scripts\python.exe -m pytest -q
```

Machine-readable results are in
`benchmark-results/phase2-summary.json`. Captured replay arrays and their trace
manifest are in `benchmark-results/phase2-replay/`.

## Phase 3 decision

The next experiment should not add more arithmetic to this kernel. The proof
already shows resident arithmetic is fast. Phase 3 should generate fused CUDA
expressions directly from primitive chooser and alternative columns, compute
or cache mode-choice logsums on device, and retain compatible arrays across
component boundaries. Its acceptance test should be transfer-inclusive
component time against warm Sharrow on identical captured inputs, with zero
choice mismatches across all six batches.

