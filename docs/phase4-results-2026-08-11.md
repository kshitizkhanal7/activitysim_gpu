# Phase 4 results: complete ActivitySim mandatory scheduling backend

Date: August 11, 2026

## Outcome

Phase 4 moves ChoiceForge from replay into the real ActivitySim 1.4 workflow.
The backend is selected explicitly with `CHOICE_BACKEND: choiceforge` at the
existing mandatory-tour-scheduling interaction boundary. ActivitySim still
constructs feasible alternatives, calculates mode-choice logsums, owns random
draws, updates previous-tour state, mutates the timetable, and writes pandas
tables.

Three normal cached-Sharrow runs had a mandatory scheduling median of 5.515
seconds. Three normal ChoiceForge runs had a median of 4.925 seconds. That is a
1.120x complete-component speedup. All 9,806 final tour rows matched exactly in
`tdd`, `start`, `end`, `duration`, `destination_logsum`, and
`mode_choice_logsum`.

A second, more controlled checkpoint experiment restored the identical state
immediately after `mandatory_tour_frequency`. Cached Sharrow took a median of
8.599 seconds and ChoiceForge took 8.014 seconds, a 1.073x speedup. All 4,477
mandatory tours matched on every trial. The checkpoint runs are slower because
each fresh process must reconstruct runtime state and caches, but they provide
the tightest matched-input comparison.

## What is inside the timer

The ActivitySim workflow timer includes:

- feasible time-alternative construction;
- representative mode-choice logsum calculation with cached Sharrow;
- seven timetable-dependent scheduling primitives;
- compact array packing;
- ActivitySim-owned random-number generation;
- all host-to-device and device-to-host transfers;
- the fused CUDA expression, logsum-exp, and choice kernel;
- mapping selected positions back to TDD labels;
- previous-tour and timetable updates; and
- pandas table integration.

CUDA context initialization and cached-kernel loading remain inside each fresh
process's component timing. No compilation cost was subtracted to improve the
headline.

## Integration design

The small ActivitySim 1.4 patch adds one model setting and one dispatch:

```yaml
inherit_settings: true
CHOICE_BACKEND: choiceforge
```

`activitysim` remains the default. Unsupported paths such as estimation,
household tracing, explicit interaction chunking, or skim-bearing calls
delegate to ActivitySim's original function. The source-controlled patch is
`integration/activitysim-1.4-choiceforge.patch`; the backend itself is
`src/choiceforge/activitysim_scheduling.py`.

This is an explicit backend, not capture-time monkey-patching. It preserves the
framework's random stream and returns the same pandas choice contract expected
by the existing previous-tour and timetable code.

## The optimization that made Phase 4 succeed

The first correct integration was slower. Profiling the largest batch showed:

- about 4,030 ms evaluating timetable functions through the generic row path;
- about 18 ms packing compact arrays; and
- about 387 ms for GPU setup, transfers, execution, and return.

The timetable has only 21 daily time slots per chooser, but the generic path
revisited it for 642,390 chooser-alternative rows. ChoiceForge now computes
previous-start/end flags, adjacent available windows, and remaining available
periods on a 3,381-by-21 representation, then gathers the needed row values.
The largest-batch timetable stage fell to a three-trial median of 330.579 ms.

## Correctness evidence

| Check | Result |
|---|---:|
| Mandatory scheduling batches | 6 |
| Mandatory tours per run | 4,477 |
| Full final tour rows | 9,806 |
| `tdd` mismatches | 0 |
| `start` mismatches | 0 |
| `end` mismatches | 0 |
| `duration` mismatches | 0 |
| `destination_logsum` mismatches | 0 |
| `mode_choice_logsum` mismatches | 0 |
| Automated tests | 24 passed |

The final comparison treats NaN values as equal only when both sides are NaN.
No rows, difficult random draws, or second-tour cases were removed.

## Performance evidence

Hardware: AMD Ryzen Threadripper PRO 5965WX and NVIDIA RTX A4000 16 GB.
Software: ActivitySim 1.4.0, cached Sharrow, CuPy CUDA, and ChoiceForge.

### Normal full-model runs

| Backend | Trial 1 | Trial 2 | Trial 3 | Median |
|---|---:|---:|---:|---:|
| Cached Sharrow | 5.775 s | 5.502 s | 5.515 s | 5.515 s |
| ChoiceForge | 4.934 s | 4.925 s | 4.888 s | 4.925 s |

Complete-component speedup: **1.120x**, or a 10.7% reduction in mandatory
scheduling time.

### Matched checkpoint runs

| Backend | Trial 1 | Trial 2 | Trial 3 | Median |
|---|---:|---:|---:|---:|
| Cached Sharrow | 8.763 s | 8.599 s | 8.512 s | 8.599 s |
| ChoiceForge | 8.193 s | 8.014 s | 7.818 s | 8.014 s |

Complete-component speedup: **1.073x**, or a 6.8% reduction.

### Largest ChoiceForge batch, median stage time

| Stage | Time |
|---|---:|
| Vectorized timetable primitives | 330.579 ms |
| Compact packing | 15.753 ms |
| ActivitySim RNG | 17.418 ms |
| GPU initialization/transfers/kernel/return | 394.956 ms |
| Result mapping | 0.194 ms |
| Total ChoiceForge boundary | 757.152 ms |

The largest batch contains 3,381 choosers, 642,390 feasible rows, and 22.033 MB
of compact input. Mode-choice logsums remain upstream and dominate much of the
full component, which explains why a 3.83x Phase 3 kernel-boundary win becomes
a 1.07x-1.12x complete-component win.

## Reproduction

Apply the tracked ActivitySim 1.4 patch, install ChoiceForge editable in the
Python 3.11 environment, and place the ChoiceForge model-settings overlay
before the normal settings directory. The full-run command is:

```powershell
.venv-asim\Scripts\activitysim.exe run `
  -c benchmark-data\configs_choiceforge `
  -c benchmark-data\prototype_mtc\prototype_mtc\configs_sharrow `
  -c benchmark-data\prototype_mtc\prototype_mtc\configs `
  -d benchmark-data\prototype_mtc\prototype_mtc\data `
  -o benchmark-data\prototype_mtc\prototype_mtc\output_phase4_final
```

Regenerate the evidence summary:

```powershell
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase4_activitysim_component.py
.venv-asim\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Machine-readable timings and mismatch counts are in
`benchmark-results/phase4-summary.json`.

## Honest interpretation and next step

Phase 4 proves a narrow but practical result: on this public ActivitySim model
and this RTX A4000, a configured GPU backend makes the complete mandatory tour
scheduling component faster than cached Sharrow while preserving every tested
output exactly. It does not prove the whole ActivitySim run is 1.12x faster or
that every model component benefits from a GPU.

Phase 5 should reuse the backend for non-mandatory, joint, and at-work tour
scheduling. That increases the share of total runtime under acceleration and
tests whether the expression compiler and timetable optimization generalize.
After that, large destination choice needs tiled online logsum-exp and
GPU-resident skim data.
