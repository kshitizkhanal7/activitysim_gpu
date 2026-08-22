# Phase 5 results: one GPU backend for four scheduling components

Phase 5 broadens the real ActivitySim integration from mandatory tours to the
entire tour-scheduling family in the public 25-zone prototype MTC model:

- mandatory tour scheduling;
- joint tour scheduling;
- non-mandatory tour scheduling; and
- at-work subtour scheduling.

The result is a faster complete scheduling suite with byte-identical final
model outputs. This is the strongest practical evidence in the project so far,
but it is a scheduling-suite claim, not a claim that the entire model is faster.

## Headline result

On an NVIDIA RTX A4000 16 GB, three fresh-process runs of ActivitySim 1.4 gave:

| Complete ActivitySim component | Cached Sharrow median | ChoiceForge median | Speedup | Time reduction |
|---|---:|---:|---:|---:|
| Mandatory | 5.515 s | 5.077 s | 1.086x | 7.9% |
| Joint | 0.671 s | 0.504 s | 1.331x | 24.9% |
| Non-mandatory | 1.560 s | 0.550 s | 2.836x | 64.7% |
| At-work subtour | 0.309 s | 0.202 s | 1.530x | 34.6% |
| **Four-component suite** | **8.045 s** | **6.325 s** | **1.272x** | **21.4%** |

The suite saves a median 1.720 seconds per full model run at this benchmark
scale. The sum is computed inside each run before taking the median; it is not
the sum of independently selected best samples.

## Every measured sample

| Trial | Cached Sharrow suite | ChoiceForge suite |
|---:|---:|---:|
| 1 | 8.388 s | 6.325 s |
| 2 | 8.045 s | 6.341 s |
| 3 | 8.005 s | 6.297 s |

There is no overlap: the slowest ChoiceForge suite trial (6.341 seconds) is
still faster than the fastest cached-Sharrow trial (8.005 seconds), a 1.262x
worst-versus-best advantage. The same complete separation holds for each of
the four components individually.

Each sample comes from ActivitySim's own workflow timers. The boundary includes
upstream mode-choice logsums where configured, timetable-state extraction,
compact packing, ActivitySim-owned random draws, host-to-device and device-to-
host transfers, CUDA execution, pandas result mapping, and timetable mutation.
CUDA context and kernel-load costs remain inside each fresh process.

## Why non-mandatory scheduling improved most

Non-mandatory scheduling has 4,521 tours and five sequential scheduling
batches. Its largest batch contains 3,280 choosers and 501,136 feasible
chooser-alternative rows. That is enough repeated arithmetic to use the GPU
well, while its model omits the expensive mode-choice logsum work that still
dominates much of mandatory scheduling.

The Phase 5 lowerer recognizes simple categorical assignments such as an
"escort tour" flag and compiles their later uses into the fused expression
kernel. Ordinary dataframe arithmetic is also compiled instead of being
materialized as one value per interaction row. Only genuinely stateful
timetable operations stay outside the generated kernel.

Joint scheduling uses a timetable owned by tour IDs rather than person IDs.
The optimizer now discovers that row-owner column from the model expression.
At-work scheduling has a different expression mix, including capped time
arithmetic. Unsupported scalar operations remain in the small, explicit
stateful front end; the fused utility, stable logsum-exp, and sampling path is
still shared.

## Stronger correctness proof

Every Phase 5 trial produced byte-for-byte identical versions of all seven
substantive final CSV files from the cached-Sharrow reference:

- `final_accessibility.csv`;
- `final_households.csv`;
- `final_joint_tour_participants.csv`;
- `final_land_use.csv`;
- `final_persons.csv`;
- `final_tours.csv` (9,806 rows); and
- `final_trips.csv` (23,583 rows).

The benchmark verifies SHA-256 hashes, which is stronger than comparing only a
few schedule columns. `final_checkpoints.csv` is deliberately excluded because
it contains run timestamps and timing metadata. The Python regression suite
also passes 26 tests, including new categorical-assignment, inequality,
joint-tour timetable-ownership, and empty-compact-table cases.

## Whole-model context and the honest boundary

The three cached-Sharrow all-model samples were 61.650, 59.119, and 61.790
seconds (median 61.650). The Phase 5 samples were 63.546, 63.056, and 63.653
seconds (median 63.546). Therefore this experiment does **not** prove a faster
whole model. Unchanged stages account for roughly 55 seconds and varied by
several seconds between the older baseline session and the Phase 5 session,
which is larger than the 1.720-second scheduling gain.

The defensible claim is the timer-owned four-component scheduling boundary:
it is 1.272x faster with identical final outputs. A whole-model claim needs an
interleaved A/B experiment or many more runs to control session-level noise,
and a larger scheduling share or another accelerated component.

## Reproduce the result

Run three clean cached-Sharrow trials and three clean ChoiceForge trials with
the same base data and configuration. The ChoiceForge configuration overlay is
`benchmark-data/configs_phase5_choiceforge`.

One ChoiceForge trial is:

```powershell
.venv-asim\Scripts\activitysim.exe run `
  -c benchmark-data\configs_phase5_choiceforge `
  -c benchmark-data\prototype_mtc\prototype_mtc\configs_sharrow `
  -c benchmark-data\prototype_mtc\prototype_mtc\configs `
  -d benchmark-data\prototype_mtc\prototype_mtc\data `
  -o benchmark-data\prototype_mtc\prototype_mtc\output_phase5_final1
```

Use a different clean output directory for each trial. For the cached-Sharrow
baseline, omit the Phase 5 overlay.

Summarize timings and verify all output hashes:

```powershell
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase5_scheduling_suite.py
```

The machine-readable result is
`benchmark-results/phase5-summary.json`. The script fails rather than publishing
a result if a substantive final-file hash differs.

## What should come next

The highest-value next target is destination choice, especially trip
destination, because the cached profile assigns it about one quarter of total
runtime. That requires tiled or streaming logsum-exp over large sampled
alternative sets. Before broad performance claims, repeat Phase 5 on a larger
public model, a second GPU, and an interleaved A/B run schedule.
