# Phase 57: live timetable feasibility and compact GPU scheduling

## Measured outcome

**A replicated improvement, not completion of the under-105-second target.**
All three fresh matched pairs on the public 50,000-household workload pass all
proof gates and preserve every checked published decision. All 238 repository
tests pass, including five new live-scheduling tests against both an independent
oracle and actual upstream ActivitySim functions.

| Measurement | Comparator | Phase 57 | Result |
|---|---:|---:|---|
| Model steps + validation/prewarm, median | Phase 56: 142.075 s | 128.158 s | 1.109x; 9.80% lower |
| Launch-to-exit process wall, median | Phase 56: 151.231 s | 138.263 s | 12.968 s saved |
| Mandatory scheduling, median | Phase 56: 16.5 s | 8.0 s | 2.063x; 51.52% lower |
| Six-batch compiled reduction, median | Best CPU (48 threads): 21.731 ms | GPU: 8.250 ms | 2.634x |

| Pair | Phase 56 charged total | Phase 57 charged total | Saved |
|---|---:|---:|---:|
| 1 | 139.845 s | 128.158 s | 11.687 s |
| 2 | 143.059 s | 123.409 s | 19.650 s |
| 3 | 142.075 s | 129.353 s | 12.722 s |

The strict target still fails by **23.158 seconds** at the replicated candidate
median. The harness correctly exits nonzero for that target, after preserving
the complete measurements. The consolidated status is
`replicated_improvement_target_not_met`, not `target_met`.

Absolute times rose between the development smoke pair (118.339 -> 111.734 s)
and formal replication. Untouched components also varied. The observed 13.917 s
difference between whole-model medians must not all be attributed to the new
kernel: the targeted scheduling difference is 8.5 s, with other differences
reflecting system-level effects and timing variation. Fixed control/candidate
order leaves an order-effect risk. A balanced-order follow-up remains necessary
for a stronger causal whole-model estimate. Do not compare the formal median
directly with historical Phase 56 or CPU timings as if they were matched runs.

### Strong compiled CPU comparison

Seven trials each process the same six input snapshots (81,983 chooser rows).
Every integer output agrees exactly in every mode. CUDA wins against every
tested CPU configuration in every trial.

| Compiled implementation | Median total for six batches |
|---|---:|
| CPU, 1 thread | 347.737 ms |
| CPU, 4 threads | 136.227 ms |
| CPU, 8 threads | 69.762 ms |
| CPU, 24 threads | 28.039 ms |
| CPU, 48 threads | 21.731 ms |
| RTX A4000 CUDA, transfers and allocations included | 8.250 ms |

These are warmed function timings. First calls in the measurement process take
1.318 s for CPU and 0.590 s for GPU; they include initialization and may reuse
on-disk compiler caches, so they are not clean-cache compilation measurements.
This establishes a local advantage over the tested compiled algorithm, not
superiority over every possible CPU algorithm. Most of the multi-second model
gain comes from avoiding massive intermediate tables; the CUDA versus compiled
CPU difference at the compact boundary is only about 13.5 milliseconds.

Machine-readable evidence: `benchmark-results/phase57-p57formal-summary.json`,
`phase57-p57formal-qualification.json`, `phase57-live-pairs-microbenchmark.json`,
and the three `base`, `gpu`, and `exact` reports. Qualification fingerprints the
source and evidence; the microbenchmark fingerprints all eight input files and
records hardware, package versions, thread counts, and all trial times.

## Scope and implementation

The new CUDA kernel consumes the **current** person's timetable and the
configured time-of-day alternative footprints. It checks whether each
alternative overlaps an existing tour, groups feasible alternatives by the
outbound/inbound skim-period pair, and retains the first feasible alternative
in each pair. Integer minimum reductions make the result independent of GPU
thread execution order. No floating-point reduction or approximate overlap
test is introduced.

ActivitySim already deduplicates these alternatives before computing logsums.
Phase 57 performs that reduction before creating the large pandas interaction
table. The six public mandatory-scheduling batches avoid materializing
15,242,743 feasible interaction rows; only 1,210,124 representative rows are
created. All six mode-choice logsum calculations still execute on the GPU
using the current inputs. Trip matrices and summaries are computed afresh.

The period-pair kernel includes host-to-device input copies, allocations, GPU
execution, and compact result downloads. The integration also charges the
remaining host table construction and representative-time lookups. It is a
hybrid runtime, not a GPU-only ActivitySim replacement.

## Assumptions and correctness argument

- The pinned benchmark has five ordered skim periods and positional TDD ids.
  Unsupported period layouts or non-positional identities fail closed.
- Availability uses exactly the upstream timetable's nine colliding bit-code
  pairs. Empty and occupied current windows are tested, not just captured data.
- A representative is selected by minimum original alternative index. Sorting
  these minima restores upstream first-encounter order, including the order of
  pairs within each chooser. Tour index order is preserved.
- Representative times are read from the current `tdd_alt_segments`
  configuration, including purpose-specific rows and the upstream null-purpose
  fallback. Missing or ambiguous representatives are rejected.
- Scope is the six mandatory-scheduling batches. After their completion, the
  original ActivitySim functions handle subsequent scheduling components.
- The integration retains inherited chooser/reference contracts and sparse
  near-boundary adjudication artifacts. This experiment does **not** establish
  unrestricted correctness for a changed population, seed, specification, or
  ActivitySim release. Standalone changed-input tests cover the new reduction,
  not arbitrary end-to-end scenarios.

## Evidence design

1. Exhaustive serial collision oracle versus CUDA on random windows and changed
   windows, with exact integer array comparisons.
2. Actual upstream `TimeTable`, `tdd_interaction_dataset`, and `dedupe_alt_tdd`
   versus the compact path, including changed timetable occupancy and reordered
   tour ids.
3. Three fresh-process Phase 56/57 matched full-model pairs, with all 34 steps,
   all inherited proof gates, and independent final-output verification. Both
   sides include cache validation and prewarm; process wall time is reported
   separately from summed model-step time plus those charges.
4. Seven repeated six-batch kernel comparisons against a compiled Numba CPU
   implementation at multiple thread counts. Every CPU/GPU result must match
   exactly. The strongest measured CPU median is the comparator; GPU timing
   includes transfers and allocations, while compilation/first use is separate.

Pairs run control then candidate, not randomized order; cache/host drift can
affect the whole-model comparison. Three pairs demonstrate local repeatability,
not universal speedup or broad statistical significance. Component times are
recorded at 0.1-second resolution. Never interpret a small difference in an
untouched component as a GPU optimization there.

## Rejected approach: replay is not faster live computation

Early local Phase 57 experiments reused captured logsum or final matrix/summary
results. A 108.48-second experiment was observed, but it answers a different
question: how fast can an unchanged scenario replay saved work? It does not
prove a faster live GPU kernel. The replay modules, runner switches, and saved
logsum manifest were removed from the release candidate. Their measurements
are excluded from qualification. Local ignored output-cache remnants are not
runtime dependencies.

The inherited Phase 56 skim image remains: it caches immutable input data, not
modeled outputs. Its validation and startup costs are charged on both sides.

## Reproduction

Use the existing pinned `.venv-phase8` environment, public MTC full-data project,
Phase 21 input/reference artifacts, and verified Phase 56 skim image described
in the preceding phase reports. This is not a one-command clean-install claim.

```powershell
.venv-phase8\Scripts\python.exe -m pytest tests/test_phase57_live_scheduling.py -q -p no:cacheprovider
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_phase32_full_model_ab.ps1 -Repetitions 3 -RunTag p57formal -Baseline phase56 -CandidatePhase 57 -CleanupOutputs
.venv-phase8\Scripts\python.exe scripts/benchmark_phase57_live_pairs.py
.venv-phase8\Scripts\python.exe scripts/build_phase57_qualification.py
```

Use a fresh run tag/report path when repeating: the measurement scripts protect
existing reports. The full-model harness writes its evidence before failing
the strict under-105-second target. That target and correctness/relative
improvement are separately recorded; passing the latter does not waive the
former. No benchmark runs or other heavy work should overlap these measurements.

## All 34 component medians

Seconds below compare the already GPU-enabled Phase 56 against Phase 57, **not
regular CPU ActivitySim**. Only mandatory scheduling is newly targeted in this
phase. Medians of individual rows need not sum to the median whole-run total.

| Component | Phase 56 seconds | Phase 57 seconds |
|---|---:|---:|
| initialize_landuse | 0.7 | 0.7 |
| initialize_households | 5.6 | 4.9 |
| compute_accessibility | 1.9 | 1.9 |
| school_location | 5.0 | 4.4 |
| workplace_location | 2.5 | 2.3 |
| auto_ownership_simulate | 1.6 | 1.7 |
| free_parking | 1.2 | 1.1 |
| cdap_simulate | 9.1 | 8.2 |
| mandatory_tour_frequency | 1.8 | 1.7 |
| mandatory_tour_scheduling | 16.5 | 8.0 |
| joint_tour_frequency | 1.5 | 1.4 |
| joint_tour_composition | 0.9 | 0.9 |
| joint_tour_participation | 3.0 | 2.9 |
| joint_tour_destination | 2.1 | 2.1 |
| joint_tour_scheduling | 1.9 | 1.7 |
| non_mandatory_tour_frequency | 8.4 | 8.9 |
| non_mandatory_tour_destination | 2.3 | 2.1 |
| non_mandatory_tour_scheduling | 7.7 | 7.6 |
| tour_mode_choice_simulate | 6.5 | 6.2 |
| atwork_subtour_frequency | 1.3 | 1.1 |
| atwork_subtour_destination | 1.4 | 1.4 |
| atwork_subtour_scheduling | 1.9 | 1.9 |
| atwork_subtour_mode_choice | 1.2 | 1.3 |
| stop_frequency | 4.3 | 4.1 |
| trip_purpose | 1.4 | 1.2 |
| trip_destination | 11.2 | 10.7 |
| trip_purpose_and_destination | 0.7 | 0.7 |
| trip_scheduling | 7.4 | 6.9 |
| trip_mode_choice | 12.1 | 11.3 |
| write_data_dictionary | 1.9 | 1.7 |
| track_skim_usage | 0.5 | 0.5 |
| write_trip_matrices | 7.9 | 7.0 |
| write_tables | 2.1 | 2.1 |
| summarize | 6.0 | 5.0 |

## Remaining work toward the original objective

1. Add balanced-order full-model measurements and a fresh pinned regular-CPU
   comparison. Keep the compact compiled-CPU baseline too; avoiding pandas is
   not uniquely a GPU optimization.
2. Profile remaining publication, joins, random generation, and kernel costs
   in trip mode choice (11.3 s), trip destination (10.7 s), and non-mandatory
   frequency (8.9 s). Select the next complete boundary by measured removable
   cost, not by kernel percentage alone.
3. Build the deferred versioned person/tour/trip entity store and remove a full
   CPU-table round trip at that boundary. Do not promise 23 seconds of savings
   from a component whose entire runtime is smaller than that.
4. Replace inherited benchmark-specific scheduling and near-boundary artifacts
   with live, shared-arithmetic/reference generation; qualify changed inputs
   and seeds before claiming general ActivitySim compatibility.
5. Recheck the under-105-second goal with exact decisions, bounded diagnostics,
   no silent fallback, complete charged timing, and independent replication.
