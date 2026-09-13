# Phase 61: shared live inputs - qualified improvement, target missed

## Final result

All six matched pairs improve on both clocks. The complete-model elapsed
median falls from **91.3255 to 79.2211 seconds**, saving **12.1044 seconds
(13.2541%)** versus the fresh Phase 60 control. The ratio is 1.15279x.
Charged medians fall from 84.6713 to 72.8153 seconds (1.16282x). Candidate
elapsed times are 78.7466-79.6771 seconds.

The 75-second elapsed-median objective and 70-second stretch objective both
fail. The formal status is `replicated_improvement_wall_target_not_met`.
We do not substitute the 72.82-second charged clock to claim a 75-second
elapsed target. The remaining gap is 4.2211 elapsed seconds.

Two fresh regular CPU controls at 48 Numba threads have a 201.1664-second
elapsed median, versus 296.5242 seconds at one thread. The final hybrid is
**2.53930x** faster than the stronger configuration (60.6191% less elapsed
time), and 3.74300x faster than the one-thread configuration. These CPU
controls are separately measured, not additional balanced Phase 60/61 pairs.

| Pair | Run order | Phase 60 elapsed (s) | Phase 61 elapsed (s) |
|---|---|---:|---:|
| 1 | 60 then 61 | 91.738 | 79.677 |
| 2 | 61 then 60 | 91.872 | 78.759 |
| 3 | 60 then 61 | 91.575 | 79.385 |
| 4 | 61 then 60 | 90.931 | 79.057 |
| 5 | 60 then 61 | 91.076 | 79.418 |
| 6 | 61 then 60 | 90.661 | 78.747 |

All modeled-decision comparisons, 115 logical matrices and values in 24
summary reports pass the existing contracts. The four permitted departure-key
spelling differences and diagnostic numerical bounds remain explicit. The
three changed scenarios (10k/seed991, 50k/seed17, 50k/retry7) also pass against
independent regular CPU outputs. No live boundary adjudication or RNG ledger
contract was removed. Source/configuration bytes remained frozen through
the final runs. Full tests passed both before and after: **545 passed**,
92 warnings, in 32.45 and 32.02 seconds respectively.

See [all 34 component timings](phase61-component-comparison.md),
[formal qualification](../benchmark-results/phase61-formal-qualification.json),
[complete comparison](../benchmark-results/phase61-complete-comparison.json),
and [equivalent primitive controls](../benchmark-results/phase61-live-cpu-controls-passive.json).
The final series is `phase58-p61formal-summary.json`; CPU controls and changed
scenarios use the `p61formal` prefixes recorded by the reproduction script.

## Approved objective and gates

Target a complete-model elapsed median of 75 seconds, with a 70-second stretch
target, from Phase 60's qualified 90.16 seconds. These are objectives, not
predicted results. Use fresh matched Phase 60 controls, not the historical
median, to measure the incremental effect.

Priorities are mandatory and non-mandatory scheduling (13.70 seconds combined),
trip destination and trip mode choice (14.20 seconds combined), and tour mode
choice (5.40 seconds). Profile the actual final Phase 60 paths before selecting
changes. Seek live consumers of the existing entity store, shared inputs,
fewer transfers and less repeated construction. Preserve live CPU adjudication
of numerically borderline choices. Do not move model work outside the clocks.

Qualify six balanced complete-model pairs, fresh strong CPU controls and the
three changed scenarios. Require the existing exact-decision, matrix,
summary-report, random-ledger and bounded-diagnostic gates. Compare any new
GPU calculation with an equivalent compiled multithreaded CPU implementation,
including transfers; report CPU wins and unsuccessful experiments. Update the
plain-English explainer and PDF after final verification and publish tested
changes. Do not treat a target miss as permission to weaken correctness.

## Initial measurement

`p61profile1` instruments the Phase 60 candidate, adding complete tour mode
choice to the existing scheduling/trip/CDAP/summary profiles. This run is
diagnostic and cannot be performance evidence. Frozen source/configuration
checks and all output gates remain enabled.

## Implemented development work

- Checked direct access to ordinary in-memory skim cubes, retaining the original
  path for encoded, blended, lazy or unsupported layouts. It reads live mapped
  positions, not saved trip times. Tests include reverse directions, permuted
  dimensions, fixed time, repeated row IDs and upstream error behavior.
- A four-mask timetable representation shared by CUDA and parallel compiled
  CPU implementations. Every query reads current authoritative windows, even
  after mutation or rollback. The GPU path publishes versioned entity columns;
  the CPU path does not upload them. The reviewed domain is at most 32 periods
  with the existing 0/2/4/6/7 collision vocabulary.
- Real consumers of versioned tour/trip columns in generated mode utility
  evaluation. Reuse requires matching dtype and exact live host bytes. Derived
  columns stay live inputs. Trip state is published across destination,
  scheduling and mode-choice boundaries.
- Tour mode choice now uses the existing resident 21-mode pipeline: GPU
  utilities, nested reduction and draws, with the existing live CPU boundary
  adjudication. It does not delete numerical checks or substitute saved answers.
- Compiled CPU batching of standard-normal generation, preserving each
  ActivitySim row seed and offset. Broadcast, scaling and lognormal transforms
  stay in ActivitySim. Unsupported requests retain its original implementation.
- Earlier model steps use the already reviewed keyed GPU uniform generator;
  empty or repeated-identity frames retain the original path.
- Report bins format one representative per observed category; integer
  person/tour pair keys avoid building hundreds of thousands of Python tuples.

These are deliberately hybrid changes. CPU preparation, reuse, batching and
GPU execution must not all be attributed to GPU hardware.

## Development evidence and rejected assumptions

All listed complete development runs passed the independent output gates.
They are not the six-pair qualification and must not replace it.

| Run | Elapsed seconds | Charged seconds | Change |
|---|---:|---:|---|
| p61dev1 | 87.77 | 81.22 | Direct skims, timetable, resident tour modes, entity inputs |
| p61dev2 | 84.54 | 78.32 | CPU normal batching and cross-step trip publication |
| p61dev3 | 80.46 | 73.82 | Earlier keyed uniforms and representative report labels |
| p61dev4cpu | 81.24 | 74.92 | Packed chain IDs and CPU timetable variant |
| p61dev5passivegpu | 95.71 | 89.22 | GPU timetable, passive worker-wait experiment |
| p61dev5passivecpu | 78.90 | 72.32 | CPU timetable, passive worker-wait experiment |

The 75-second objective has not been established by these measurements. A
single faster result is not a replicated median.

The first normal-generator probe produced 58,146 mismatches out of 60,000
values: unused random draws were optimized away, so stream offsets were wrong.
That implementation was rejected. The accepted design returns and checks a
checksum of discarded uniforms, making stream advancement observable. The
revised probe had zero bit mismatches. Expanded tests cover seeds including
2^32-1, odd and even draw counts, long offsets, broadcast and exact ledgers.
An independent capture then matched the original NumPy values bit for bit on
933,292 live row streams in 33 batches. This is a CPU result, not a GPU result.

Initial skim tests also caught an invalid fast-path acceptance: a 2D cube with
an extra time index must retain Sharrow's rejection. That case now falls back
to the original operation. Empty chain-group inputs likewise retain the
original exception rather than silently inventing behavior.

`p61capture1` collected 14 timetable batches (17,794,070 queries), nine tour
mode segments and 33 normal batches. Its profiling and capture work are
explicitly excluded from performance qualification. Captures are input-only
diagnostics for the reducers; normal result values are saved solely for the
independent bitwise check. Production never reads these files.

The first isolated CPU/GPU sweep exposed sensitivity to idle OpenMP worker
spinning between CPU and GPU trials. A second sweep uses `OMP_WAIT_POLICY=PASSIVE`
before initialization to avoid that contention. Both experiments are retained;
only the controlled result should be used for final primitive comparisons.
Primitive timings omit common identity lookup and application bookkeeping,
which remain present in whole-model runs. GPU transfer-inclusive timings
include uploads and result downloads; resident timings do not.

## Selected candidate and scope of the controls

The final candidate uses the 24-thread packed CPU timetable, 48-thread
compiled CPU standard normals, all eight Phase 61 features, and a passive
OpenMP waiting hint. Both CPU operations restore the previous Numba mask.
The earlier 48-thread frequency operation and four-worker CPU matrix writer
remain. Utility scoring, keyed uniforms and the established GPU choice paths
remain GPU work. The alternate GPU timetable is implemented and tested but
is not selected just to increase the count of GPU kernels.

The isolated controlled sweep tests CPU thread counts 1, 4, 12, 24 and 48.
For the 17,794,070 timetable queries, the best transfer-boundary CPU result
is 0.0500 seconds (12 threads), versus 0.0908 seconds for GPU encoding,
upload, calculation and download in the corresponding trial series.
The selected CPU service uses 24 threads (0.0520 seconds in this control),
not a claim that 24 is universally optimal. For already packed/resident
inputs, the 24-thread CPU calculation takes 0.00855 seconds versus 0.00477
seconds on GPU in that series. This modest resident GPU advantage reverses
once live input movement is included. The resident GPU timing also varies
across the sweep; all samples are published rather than choosing its single
fastest result.

The actual tour-mode reducer favors the best compiled CPU comparator:
CPU/GPU ratios are 0.811 for resident inputs and 0.502 including transfers.
Those are reducer-only comparisons, not the full tour-mode component. They
exclude generating utilities and the CPU boundary adjudication from both
implementations. The selected full pipeline keeps its existing GPU reduction
after GPU utility generation, avoiding a newly introduced host round trip;
this is not a claim to have proven it beats every possible CPU-reducer
pipeline. Such a pipeline deserves its own controlled whole-component test.

In the same controlled sweep, the original NumPy normal-generation loop and
compiled CPU batch use identical captured seeds, offsets and requested draw
counts. The compiled 48-thread batch processes the captured calls in about
0.053 seconds, excluding common ledger gather/commit work. All 933,292 row
streams match bit for bit. Complete-model timings, not this isolated ratio,
determine the reported user-facing improvement.

The waiting-policy experiment is empirical, not a universal runtime rule.
The OpenMP standard describes `OMP_WAIT_POLICY` as a hint; implementation
behavior can vary. LLVM also documents active spinning and `KMP_BLOCKTIME`.
We did not change GPU power limits or claim to eliminate every source of
timing variability. See [OpenMP's specification](https://www.openmp.org/spec-html/5.0/openmpse55.html)
and [LLVM's runtime documentation](https://openmp.llvm.org/design/Runtimes.html).

## Reproduction and evidence boundaries

Hardware: AMD Threadripper PRO 5965WX (24 cores/48 threads), NVIDIA RTX A4000
with 16 GB VRAM, Windows. Runtime: the repository's Python 3.11.14 environment,
NumPy 2.4.6, Numba 0.66, CuPy 14.1.1 and Sharrow 2.16.2. ActivitySim is pinned
at `16ab11180a26912987eb902daf945e268f3efc11`; existing integration hooks remain
disabled in regular CPU controls. The benchmark is the public MTC extended
configuration with 50,000 sampled households and 1,454 zones, not all regional
households.

From the prepared repository, `scripts/run_phase61_qualification.ps1` runs
the full tests, six alternating Phase 60/61 pairs, two fresh regular CPU
controls at each of one and 48 Numba threads, three scenario comparisons,
the fail-closed qualifier, complete timing report and final tests. Use a fresh
`-Tag` for a new series: the harness refuses to overwrite an existing run.
It does not delete prior outputs or silently skip failed runs. The script
expects the already independently generated scenario CPU baselines and the
live primitive-control JSON identified in its arguments.

To regenerate those primitive inputs in a separate diagnostic run:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag UNIQUE-capture --phase61 --modes candidate --repetitions 1 --phase61-capture-inputs benchmark-data/UNIQUE-live-inputs
.venv-phase8/Scripts/python.exe scripts/benchmark_phase61_live_inputs.py --inputs benchmark-data/UNIQUE-live-inputs --output benchmark-results/UNIQUE-live-cpu-controls.json
```

Use the resulting control file in `report_phase61_comparison.py` (or update
the reproduction script's control argument for that independent series).
Do not run diagnostics, rendering or tests concurrently with model timing.
Profile runs, input captures and output verification are explicitly outside
performance eligibility. The actual model's preparation, execution,
validation/prewarming, reports and process overhead are not moved out of the
appropriate clocks. Timings use warm compiler/filesystem caches in fresh
processes. This is finite workstation replication, not a clean-install,
cross-hardware or arbitrary-model guarantee.

## Reviewer-facing limits of the shared store

The first final candidate's telemetry shows 19 actual shared-input consumers:
nine tour segments reuse `number_of_participants`, and ten trip segments reuse
`outbound`. Together these consume 1,726,514 matched host bytes. The new store
retains 42,055,332 device bytes and the same amount of host snapshots: it
publishes more numeric columns than these consumers currently need. Departure
publication updates only the changed column (1,770,728 bytes), but that column
is not claimed as a direct utility consumer in this phase.

This is a real, generation-checked connection, not yet a broadly device-resident
model. Its value is a tested integration foundation; the complete speed gain
must not be attributed to that store alone. Demand-driven publication and
additional derived-column/person-table consumers remain unimplemented
opportunities. Each needs its own measured benefit and live-value checks.

## Where a subsequent substantial improvement should come from

The diagnostic profiles still show expensive Sharrow flow/module setup in
household activity coordination and in the live CPU mandatory-scheduling
boundary reference. The latter must remain an independent live calculation:
removing it, replacing its results with saved answers, or silently relaxing
its threshold is not an optimization. A compiled or reusable reference plan
with the same arithmetic semantics is a plausible next target, but is not
implemented or proven by Phase 61.

Trip destination also retains native-input construction, contract compilation
and its fused GPU utility work. Target complete services with profiler-visible
seconds, rather than celebrating milliseconds saved in an isolated reducer.
Further shared-store consumers should be demand-driven: publish the columns
the compiled program actually needs and preserve generation and live-value
checks. Measure that change against leaving the inputs on CPU.

Before expanding claims, independently reproduce the prepared environment,
add cold-start measurements and test additional scales/configurations. The
current finite scenario checks establish replication for the tested cases;
they do not prove universal floating-point equivalence, all ActivitySim
features, all public households, or the accuracy of behavioral forecasts.
