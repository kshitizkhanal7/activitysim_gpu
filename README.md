# ChoiceForge

ChoiceForge is a correctness-first GPU prototype for ActivitySim-style discrete
choice and logsum calculations. It uses open-source NumPy and CuPy and currently
contains:

- an independent NumPy reference implementation;
- a CUDA utility-to-choice/logsum kernel;
- a fused linear-utility CUDA kernel that avoids a global `N x A` utility table;
- a ragged real-data interaction kernel for mandatory tour scheduling;
- a single-launch segmented destination kernel for ragged purpose batches;
- a fused FP64 CUDA reducer for the canonical 21-mode nested-logit tree;
- a safe compact-expression compiler that emits fused scheduling CUDA kernels;
- an explicit configured ActivitySim backend for four tour-scheduling components;
- a vectorized 21-period timetable primitive path;
- fused serial and parallel Numba CPU baselines using the same contract;
- explicit caller-owned random draws for ActivitySim reproducibility;
- a hashed strict IR and normative CPU utility evaluator;
- a strict CUDA C++ generator from that same IR;
- an exact, first-divergence Sharrow comparison gate;
- a fail-closed GPU-native state runtime with transfer/fallback telemetry;
- a fail-closed compact chooser/slot/CSR input reconstructor for sealed CUDA graphs;
- entity-stable, partition-invariant GPU random streams;
- deterministic ordered GPU group aggregation;
- CPU/GPU correctness tests;
- a transfer-inclusive and GPU-resident benchmark harness;
- ActivitySim fallback handling and a documented integration roadmap.

The Python 3.11 ActivitySim/CUDA integration suite passes 131 tests, including
exact comparison with ActivitySim's actual Numba `choice_maker` and multi-warp
CUDA regression cases at 33 and 190 alternatives, canonical execution of all
379 MTC utility terms across 21 alternatives, and an independent scalar check
of the strict float32 utility accumulator. The generated strict CUDA path also
matches the CPU oracle exactly on numeric edge cases and 30 real public batches.

## Why this target

Activity-based models repeatedly evaluate utility expressions for many choosers
and alternatives. Interaction models can create very large intermediate tables.
The project tests whether utility evaluation, stable logsum-exp, and inverse-CDF
choice can be fused into a single GPU operation while preserving model results.

## Setup

The GPU extra currently targets NVIDIA CUDA 12. CuPy 14 needs CUDA runtime
headers for runtime compilation, so the extra installs its `ctk` dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gpu,test]"
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the benchmark:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_linear_choice.py `
  --choosers 100000 --alternatives 32 --features 16 `
  --output benchmark-results\local.json
```

## Python API

```python
import numpy as np
from choiceforge.cuda_backend import CudaChoiceBackend

n_choosers, n_alternatives, n_features = 100_000, 32, 16
rng = np.random.default_rng(7)
x = rng.normal(size=(n_choosers, n_features)).astype("float32")
beta = rng.normal(size=(n_alternatives, n_features)).astype("float32")
constants = np.zeros(n_alternatives, dtype="float32")
draws = rng.random(n_choosers, dtype="float32")

result = CudaChoiceBackend().linear_choice(x, beta, constants, draws)
print(result.choices, result.logsums)
```

Random draws are supplied by the caller. In a production ActivitySim backend,
they must come from ActivitySim's random-number manager rather than a GPU RNG.

## Project status

This is a research prototype, not a complete ActivitySim replacement. Phase 9A
adds public full Prototype MTC geography: 2,875,192 available households,
7,566,527 persons, and 1,454 zones. A local 50,000-household, 1,454-zone
fresh A/B pair keeps seven substantive final CSVs byte-identical while reducing
trip destination from 39.2 to 28.0 seconds (1.400x observed). The whole model
falls from 198.794 to 187.010 seconds (1.063x observed). These are one-pair
scale-gate observations, not median superiority claims; the full population
needs a high-memory host and interleaved repetitions. Phase 8
ports the opt-in integration to pinned current ActivitySim commit
`16ab11180a26912987eb902daf945e268f3efc11` and runs the official public
`sharrow-contrast/mtc_mini` workflow at 50,000 households and 190 zones. In
fresh-process A1/B1/A2/B2/A3/B3 trials, median whole-model time falls from
147.225 to 131.812 seconds (1.117x), the four scheduling components fall from
30.1 to 23.7 seconds (1.270x), and trip destination falls from 30.9 to 21.8
seconds (1.417x). All seven substantive final CSVs are byte-identical in all
six runs, covering 111,130 persons, 142,761 tours, and 350,751 trips.

Phase 7
batches repeated trip-destination preprocessing by trip number and adds a
transfer-inclusive CUDA nested-logit reduction. On 30 captured real batches,
the conservative 31-trial reducer median is 120.737 ms versus 486.626 ms for
ActivitySim's pandas implementation (4.030x), with maximum absolute logsum
error of 3.6e-15. In six
interleaved full-model trials, `trip_destination` is 1.185x faster and all 34
steps are 1.044x faster than the Phase 6 backend; all seven substantive final
CSVs remain byte-identical in every run.

Phase 5 uses one explicit backend for mandatory, joint, non-mandatory, and
at-work tour scheduling. Across three full-model runs, their combined workflow
time falls from 8.045 to 6.325 seconds: 1.272x faster. See
[Phase 9 full-MTC results](docs/phase9-full-mtc.md),
[Phase 8 results](docs/phase8-results-2026-08-11.md),
[Phase 7 results](docs/phase7-results-2026-08-11.md),
[Phase 6 results](docs/phase6-results-2026-08-11.md),
[Phase 5 results](docs/phase5-results-2026-08-11.md),
[Phase 4 results](docs/phase4-results-2026-08-11.md),
[Phase 3 results](docs/phase3-results-2026-08-11.md),
[Phase 2 results](docs/phase2-results-2026-08-10.md),
[design and limitations](docs/design.md), and the
[benchmark evidence protocol](docs/benchmarking.md). The first measured sweep is
recorded in [preliminary results](docs/results-2026-08-10.md), the strong CPU
and warm Sharrow comparison is in
[Phase 1 results](docs/phase1-results-2026-08-10.md), and the pinned framework
setup is in [ActivitySim integration](docs/activitysim-integration.md).
## Phase 10 precision guard

Phase 10 adds a scheduling precision guard that snapshots and restores
ActivitySim's controlled random-stream offsets before a shadow comparison. It
returns ActivitySim's result on any GPU mismatch, preventing boundary-rounding
differences from changing downstream choices. The full-geography 1,001-household
smoke run has 14 zero-mismatch scheduling batches and byte-identical outputs.
A 50,000-household guarded run then found three real one-choice GPU/CPU boundary
differences, all safely returned to ActivitySim; the conservative memory cap
kept memory bounded but was far too slow for a performance claim. The repeatable
100,000-household gate requires a high-memory host. See
[Phase 10 precision guard](docs/phase10-precision-guard.md).

## Phase 11 reproducible destination result

Phase 11 upgrades the full-geography destination result to three interleaved
fresh-process 50,000-household A/B pairs. All six runs have byte-identical
substantive final outputs. Median whole-model time improves from 202.492 to
190.380 seconds (1.064x), and trip destination improves from 39.7 to 28.6
seconds (1.388x). The integration patch, environment lock, config hashes, and
GPU telemetry are now recorded for reruns. See
[Phase 11 reproducible destination result](docs/phase11-reproducible-destination.md).

## Phase 12 device utility foundation

Phase 12 adds a strict float64 GPU ABI for already-lowered trip-mode utility
features and keeps its 21-column utility matrix on the device through the
MTC-21 nested-logsum reduction.  A fixed 250,000-row, 64-feature
microbenchmark is 8.114x faster than its NumPy reference while passing a
`1e-11` logsum equivalence gate. A reviewed AST compiler now evaluates all 253
unique expressions in the public MTC trip-mode specification on NumPy and
CuPy; unknown syntax fails closed. A dense-`SkimDict` CUDA adapter now covers
2D and time-stacked 3D directional gathers while retaining ActivitySim's zone
mapper and rejecting sparse MAZ layouts. This is deliberately not yet an
ActivitySim end-to-end result: captured-batch equivalence and an opt-in
call-site are still required, with CPU fallback. See
[Phase 12 device utility foundation](docs/phase12-device-utility-foundation.md).

## Phase 13 strict CPU reference

Phase 13 completes the normative CPU answer key for future generated backends.
The original IR version 2 fixed float64 expression arithmetic, one float32 feature and
coefficient cast, separate source-ordered float32 multiply/add operations,
IEEE round-to-nearest-even, no FMA contraction, and no fastmath. The evaluator
fails closed on a changed policy or hash, unresolved coefficients, and invalid
shapes.

A public 1,001-household full-geography ActivitySim run compared 30 real
Sharrow batches: 85,126 rows, 32,262,754 feature cells, and 1,787,646 utility
cells. Every batch evaluated all 379 terms and 21 alternatives while Sharrow
remained authoritative. The diagnostic reports localize current Sharrow
differences into expression-policy and ordered-accumulation stages. This is a
correctness milestone, not a new speed claim. Phase 14 has since generated CUDA
from the revised IR and matched the strict CPU arrays exactly. See
[Phase 13 strict CPU reference](docs/phase13-strict-cpu-reference.md).

## Phase 14 strict CUDA generator

Phase 14 generates CUDA C++ directly from strict IR version 3. The revision
adds an explicit float32 subnormal flush-to-zero rule discovered while testing
the deployed NVIDIA runtime; the CPU oracle applies the same rule. Separate
multiply/add operations, source order, round-to-nearest, disabled fastmath, and
NaN/infinity preservation remain part of the hashed contract.

On a public 1,001-household full-geography ActivitySim run, all 30 captured
trip-mode batches passed exactly: 32,262,754 of 32,262,754 feature cells and
1,787,646 of 1,787,646 utility cells matched bit for bit across 85,126 rows,
379 terms, and 21 alternatives. Sharrow remained authoritative. The full suite
passes 69 tests, including subnormal and device-resident nested-logsum handoff
coverage. This closes the cross-device correctness gate but is not a new
production speed claim. See
[Phase 14 strict CUDA generator](docs/phase14-strict-cuda-generator.md).

## Phase 15 device-resident candidate

Phase 15 connects strict generated CUDA utilities to the real MTC-21 nested
logsum path with zero utility download and zero reducer re-upload. A compact
ABI passes cached skim cubes plus ActivitySim's mapped OD/time indices directly
to the kernel. On the public 1,001-household full-geography workload, 30/30
real batches match the strict CPU oracle exactly: 32,262,754 feature cells and
1,787,646 utility cells. Three direct Phase 11-versus-Phase 15 pairs improve
trip destination from 11.3 to 10.3 seconds (1.097x), with every candidate below
every baseline and exact modeled decisions.

The scale gate rejects promotion. At 50,000 households, one diagnostic pair is
exact but slower: trip destination is 33.3 versus 28.4 seconds and the whole
model is 197.871 versus 192.519 seconds. Phase 11 remains the supported result;
Phase 15 is opt-in research code with Sharrow fallback. See
[Phase 15 device-resident candidate](docs/phase15-device-resident-candidate.md).

## Phase 16 locality and FP32 compiler

Phase 16 recovers the large public benchmark at the GPU-kernel/component
boundary. Scalar and dense-input compaction, grouped skim indices, and cached
IR/skim bindings reduce setup and transfer work. An explicit FP32 expression
policy then reduces the generated utility kernel from 4.107 seconds for the
strict-locality FP64 candidate to 1.2-1.6 seconds across 4,188,312 utility rows.

Three fresh interleaved 50,000-household pairs improve median trip destination
from 28.5 to 27.8 seconds (1.025x); all three candidates beat all three
baselines, and every modeled decision matches. The component promotion gate
passes. Whole-model median time is 194.312 versus 193.014 seconds (0.993x), so
the separate whole-model gate remains failed. Cooperative tiles and two sparse
coefficient lowerings were exact but slower and remain disabled experiments.
See [Phase 16 locality and FP32 compiler](docs/phase16-locality-and-fp32-compiler.md).

## Phase 17 persistent execution and trip-mode continuation

Phase 17 turns the generated kernel into a reusable execution backend. A
schema-checked compiled plan retains the generated source, CUDA kernel,
coefficients, and ABI. Changed scalar values are allowed, but changed input
roles, types, aliases, or skim ranks fail closed and build a different plan.
The destination candidate now continues into trip mode choice so bypassing
Sharrow does not merely move its cold compilation cost to a later model step.

The 1,001-household qualification is exact on all 30 real batches:
32,262,754/32,262,754 feature cells and 1,787,646/1,787,646 utility cells.
All trip decisions match the frozen reference; destination and mode-choice
logsums differ by at most 0.000008 and 0.00000191, below their declared gates.

Five fresh interleaved 50,000-household pairs improve median trip destination
from 28.4 to 27.3 seconds (1.040x). Every candidate destination run beats every
baseline, the bootstrap median saving interval is 0.8 to 1.9 seconds, and the
component gate passes. Whole-model median improves from 191.474 to 190.307
seconds (1.006x), but two pairs are slower by 0.042 and 0.038 seconds and the
95% bootstrap interval includes -0.042 seconds. This is encouraging evidence,
not the repository's strict whole-model superiority claim.

Reusable device/output workspaces are implemented behind an opt-in switch and
exactly qualified. A diagnostic 50,000-household run reduced measured
destination pack-plus-upload work by about 107 ms, but did not establish an
independent whole-model win, so reuse remains experimental. See
[Phase 17 persistent execution](docs/phase17-persistent-execution.md).

## Phase 18 GPU-native runtime foundation

Phase 18 moves beyond one accelerated component and proves a model-shaped chain
that remains on the GPU after one input upload: public-data feature
construction, entity-stable random generation, two dependent 21-alternative
choices, and deterministic zone aggregation. A sealed runtime rejects modeled
host arrays, device downloads, and CPU fallbacks.

On all 2,875,192 households in the public Prototype MTC table, nine measured
runs give a 0.031357-second GPU modeled-compute median versus 0.457023 seconds
for fused parallel Numba: **14.575x faster**. Including the permitted input and
output transfers, the GPU median is 0.055327 seconds, or **8.260x faster**.
Running the same workload as one table or deterministic 250,000-household
partitions produces bit-identical GPU choices and logsums.

The CPU/GPU comparison is explicitly numerical rather than falsely bit-exact:
one first-stage choice and three dependent choices differ among 2.875 million
rows because CUDA and Numba do not promise identical exponential/FMA
arithmetic. Those observed rates pass the published one- and two-per-million
gates. The benchmark uses real public household rows but synthetic coefficients,
so this is a systems result, not a calibrated forecast or a complete GPU
ActivitySim model.

The live capacity audit also shows why the final architecture needs a hot-skim
cache and deterministic population partitions: the 826 uncompressed skim
arrays total 13.389 GiB on a 16,376 MiB RTX A4000. See
[Phase 18 GPU-native runtime](docs/phase18-gpu-native-runtime.md),
[full-household qualification](benchmark-results/phase18-gpu-native-full-households.json),
and [capacity audit](benchmark-results/phase18-capacity-audit.json).

## Phase 19 calibrated household-to-person chain

Phase 19 replaces the synthetic Phase 18 equations with the public Prototype
MTC Extended auto-ownership and mandatory-tour-frequency specifications. The
GPU evaluates 127 published expressions, reproduces ActivitySim's per-entity
MT19937 random draws bit for bit, makes the calibrated household choice, joins
that new result into person state by `household_id`, and makes the dependent
person choice without a modeled CPU fallback or intermediate transfer.

On the 50,000-household public checkpoint (132,536 persons; 78,900 mandatory
choosers), both GPU output columns match the saved ActivitySim checkpoints
exactly: **zero choice mismatches**. Expression features and random draws are
bit-exact; maximum utility and probability errors are `1.776e-15` and
`4.441e-16`. Across nine measured runs on the RTX A4000, median GPU compute is
0.025713 seconds versus 0.458724 seconds for the independent CPU replay,
**17.840x faster**. Including one ingress and final egress, the GPU median is
0.037997 seconds, **12.073x faster**.

This result's scope is precise: upstream school/work location and CDAP state
are frozen public checkpoint inputs. Phase 20 has now ported its downstream
mandatory-tour row creation. See
[Phase 19 calibrated chain](docs/phase19-calibrated-chain.md) and the
[qualification artifact](benchmark-results/phase19-calibrated-chain.json).

## Phase 20 variable-length tours and calibrated scheduling

Phase 20 turns the 78,900 exact mandatory-frequency choices into 81,983 real
tour rows on the GPU. Every value in all 12 ActivitySim tour columns, including
stable 41-channel tour IDs, work/school ordering, destinations, and the
non-worker scheduling swap, matches the public checkpoint. The fused row builder
is **11.496x faster** with resident inputs and **6.272x faster** including input
upload and `tour_id` download.

Those IDs link exactly to a fresh full-population capture of calibrated
mandatory-tour scheduling: six batches and 15,242,743 real feasible
tour-time rows. A precision gate exposed and fixed float32 random-draw rounding
in the older scheduling kernel. The corrected GPU reproduces all 81,983 saved
TDD choices with zero mismatches and bit-repeatable results. It is **18.097x
faster** than the independent CPU compiler with resident compact inputs and
**2.935x faster** including compact upload and choice/logsum download.

This is a scheduling-kernel replay, not yet a whole scheduling-component claim:
ActivitySim still prepares time-dependent mode-choice logsums, timetable
primitives, and feasible alternatives on the CPU. Phase 20 also adds exact
arbitrary MT19937 offsets and a hash-complete device checkpoint/audit manifest.
See [Phase 20 variable tours and scheduling](docs/phase20-variable-tours-and-scheduling.md),
the [qualification artifact](benchmark-results/phase20-tour-chain.json), and
the [checkpoint manifest](benchmark-results/phase20-device-checkpoint.json).

## Phase 21 GPU scheduling preparation

Phase 21 moves mandatory-scheduling preparation onto the GPU. A device-resident
21-period timetable now filters all 190 TDD alternatives, builds exact CSR
rows, generates all seven stateful timetable primitives, gathers a compact
5-by-5 skim-period logsum cache, makes each calibrated choice, and mutates the
timetable before the next of six sequential batches.

On the public 50,000-household checkpoint, both CPU and GPU regenerate all
15,242,743 captured feasible rows exactly and select all 81,983 saved TDDs with
zero mismatches. Nine runs give a **10.199x** resident speedup and an **8.680x**
primitive-transfer-inclusive speedup versus compiled parallel Numba. The
primitive ABI is 12,688,620 bytes, **40.896x smaller** than the captured
prepared-row arrays.

A separate live ActivitySim gate reads real raw network skims and executes all
six mode-logsum batches on generated CUDA: 1,210,124 rows, zero fallbacks, zero
utility-to-nest host handoff bytes, and zero changed TDD/start/end outputs. A
Sharrow-compatible fused float32 utility policy and an ActivitySim-compatible
mixed-precision nest reducer were required to close one real random-boundary
case; the strict no-FMA compiler default remains unchanged.

The 8.680x-to-10.199x claim starts from the compact logsum cache. The live
raw-skim run is a correctness/integration proof, not a repeated speed claim,
and ActivitySim still owns pandas orchestration. See
[Phase 21 GPU scheduling preparation](docs/phase21-gpu-scheduling-preparation.md),
the [nine-run qualification](benchmark-results/phase21-scheduling-pipeline.json),
and the [live proof](benchmark-results/phase21-logsum-live2.json).

## Phase 22 continuous raw-skim-to-schedule path

Phase 22 joins the raw-skim CUDA logsum engine to the GPU timetable scheduler
without materializing the bulk modeled-logsum cache on the host. All six
sequential mandatory-tour batches now run as one live path: raw skims,
generated utility, nested logit, device cache scatter, feasible-alternative
preparation, random choice, and timetable mutation.

Three paired runs on the public 50,000-household checkpoint were exact and GPU
won every pair. CPU times were 42.358, 40.389, and 40.250 seconds; GPU times
were 36.599, 31.963, and 32.030 seconds. The median paired speedup is **1.257x**
(ratio of medians: **1.261x**) with zero differences across 81,983 TDD, start,
or end outputs.

This is not falsely labeled CPU-free. CUDA detected 57 numerically ambiguous
draws per run (0.0695%) without consulting saved answers. The real
ActivitySim/Sharrow arithmetic adjudicated only those rows, transferring
11,400 bytes of logsums; the remaining bulk logsums stayed on the GPU. See
[Phase 22 continuous GPU scheduling](docs/phase22-continuous-gpu-scheduling.md),
the [three-pair proof](benchmark-results/phase22-live-paired-summary.json), and
the [restart checkpoint](benchmark-results/phase22-integrated-checkpoint.json).

## Phase 23 next-generation device-resident runtime

Phase 23 changes the system boundary instead of tuning another isolated
kernel. A versioned, fail-closed CUDA state graph now keeps calibrated auto
ownership, mandatory-tour frequency, variable-length tour construction,
stable ID linkage, six scheduling batches, and timetable mutation resident
across component boundaries. Host arrays cannot be committed after ingress is
sealed; publication and checkpointing are explicit named boundaries.

A persistent device topology compiles static household, zone, person, and tour
joins once. A new fused MNL compiler evaluates all 98 mandatory-frequency
expressions, five utilities, probabilities, logsum, and choice in one CUDA
kernel while consuming the preceding GPU auto-ownership result.

Three independent processes ran nine repetitions each on the public
50,000-household checkpoint. The median modeled result is **24.405x faster**
than the independent CPU chain (0.031456 versus 0.767301 seconds); the slowest
process still achieved **21.555x**. Charging one-time setup and final
publication to a single run remains **1.356x faster** at the median. Ten-run
setup amortization is **9.055x**.

All calibrated choices, all 12 tour columns, all 81,983 TDDs, and the final
timetable match exactly. Every one of 27 measured GPU repetitions is
bit-repeatable, all three self-contained checkpoints restore exactly, and
post-seal modeled transfers and CPU fallbacks are zero. The compact scheduling
logsum cache and upstream location/CDAP state remain named ingress boundaries;
this is a calibrated multi-component vertical slice, not yet a whole model.

See the [Phase 23 technical report](docs/phase23-device-resident-runtime.md),
the [three-process qualification](benchmark-results/phase23-device-resident-summary.json),
and the [primary evidence](benchmark-results/phase23-device-resident.json).

## Phase 24 budgeted resident hot-skim cache

Phase 24 builds the raw-network-data layer required by the next resident
mode-logsum stage. The reviewed 315-term tour-mode IR discovers 209 logical
skim bindings instead of relying on a hand-maintained list. Directional aliases
are deduplicated, leaving 149 physical float32 cubes and a 6,378,932,500-byte hot
set under an explicit 8 GiB budget on the 16 GiB RTX A4000.

The public proof covers 1,204,594 valid mandatory-scheduling OD/period rows and
reads all 209 logical bindings for each row: 251,760,146 raw skim reads per
run. Independent CPU and CUDA implementations produce two exact bit hashes per
row. Three processes and 15 measured GPU repetitions have zero CPU/GPU hash
mismatches, zero repeat mismatches, zero post-seal modeled transfers, and zero
CPU fallbacks.

The replicated middle resident cache-layer speedup is **193.114x**; the
conservative minimum is **82.323x**. Charging the complete one-time 6.38 GB
upload plus one compute and publication to one run still gives a **1.813x**
middle speedup; ten-run amortization is **16.702x**. These are deliberately
cache-layer measurements, not full model or utility/logsum claims. The Phase
23 precomputed 5-by-5 scheduling-logsum ingress remains until the real strict
utility and nested-logit plan consumes these cubes in the next phase.

See the [Phase 24 technical report](docs/phase24-resident-hot-skim-cache.md),
the [three-process qualification](benchmark-results/phase24-resident-skim-cache-summary.json),
and the [primary evidence](benchmark-results/phase24-resident-skim-cache.json).

## Phase 25 resident expression-to-logsum runtime

Phase 25 connects those resident skims to the real public tour-mode equations.
Each of six sealed programs reuses its compiled CUDA kernel, coefficients,
dense chooser inputs, OD/time coordinates, 149 shared skim arrays, utility
workspace, nested-logit reducer, and a precompiled GPU cache-scatter plan. The
timed path accepts no precomputed scheduling logsum and performs no host layout
work or bulk modeled download.

The public workload contains 1,210,124 rows per replay, 315 expression terms,
21 alternatives, 209 logical skim bindings, 381,189,060 term evaluations, and
252,915,916 logical skim reads. Three fresh processes ran five measured replays
each. Their medians were 0.169039, 0.169739, and 0.169423 seconds; the
cross-process median is **0.169423 seconds** and the slowest is **0.169739
seconds**. All 15 replays are bit-identical. Relative to each process's initial
live CUDA setup-and-execution path, the resident speedups are 10.389x, 9.655x,
and 9.475x (**9.655x median**).

The paired live ActivitySim runs also end with zero TDD, start, or end
differences. This is not called CPU-free: ActivitySim still prepares the dense
batches, and the complete live scheduling path still explicitly adjudicates
57 near-boundary choices (11,400 logsum bytes). The 9.655x figure compares
resident execution with initial CUDA setup/execution, not CPU or whole-model
time.

See the [Phase 25 technical report](docs/phase25-resident-expression-runtime.md)
and the [three-process qualification](benchmark-results/phase25-resident-raw-skims-summary.json).

## Phase 26 sealed raw-skim-to-timetable graph

Phase 26 connects the Phase 25 producer directly to device-generated
scheduling rows, choices, and timetable mutation inside one versioned resident
stage. Three independent 50,000-household processes and 15 measured replays
produced bit-identical logsums and all 81,983 TDDs exactly at a 0.200852-second
cross-process median. The former 57-row CPU resolver is replaced by an explicit
qualified Sharrow decision map resident on the GPU: all 57 rows stay on-device,
one requires correction, and zero boundary bytes are downloaded.

See the [Phase 26 technical report](docs/phase26-resident-raw-skim-to-timetable.md)
and the [hash-chained qualification](benchmark-results/phase26-resident-schedule-summary.json).

## Phase 27 compact input reconstruction

Phase 27 removes 503,411,584 bytes of captured row-dense chooser inputs and
skim coordinates from the timed sealed graph. A fail-closed compiler represents
them with 25,042,522 bytes of constant, per-chooser, exact-slot,
chooser-response-pattern, and CSR state—a 20.102x reduction—and CUDA rebuilds
the exact ABI before running the complete Phase 26 chain.

Across three fresh 50,000-household processes, the matched reconstruction
boundary is 0.002915 seconds on CUDA versus 0.491203 seconds with NumPy
(168.52x). The complete compact-input-to-timetable graph is 0.205337 seconds,
only 2.23% above Phase 26. All 15 measured full replays have bit-identical
logsums and exact final TDDs, with no captured row pointer, post-seal modeled
transfer, or CPU fallback. ActivitySim still supplies the dense arrays once
during qualification; direct upstream production of the compact factors is
the next boundary.

See the [Phase 27 technical report](docs/phase27-compact-input-reconstruction.md)
and the [hash-chained qualification](benchmark-results/phase27-generated-input-summary.json).

## Phase 28 named semantic input generation

Phase 28 replaces all anonymous chooser-response dictionaries with declared
CUDA formulas. `daily_parking_cost` is generated from a compact per-tour rate
and exact duration; 14 mode-availability columns are generated directly from
resident raw skims, compact coordinates, and auto ownership. Unknown response
sources fail closed.

Three fresh 50,000-household processes and 15 measured full replays produced a
0.211799-second median complete graph. All 1,210,124 logsum rows were
bit-identical and all 81,983 final TDDs exact. Compact state is 20,258,882
bytes—a 24.849x reduction from captured rows and 19.102% smaller than Phase 27.
Five changed synthetic populations/skims exercised all 15 formulas across
8,000 additional rows with exact results and distinct hashes.

The semantic graph is 3.147% slower than Phase 27 because it performs real raw
skim gathers instead of dictionary lookups. ActivitySim dense inputs are still
used before sealing to qualify non-response factors and recover parking rates;
direct raw-table production remains the next boundary.

See the [Phase 28 technical report](docs/phase28-semantic-input-generation.md),
the [hash-chained public qualification](benchmark-results/phase28-semantic-input-summary.json),
and the [changed-scenario qualification](benchmark-results/phase28-changed-scenario-qualification.json).

## Phase 29 declared raw-table source compiler

Phase 29 removes dense preprocessor rows as an input-plan construction source.
Each of the six real programs now declares 57 sources from the one-row-per-tour
ActivitySim table, land use, controlled stochastic inputs, alternative slots,
or resident skims. Parking rates come directly from `land_use.PRKCST` plus the
tour's free-parking fact. All 18 road and transit availability fields are
regenerated by CUDA formulas, including four fields that happened to be
constant in the Phase 28 public run.

Three fresh 50,000-household processes and 15 full resident replays produced a
0.225311-second median raw-table-input-to-calendar graph. Every bit of all
1,210,124 logsums and all 81,983 final TDD labels remained exact. The compiler
read **zero bytes** from the dense oracle while constructing its plans.

The direct, scenario-capable representation keeps 24,849,394 bytes, a 20.259x
reduction from the 503,411,584 removed row arrays. It is 22.659% larger than
Phase 28 because facts that were accidentally constant in the public sample
are now retained per tour so they may change in a new scenario. The complete
graph is 6.380% slower than Phase 28. Five changed raw-table populations
(10,000 tours) and five changed CUDA skim worlds (8,000 rows) passed their
independent exactness gates.

The legacy ActivitySim run still creates dense rows in the qualification
harness to serve as an independent oracle and to expose the already-compiled
utility ABI. Those rows are not compiler inputs and are absent from the sealed
graph, but eliminating their creation from the cold process requires the next
native schema/IR bootstrap.

See the [Phase 29 technical report](docs/phase29-raw-table-input-generation.md),
the [hash-chained Phase 29 summary](benchmark-results/phase29-raw-table-input-summary.json),
and the [changed-world qualification](benchmark-results/phase29-changed-scenario-qualification.json).

## Phase 30 native strict-ABI bootstrap

Phase 30 removes the dense ActivitySim logsum preprocessor from the live
production bootstrap. Reviewed utility IR, an explicit raw-source type
contract, scalar settings, controlled random draws, and immutable skim-cube
metadata now compile the complete strict CUDA ABI directly. Unknown sources or
wrong-rank skims fail closed. The live path never joins or reads a dense
chooser-alternative frame.

The public 50,000-household proof bypassed **1,210,124 dense preprocessor rows**
in each of three fresh processes. All 15 resident replays kept every logsum bit
and all 81,983 final schedules exact. A separate proof-only legacy process and
native process hashed all six generated logsum vectors; their aggregate
SHA-256 is identical. Three stable purpose-specific IR and ABI hashes make the
compiled contract auditable.

The median complete resident graph is 0.226712 seconds. Median cold
checkpoint-to-result time is 30.739 seconds versus 30.759 seconds in Phase 29,
a 0.065% change that is run noise rather than a speedup claim. Cold startup is
still dominated by loading the 6.452 GB Sharrow skim dataset. The achievement
is an upstream architectural dependency removal with exact replication, not a
new whole-model performance multiplier.

Two explicit CUDA exponential policies were also tested against all six frozen
public scheduling batches. Both produced zero choice mismatches, but the live
generated-logsum path still needs its 57-entry resident boundary map for a
universal fail-closed guarantee. Phase 30 keeps that safeguard and documents
the remaining arithmetic frontier honestly.

See the [Phase 30 technical report](docs/phase30-native-abi-bootstrap.md),
the [hash-chained summary](benchmark-results/phase30-native-bootstrap-summary.json),
and the [arithmetic qualification](benchmark-results/phase30-arithmetic-contract.json).
