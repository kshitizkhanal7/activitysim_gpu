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
- CPU/GPU correctness tests;
- a transfer-inclusive and GPU-resident benchmark harness;
- ActivitySim fallback handling and a documented integration roadmap.

The Python 3.11 ActivitySim/CUDA integration suite passes 75 tests, including
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
