# Phase 21: GPU scheduling preparation and live skim-to-logsum proof

## Outcome

Phase 21 removes the largest CPU-preparation boundary left by Phase 20. The
GPU now owns the 21-period person timetable, tests all 190 tour departure and
duration alternatives, builds the feasible CSR rows, evaluates all seven
timetable-dependent primitives, gathers the correct time-dependent mode
logsum, makes the calibrated choice, and mutates the timetable before the next
tour batch.

On the public 50,000-household Prototype MTC Extended checkpoint, the
qualified workload contains 78,900 timetable people, 81,983 mandatory tours,
six sequential work/school/university batches, and 15,242,743 feasible
tour-time rows. Every regenerated CPU and GPU preparation cell matches the
fresh ActivitySim capture. Every selected TDD, start hour, and end hour matches
the public checkpoint, and a repeated GPU execution is bit-identical.

Nine measured repetitions on the NVIDIA RTX A4000 give:

| Qualified compact-cache boundary | Compiled CPU median | GPU median | Speedup |
|---|---:|---:|---:|
| Inputs already resident | 0.214878 s | 0.021069 s | **10.199x** |
| Primitive upload and final result included | 0.214878 s | 0.024755 s | **8.680x** |

The small reversal between resident and transfer-inclusive medians is ordinary
run-to-run timing noise; each value is the median of its own nine samples. The
claim is that both GPU boundaries beat the same compiled CPU reference by
roughly ten times, not that transfers make computation faster.

Phase 21 also runs the real ActivitySim scheduling mode-logsum calculation
from raw network skims through generated CUDA utilities and CUDA nested-logit
reduction. All six calls use CUDA for 1,210,124 rows with zero fallbacks and no
utility download/re-upload between the two kernels. The resulting 81,983 TDD,
start, and end values match the frozen ActivitySim reference exactly.

The machine-readable evidence is:

- [`phase21-scheduling-pipeline.json`](../benchmark-results/phase21-scheduling-pipeline.json)
- [`phase21-scheduling-pipeline-checkpoint.json`](../benchmark-results/phase21-scheduling-pipeline-checkpoint.json)
- [`phase21-logsum-live2.json`](../benchmark-results/phase21-logsum-live2.json)
- [`phase21-logsum-cpu-control.json`](../benchmark-results/phase21-logsum-cpu-control.json)

## Exact claim boundaries

There are two complementary proofs, with deliberately different timers.

The **performance qualification** starts from a compact 5-by-5 mode-logsum
cache per tour. Prototype MTC has 190 hourly time alternatives, but they map
to only 15 valid outbound/inbound combinations among five network skim
periods. The timer includes:

1. timetable collision tests for all 190 alternatives;
2. feasible-row count, prefix sum, and CSR alternative construction;
3. exact `end_previous` derivation;
4. all seven ActivitySim timetable primitives;
5. mode-logsum cache gathering;
6. the calibrated scheduling utility and probability calculation;
7. the float64 random-draw traversal;
8. chosen-TDD extraction; and
9. sequential timetable mutation across all six batches.

It does not include creating the 5-by-5 cache from raw network skims. The
reported ActivitySim component time of 23.338 seconds is context only and is
not used as a speedup denominator, because ActivitySim includes that upstream
work plus dataframe and workflow orchestration.

The **live integration proof** covers that named upstream boundary. ActivitySim
still assembles pandas chooser frames and manages the overall workflow, but
ChoiceForge reads real OD, time-dependent, reverse-OD, and round-trip skim
bindings, evaluates the public 315-term-by-21-alternative mode specification on
CUDA, and hands the device utility matrix directly to the CUDA nest reducer.
That run is a correctness/integration gate, not a repeated performance claim.

The two proofs together show that both sides of the compact logsum-cache seam
work on the GPU. They are not yet a single replacement ActivitySim component
with every orchestration table resident on the device.

## Compact primitive artifact

Phase 20's exact replay retained 518,909,174 bytes of already prepared row
arrays. Phase 21 derives a 12,688,620-byte in-memory primitive input: a
**40.896x reduction**. Its six compressed batch files total about 6.3 MB.

The artifact stores only:

- one person-row index per tour;
- chooser attributes once per tour;
- 25 float32 logsum slots plus a presence mask per tour;
- one float64 random draw per tour;
- coefficients and expected TDDs; and
- the shared 190-row start/end/duration table.

It does not store 15.24 million prepared features as an answer key. During
qualification, the CPU and GPU independently regenerate those rows and compare
them with the Phase 20 capture.

The cache builder fails closed if two TDDs mapped to the same skim-period slot
do not contain bit-identical float32 logsums. Only 15 of the 25 possible slots
are valid for a first tour because inbound time cannot precede outbound time.

## Timetable semantics

Each person owns a 21-period int8 timetable. A TDD alternative becomes a
footprint containing ActivitySim-compatible start, end, start-and-end, and
middle markers. A candidate collides when those markers conflict with an
already scheduled tour. The GPU computes a feasibility mask, converts it to
CSR offsets and alternative IDs, and constructs the row primitives only for
feasible alternatives.

Scheduling is sequential by design. First work tours are selected and written
into the timetable before first school tours, and all first tours precede the
second-tour batches. Parallel work occurs within a batch; the six batch
boundaries preserve the calibrated causal order.

Host NumPy or pandas arrays are rejected by the GPU-only preparation methods.
That fail-closed check prevents an accidental CPU fallback or hidden modeled
round trip from being described as a resident GPU result.

## The live gate failed twice before it passed

The first live attempt correctly fell back on all six batches because the
strict IR did not understand three expressions used by tour mode choice:

- `od_skims.reverse(...)`;
- `od_skims.max(...)`; and
- the `odr_skims` and `dor_skims` round-trip directions.

Phase 21 added explicit IR nodes and CUDA bindings for those semantics. Focused
tests prove that reverse swaps only origin and destination indices and that
the same immutable skim cube remains cached.

The second live attempt ran all CUDA batches but changed one schedule among
81,983. The difference was not ignored. A fresh unmodified ActivitySim/Sharrow
control reproduced the frozen reference exactly, isolating the defect to the
candidate arithmetic policy.

Sharrow's generated utility flow uses a float32 `np.dot` compiled with
`fastmath=True`. The strict cross-device compiler used separate, ordered
multiply and add operations with contraction disabled. Both policies are
reasonable, but one near-boundary scheduling draw made the difference
observable. Phase 21 therefore adds an isolated `fused_utility_accumulation`
policy that emits `fmaf` and enables contraction only for the Sharrow-compatible
live path. The strict no-FMA default remains unchanged for its original proof.

The ActivitySim mode-logsum path also has mixed precision: Sharrow creates the
float32 utilities, then ActivitySim's pandas reducer promotes them to float64
and applies a particular leaf-scaling, child-sum, exponent, and logarithm
order. A dedicated CUDA reducer now mirrors that real boundary. A standalone
Sharrow-float32 reducer is kept as a separately tested policy; using it for the
ActivitySim path was rejected after it changed 1,203 schedules.

With Sharrow-compatible fused utility accumulation and ActivitySim-compatible
mixed-precision nesting enabled together, the full live gate has zero changed
TDD, start, or end values.

## Correctness and replication guarantees

The release gates require all of the following:

- 15,242,743 generated interaction rows, exactly matching the capture count;
- zero CPU and GPU CSR-offset mismatches;
- zero CPU and GPU feasible-alternative-ID mismatches;
- zero CPU and GPU row-primitive mismatches;
- zero CPU and GPU chooser-value mismatches;
- zero CPU, GPU, and CPU-versus-GPU TDD mismatches;
- zero differences on a repeated GPU run;
- GPU faster than compiled CPU at both named timing boundaries;
- a smaller primitive input than the captured prepared rows;
- six live CUDA mode-logsum calls and zero fallbacks;
- zero utility device-to-host and nested host-to-device handoff bytes; and
- zero live TDD, start, and end differences against ActivitySim.

The result JSON records the software/hardware environment, all nine timing
samples, source and input hashes, workload sizes, per-batch mismatch counts,
and a checkpoint fingerprint. Tests cover period boundaries, timetable
footprints, cache factorization, sequential mutation, host-input rejection,
Numba/GPU equality, reverse skim bindings, round-trip IR parsing, each numeric
nest policy, and the fused-source policy.

## Reproduce

Build the compact input from the preserved Phase 20 capture:

```powershell
./.venv-phase8/Scripts/python.exe scripts/build_phase21_scheduling_inputs.py
```

Run the nine-repetition performance and correctness qualification:

```powershell
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase21_scheduling_pipeline.py `
  --repetitions 9 `
  --output benchmark-results/phase21-scheduling-pipeline.json `
  --checkpoint benchmark-results/phase21-scheduling-pipeline-checkpoint.json
```

For the live gate, first copy the preserved pipeline checkpoint to a new output
directory. ActivitySim opens a resumed pipeline for update, so never point the
command at the preserved reference itself. Then run:

```powershell
./.venv-phase8/Scripts/python.exe scripts/run_phase21_activitysim_logsum.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <copied-checkpoint-directory> `
  --reference-pipeline benchmark-data/phase9-mtc-full/prototype_mtc_extended/o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline `
  --report benchmark-results/phase21-logsum-live2.json `
  --kernel-reports <new-kernel-report-directory>
```

Add `--engine cpu` to create an unmodified ActivitySim/Sharrow control. Add
`--logsum-capture <directory>` only for precision diagnostics; those large
diagnostic arrays are not needed for ordinary qualification.

## What remains

Phase 21 is a major scheduling success, but it is not a complete GPU-only
travel model. The next proof should join the two qualified halves in one
device-resident production scheduler rather than through ActivitySim's pandas
logsum-cache API, then measure that complete component repeatedly against a
fresh CPU baseline.

After that, the project should:

1. carry the exact timetable state through a restartable device checkpoint;
2. implement non-mandatory, joint, and at-work scheduling with the same gates;
3. manage the 13.389-GiB raw skim collection with an explicit hot-cache and
   deterministic population partitioning policy;
4. qualify a second NVIDIA architecture;
5. qualify a second public activity-based model; and
6. run repeated whole-model A/B pairs before making any whole-model superiority
   claim.

Phase 21 proves that the former CPU preparation bottleneck can be regenerated
exactly and 8.680x to 10.199x faster at its honest compact-cache boundary, and
that the upstream real skim-to-logsum CUDA path can drive the unchanged public
scheduling outcome. It does not erase the value of Phases 1-20: their captured
answer keys, random semantics, strict compiler, stable IDs, and fail-closed
gates are what made this larger claim auditable.
