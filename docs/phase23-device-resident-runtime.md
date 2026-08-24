# Phase 23: next-generation device-resident runtime

## Outcome

Phase 23 replaces the one-component ActivitySim callback boundary with a
versioned model state graph that remains on the GPU across six connected model
stages. On the public 50,000-household Prototype MTC Extended checkpoint, the
resident graph performs calibrated household auto ownership, calibrated
mandatory-tour frequency, variable-length mandatory-tour table creation,
stable tour-to-scheduling identity linkage, six ordered scheduling batches,
and final timetable mutation.

There is no intermediate modeled host array, CPU fallback, pandas table, or
publication between the first and last stage. All published choices, all 12
tour columns, all 81,983 time-of-day decisions, and the final timetable match
the independent CPU/public ActivitySim references. A self-contained checkpoint
restores the final device state and reproduces the schedule exactly.

Three independent Python/CUDA processes each ran nine measured CPU and GPU
repetitions after compilation warm-up. The replicated medians are:

| Measure | Result |
|---|---:|
| Independent CPU modeled chain | 0.767301 seconds |
| GPU resident modeled chain | 0.031456 seconds |
| Resident modeled speedup | **24.405x** |
| Slowest process-level resident speedup | **21.555x** |
| One-time setup plus one GPU run and publication | **1.356x** median speedup |
| Ten repeated runs with setup amortized | **9.055x** median |
| One hundred repeated runs with setup amortized | **20.868x** median |

The canonical replicated evidence is
[`phase23-device-resident-summary.json`](../benchmark-results/phase23-device-resident-summary.json).

## Why this is different from Phase 22

Phase 22 measured a real ActivitySim scheduling callback. Its 1.257x median
paired gain included about 30 seconds of ActivitySim initialization, pandas
orchestration, checkpoint access, and other host work surrounding a much
smaller GPU scheduling interval. It correctly proved that the compatible live
component was faster, but also exposed the limit of accelerating only a kernel
inside the existing CPU-owned runtime.

Phase 23 changes the ownership boundary. It uploads immutable state once,
compiles stable joins once, executes dependent modeled stages against versioned
CUDA tables, and publishes only named final outputs. This is why the modeled
speedup is large even though the fully compatible Phase 22 component gain was
modest. The two measurements answer different questions and neither replaces
the other.

## Runtime contract

[`device_resident_runtime.py`](../src/choiceforge/device_resident_runtime.py)
implements these fail-closed rules:

- `ingress_table` is allowed only before `seal_ingress`.
- Every modeled stage declares its input and output table names.
- A stage receives only the declared `DeviceTable` inputs.
- Every output column must implement the CUDA array interface; a NumPy result
  raises `GpuOnlyViolation` before any output is committed.
- A stage commits all validated outputs atomically and increments table versions.
- Replacing persistent state requires an explicit `replace=True` declaration.
- Intermediate tables can be released after their final consumer.
- CPU fallback is a hard error and a telemetry event.
- Publication downloads only explicitly named columns.
- CUDA events record device time while wall-clock samples retain Python launch
  and synchronization overhead.

The three qualification processes each record zero forbidden post-seal host
bytes and zero modeled CPU fallbacks. The canonical run uploaded 29,411,646
bytes, peaked at 66,706,705 bytes of persistent state, and published 8,665,731
bytes once at the end.

This contract can reject a host array that crosses a stage boundary. Like any
Python-embedded GPU runtime, it cannot prove that arbitrary callback code never
performs an unrelated CPU calculation internally. Production stages therefore
remain reviewed code, and correctness is checked independently at every
published behavioral boundary.

## Compiled resident topology

Repeated keyed joins were one of the first runtime bottlenecks. The Phase 19
prototype sorted source keys every time it joined zones to households or
households to persons. Phase 23 adds `key_rows_gpu`, which validates unique
source keys and missing targets once, then retains the resulting CUDA row map.

Before measured execution, the runtime compiles three static topology tables:

- household rows to land-use and accessibility rows;
- mandatory-person rows to household and auto-ownership rows; and
- mandatory-tour chooser rows to the frequency-choice rows that create them.

Subsequent executions use direct device gathers. This reduced the smoke-run
resident median from 0.0667 seconds to 0.0523 seconds before the expression
compiler optimization described next.

## Fused mandatory-frequency compiler

The old calibrated path launched many array operations to materialize 98
expression columns, then performed a separate matrix product, probability
calculation, and choice. That was correct but was the largest avoidable
resident overhead outside scheduling.

[`fused_mnl.py`](../src/choiceforge/fused_mnl.py) now generates one CUDA kernel
for the public mandatory-frequency MNL. Each thread reads one mandatory
person's resident fields, inserts the immediately preceding GPU auto-ownership
result, evaluates all 98 published expressions, accumulates five float64
utilities, computes normalized probabilities and the logsum, and traverses the
exact ActivitySim-compatible random draw to select a choice.

The compiler accepts the reviewed arithmetic/Boolean subset used by this
specification and rejects host inputs. The fused result has zero choice
differences across 78,900 mandatory persons. Its maximum logsum error against
the independent dense CPU reconstruction is `8.882e-16`. After fusion, the
three process-level resident speedups were 24.516x, 21.555x, and 24.405x.

## Public proof gates

All three process reports passed every gate:

- zero auto-ownership checkpoint differences;
- zero mandatory-frequency checkpoint differences;
- auto and fused-frequency logsums within `1e-10`;
- zero differences in every one of 12 generated tour columns;
- zero scheduled-tour identity and TDD differences across 81,983 tours;
- zero final timetable differences against the CPU runtime;
- zero checkpoint-restart schedule differences;
- zero repeat differences for auto choice, frequency choice, tour ID, and TDD
  across all nine measured repetitions in each process;
- zero post-seal modeled transfers and CPU fallbacks;
- exactly one final publication; and
- GPU faster than CPU, both resident and setup-inclusive, in every process.

The independent-process summary hashes all three source reports. Each source
report in turn hashes the input manifests, benchmark, and runtime source.

## Checkpoint and restart

The old audit manifests contained hashes but not enough arrays to restart
independently. Phase 23 writes a compressed `state.npz` plus `manifest.json`.
For each saved table column, the manifest records its storage key, dtype,
shape, and SHA-256. It also records table versions, completed stages, the
random-stream ledger, metadata, and the archive hash.

Restore verifies the archive and every column before uploading the state and
resealing ingress. Unit tests continue with a new modeled CUDA stage after
restore, proving that this is a continuation boundary rather than only an
audit file. The public checkpoint is about 2.4 MB compressed because only
published and continuation state is stored, not temporary interaction rows.

## Timing interpretation

The resident timing includes the connected modeled stages, Python kernel
launch overhead, and a final synchronization. It excludes file/configuration
parsing and JIT compilation, just as the independent CPU timing excludes its
Numba compilation.

The conservative one-run timing additionally charges one input upload (about
0.31 to 0.33 seconds), scheduler state initialization (about 0.09 to 0.10
seconds), device topology compilation (about 0.11 to 0.13 seconds), and one
final publication (about 0.002 to 0.003 seconds).

It does not include optional checkpoint serialization, which took about 0.34
seconds in the canonical run. A production model chooses checkpoint frequency;
it should not silently hide checkpoint time inside a kernel claim.

The 10-run and 100-run values are arithmetic amortizations of each measured
process, not measurements of ten different policy scenarios. They show how
fixed setup cost behaves when a calibrated or policy workflow repeatedly runs
the same resident graph. Real multi-scenario parameter switching remains a
future qualification.

## Exact scope and remaining boundary

This is a real next-generation runtime and a calibrated multi-component
vertical slice. It is not yet the whole ActivitySim model.

The ingress contains frozen upstream location/CDAP state and the compact 5x5
mode-choice-logsum cache used by scheduling. Phase 22 can produce those logsums
from raw skims on CUDA, but the present Phase 23 benchmark does not yet build
that cache inside the sealed graph. Destination choice, non-mandatory tours,
joint tours, at-work subtours, trips, shadow pricing, and pipeline output also
remain outside the Phase 23 graph.

The next engineering sequence is:

1. move the Phase 22 raw-skim logsum producer behind the resident table API;
2. add a bounded hot-skim cache for the 13.389-GiB public skim collection;
3. represent alternative sampling and destination shadow state as versioned
   device tables;
4. port non-mandatory, joint, at-work, and trip components;
5. add pre-ingressed parameter batches for measured multi-scenario execution;
6. partition larger populations with stable entity/random identities; and
7. repeat the complete proof on another GPU architecture and public model.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
./.venv-phase8/Scripts/python.exe -m pytest -q -p no:cacheprovider
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase23_device_resident.py `
  --repetitions 9 `
  --output benchmark-results/phase23-device-resident.json `
  --checkpoint benchmark-results/phase23-device-checkpoint
```

Repeat with `-2` and `-3` output/checkpoint names, then summarize:

```powershell
./.venv-phase8/Scripts/python.exe scripts/summarize_phase23_device_resident.py `
  --input benchmark-results/phase23-device-resident.json `
  --input benchmark-results/phase23-device-resident-2.json `
  --input benchmark-results/phase23-device-resident-3.json `
  --output benchmark-results/phase23-device-resident-summary.json
```

The benchmark and summarizer fail if any exactness, residency, restart,
repeatability, or performance gate fails.
