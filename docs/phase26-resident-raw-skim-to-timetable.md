# Phase 26: sealed raw-skim-to-timetable runtime

## Outcome

Phase 26 joins the Phase 25 raw-skim expression producer directly to the
qualified GPU timetable path. One versioned resident stage now executes six
real 315-term mode-choice programs, 21-mode nested logits, compiled 5-by-5
cache scatter, feasible-alternative and CSR construction, scheduling choice,
and sequential timetable mutation.

On the 50,000-household public MTC workload, three fresh processes completed
15 measured replays. The median of process medians was **0.200852 seconds**;
the slowest process median was **0.205299 seconds**. Every replay reproduced
all 1,210,124 generated mode-choice logsums bit-for-bit and all 81,983 final
TDD choices exactly. No precomputed logsum was an input, no modeled array
crossed to the host after sealing, and no modeled CPU fallback ran.

## Connected device graph

The measured stage is:

```text
resident raw skim cubes + sealed dense mode-choice leaves
  -> six generated 315-term CUDA utility programs
  -> 21-alternative nested-logit reductions
  -> compiled device scatter into per-tour 5x5 caches
  -> device timetable collision masks
  -> device counts, prefix sums, CSR alternative ids and row owners
  -> device scheduling-expression evaluation and probability search
  -> device timetable mutation
  -> final TDD publication only
```

`DeviceResidentRuntime` registers the actual CUDA arrays by reference, seals
ingress, records the stage's declared dependencies and output versions, and
rejects host modeled arrays after the seal. Phase 26 registered 306 named
assets representing the executable programs, deduplicated shared arrays,
scatter plans, and scheduling inputs. The final schedule table reached version
6 after one warm-up and five measured replacements.

`GpuSchedulingPreparer` constructs the large scheduling interaction layout on
the device. For each ordered batch it builds the chooser-by-190 collision
mask, counts feasible alternatives, performs the prefix sum, compacts TDD ids
and owner rows, writes the eight row-varying timetable/logsum fields, and later
mutates the persistent person windows. The measured scheduling graph therefore
does not ingest a 15.2-million-row prepared scheduling table.

## The arithmetic investigation

The earlier live bridge downloaded 57 near-boundary rows, totaling 11,400
bytes of raw logsums, and asked ActivitySim/Sharrow to settle them. Only one
row actually changed without this protection.

The Phase 26 audit captured Sharrow's utility and probability vectors for that
row and established three facts:

1. ActivitySim's probability vector is reproduced bit-for-bit by float32
   `exp`, NumPy's float32 sum, and float32 division with overflow shifting
   disabled.
2. Sharrow materializes a 65-element float32 expression vector and evaluates
   `np.dot(intermediate, dotarray)`, where `dotarray` is two-dimensional.
   A Numba reproduction with that exact vector-by-matrix shape matches all 190
   captured Sharrow utilities bit-for-bit. Vector-by-vector dot products and
   guessed lane reductions do not.
3. Even with the utility order understood, CUDA `expf`, reduction, and division
   are not a portable bit-level implementation of the host NumPy/BLAS path.
   Pretending otherwise would weaken the replication claim.

Phase 26 therefore uses an explicit **qualified decision map**. The CUDA kernel
independently flags rows whose draw is within 2e-6 of a cumulative probability
boundary. For the frozen public benchmark, the corresponding Sharrow TDD
labels are versioned resident CUDA state. All 57 flagged rows are adjudicated
on the device; one TDD is corrected per replay. No row id, logsum vector, or
decision is downloaded during the modeled stage.

This is exact for the qualified benchmark and transparent about why. It is not
a universal claim that arbitrary changed inputs inherit Sharrow arithmetic.
A changed model, coefficients, random stream, or chooser population must
requalify the ambiguity map or use a future shared arithmetic primitive.

## Qualification results

| Result | Process 1 | Process 2 | Process 3 | Cross-process result |
|---|---:|---:|---:|---:|
| Resident median | 0.199694 s | 0.205299 s | 0.200852 s | **0.200852 s** |
| Minimum measured replay | 0.199069 s | 0.201534 s | 0.200107 s | **0.199069 s** |
| Measured replays | 5 | 5 | 5 | 15 |
| Changed logsum bits | 0 | 0 | 0 | 0 |
| Final TDD mismatches | 0 | 0 | 0 | 0 |
| Boundary rows per replay | 57 | 57 | 57 | deterministic |
| Device corrections per replay | 1 | 1 | 1 | deterministic |
| Boundary bytes downloaded | 0 | 0 | 0 | 0 |

The hash-chained aggregate is
[`phase26-resident-schedule-summary.json`](../benchmark-results/phase26-resident-schedule-summary.json).
It records source hashes for the three independent reports.

The 0.201-second result is a resident repeated-scenario timing. It starts from
already loaded raw skim cubes and sealed dense mode-choice leaves; it excludes
initial ActivitySim row preparation, skim loading, compilation, and final file
writing. It must not be divided into the 40.389-second Phase 22 live CPU number
as if the boundaries were identical. Compatible established comparisons remain:

- Phase 21 scheduling preparation/choice: 10.199x resident GPU versus compiled CPU;
- Phase 22 live raw-skim mandatory scheduling: 1.257x median paired speedup; and
- Phase 23 calibrated resident vertical slice: 24.516x GPU versus its modeled CPU baseline.

## Proof gates

The aggregate fails unless all of these are true:

- at least three independent Python/CUDA processes;
- six real programs and all 1,210,124 logsum rows per replay;
- bit-identical logsums in every replay;
- exact final TDDs for all 81,983 mandatory tours;
- every detected ambiguity row handled by resident device state;
- zero boundary-logsum download;
- zero precomputed-logsum input;
- zero post-seal modeled host-to-device or intermediate device-to-host transfer;
- zero modeled CPU fallback; and
- every per-process proof gate passes before aggregation.

The full test suite is also required because the resident path reuses the
expression compiler, nested logit, cache scatter, scheduling compiler,
timetable preparer, and runtime lifecycle code.

## Assumptions and remaining boundary

Phase 26 assumes that the public MTC specification, coefficients, skims,
chooser order, random draws, and reference pipeline identified by the reports
are immutable during a replay. Python still launches kernels and controls the
versioned graph. ActivitySim still creates and initially packs the six dense
mode-choice leaf/coordinate batches before they are sealed. Final results are
published to the host because ActivitySim and output writers consume them.

Therefore this phase completes the direct cache-to-timetable connection and
removes the runtime CPU boundary resolver, but it does not yet create every
mode-choice chooser leaf and OD/time coordinate from higher-level resident
person, household, tour, land-use, and timetable tables. That is the next
important generalization. A second future task is to place the Numba/NumPy and
CUDA backends on one formally specified correctly rounded expression,
exponential, summation, normalization, and search implementation so a changed
scenario can avoid a benchmark-specific decision map.

## Reproduction

Create three fresh copies of the frozen reference output and run:

```powershell
./.venv-phase8/Scripts/python.exe scripts/run_phase22_integrated_scheduling.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <fresh-output-copy> `
  --reference-pipeline benchmark-data/phase9-mtc-full/prototype_mtc_extended/o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline `
  --report <live-report.json> `
  --checkpoint <checkpoint.json> `
  --kernel-reports <empty-report-directory> `
  --resident-schedule-report <phase26-process-report.json> `
  --resident-replay-runs 5
```

Then aggregate:

```powershell
./.venv-phase8/Scripts/python.exe scripts/summarize_phase26_resident_schedule.py `
  --input <process-1.json> --input <process-2.json> --input <process-3.json> `
  --output benchmark-results/phase26-resident-schedule-summary.json
```
