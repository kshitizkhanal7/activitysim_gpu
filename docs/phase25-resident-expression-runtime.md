# Phase 25: resident expression-to-logsum runtime

## Outcome

Phase 25 connects the public MTC tour-mode equations to the resident skim
layer. Six real programs now run from raw device skims through the generated
315-term CUDA utility evaluator, the 21-alternative nested-logit reducer, and a
compiled device scatter into the scheduling 5-by-5 logsum caches.

The measured resident path has **no precomputed logsum input**, no bulk modeled
logsum download, and no post-seal host scatter planning. Across three fresh
Python/CUDA processes and 15 measured replays, every one of 1,210,124 generated
logsums per replay was bit-identical to the corresponding live ActivitySim
CUDA result. The cross-process median was **0.1694235 seconds** and the slowest
process median was **0.1697392 seconds**.

## What was built

### A sealed strict-CUDA invocation

`ResidentStrictCudaInvocation` is a reusable launch object created by the
strict expression compiler. At capture time it owns immutable snapshots of:

- compact dense float and integer chooser inputs;
- scalar inputs;
- origin, destination, and time-period coordinate vectors;
- the compiled CUDA kernel, launch geometry, coefficient matrix, and output
  workspace.

The 149 physical skim arrays remain shared. They are not duplicated for each
of the six programs. Calling `execute()` launches the qualified kernel without
expression resolution, pandas conversion, host packing, host-to-device input
transfer, compilation, or allocation of its utility output.

### A compiled cache-scatter plan

The live Phase 22 bridge determined each source row's tour and 5-by-5
outbound/inbound period slot on the host. Repeating that work inside the Phase
25 timed loop would violate the resident boundary. `CompiledDeviceLogsumScatter`
therefore validates and compiles the layout once before sealing, uploads only
the unique flat positions and source positions, and owns reusable float32,
float64, and presence caches.

Each replay then clears and fills those existing device arrays directly. It
does not inspect chooser identities on the host, call `numpy.unique`, allocate
a new cache, or upload scatter indices.

### The live capture and proof bridge

The ActivitySim bridge can optionally expose each qualified resident
invocation together with its actual nest, alternative order, row identity, and
reference device logsum. This is not a substitute specification: it captures
the exact six programs that the public mandatory-scheduling run executes after
ActivitySim has prepared their chooser rows.

The same live run still executes the complete Phase 22 path and compares final
TDD, start, and end values with the frozen public pipeline. The resident replay
therefore has both a device-level bit oracle and an end-to-end model-output
oracle.

## Public workload

Each process executes:

- 50,000 public MTC households;
- 81,983 mandatory tours in the final scheduling output;
- six purpose/tour-number mode-logsum programs;
- 1,210,124 tour/period rows per resident replay;
- 315 expression terms and 21 alternatives per program;
- 209 logical skim bindings backed by 149 unique physical arrays;
- 381,189,060 term evaluations per replay; and
- 252,915,916 logical skim reads per replay.

Resident state used by this boundary is:

| State | Bytes |
|---|---:|
| 149 unique skim arrays | 6,198,588,112 |
| Sealed dense utility inputs | 348,517,296 |
| Sealed skim coordinate vectors | 154,895,872 |
| Compiled scatter positions | 19,361,984 |

The skim value differs slightly from Phase 24's 6,378,932,500-byte conservative
hot set because the live Sharrow dataset has already dropped matrices unused
by the actual run. The process-wide pointer audit counts every allocation once;
the sum of six program references must not be interpreted as six physical
copies.

## Qualification results

| Result | Process 1 | Process 2 | Process 3 | Cross-process result |
|---|---:|---:|---:|---:|
| Resident median | 0.169039 s | 0.169739 s | 0.169423 s | **0.169423 s** |
| Resident minimum | 0.168441 s | 0.168252 s | 0.167955 s | - |
| Speedup vs the same process's initial live CUDA utility/nest path | 10.389x | 9.655x | 9.475x | **9.655x median** |
| Measured replays | 5 | 5 | 5 | 15 |
| Logsum bit mismatches | 0 | 0 | 0 | 0 |
| Live TDD/start/end mismatches | 0 | 0 | 0 | 0 |

The speedup denominator is the six initial live CUDA calls' measured binding
resolution, packing, uploads, plan/constant setup, utility kernels, and nested
kernels. It is not a CPU comparison and not a whole-model speedup. Phase 22's
paired 1.257x result remains the compatible live CPU-versus-GPU component
claim; Phase 23's 24.405x result remains the calibrated resident vertical-slice
claim; Phase 24's 193.114x result remains an isolated skim-access/hash claim.

All Phase 25 gates require:

- three independent processes;
- six captured real programs per process;
- the 315-term and 21-alternative schemas;
- all 1,210,124 rows in every replay;
- zero bit or absolute logsum difference in all 15 replays;
- zero precomputed-logsum input bytes;
- zero bulk modeled-logsum device-to-host bytes;
- zero post-seal host scatter-layout builds;
- a resident win in every process; and
- exact final ActivitySim TDD, start, and end outputs in every live run.

The hash-chained summary is
[`phase25-resident-raw-skims-summary.json`](../benchmark-results/phase25-resident-raw-skims-summary.json).
Its source reports are the three `phase25b-resident-raw-skims-*.json` and
`phase25b-live-*.json` files.

## Assumptions and boundary

Phase 25 assumes the model specification, coefficients, network skims, and
batch row topology do not change while a sealed scenario executes. A changed
scenario may reuse the compiled kernel and skim arrays, but changed chooser
fields or OD/time coordinates must be ingressed and sealed as a new version.

This phase eliminates the **bulk precomputed scheduling-logsum input from the
new resident producer**. It does not silently rewrite the older Phase 23
benchmark artifact; that artifact remains a historical proof with its original
named boundary.

This is also not yet a CPU-free whole model:

- ActivitySim still prepares the six dense chooser batches and captures them
  before the resident boundary;
- Python launches the CUDA stages;
- the live end-to-end path still sends 57 independently detected
  near-boundary scheduling choices (11,400 logsum bytes) to exact
  ActivitySim/Sharrow adjudication;
- final selected schedules are published to ActivitySim; and
- non-mandatory, joint, at-work, destination, trip, shadow-pricing, and normal
  output components are outside this phase.

The resident expression-to-cache replay itself does not use the 57-row
resolver because it stops at bit-exact logsum caches. The live scheduling proof
uses the resolver and reports it explicitly.

## Reproduction

Copy the frozen pipeline into a fresh output directory, then run:

```powershell
./.venv-phase8/Scripts/python.exe scripts/run_phase22_integrated_scheduling.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <fresh-output> `
  --reference-pipeline benchmark-data/phase9-mtc-full/prototype_mtc_extended/o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline `
  --report <live-report.json> `
  --checkpoint <checkpoint.json> `
  --kernel-reports <empty-report-directory> `
  --resident-replay-report <resident-report.json> `
  --resident-replay-runs 5
```

After three independent runs:

```powershell
./.venv-phase8/Scripts/python.exe scripts/summarize_phase25_resident_raw_skims.py `
  --input <resident-1.json> --input <resident-2.json> --input <resident-3.json> `
  --live <live-1.json> --live <live-2.json> --live <live-3.json> `
  --output benchmark-results/phase25-resident-raw-skims-summary.json
```

Run `./.venv-phase8/Scripts/python.exe -m pytest -q` for the unit and integration
suite. The completed implementation passes 129 tests on the qualification
machine.

## Next engineering target

The next phase should move dense chooser construction and OD/time coordinate
generation behind the versioned device runtime, then make the generated
logsum caches direct inputs to the already-qualified resident timetable stage.
In parallel, a shared Sharrow/CUDA arithmetic primitive for scheduling utility,
exponentiation, ordered sums, and probability search should remove the 57-row
adjudication boundary. Success means exact final schedules with zero modeled
CPU fallback, not merely another fast isolated kernel.
