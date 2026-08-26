# Phase 27: compact input reconstruction on CUDA

## Outcome

Phase 27 removes the six captured row-dense strict-CUDA input matrices and all
captured origin/destination/time coordinate vectors from the **timed sealed
graph**. The graph now reconstructs those arrays on CUDA from 25,042,522 bytes
of compact state, runs the complete Phase 26 raw-skim-to-timetable chain, and
publishes the same 81,983 TDD labels.

The public 50,000-household MTC proof used three fresh Python/CUDA processes,
five warm-ups and five measured reconstruction runs per process, and one
warm-up plus five measured complete-graph runs per process. All source proof
gates passed.

| Metric | Process 1 | Process 2 | Process 3 | Median of medians |
|---|---:|---:|---:|---:|
| Complete graph | 0.205337 s | 0.208764 s | 0.204956 s | **0.205337 s** |
| CUDA reconstruction | 0.002915 s | 0.002906 s | 0.002939 s | **0.002915 s** |
| Matched NumPy reconstruction | 0.490051 s | 0.491203 s | 0.499566 s | **0.491203 s** |
| Matched reconstruction speedup | 168.13x | 169.04x | 169.96x | **168.52x** |

The complete graph is 2.23% slower than Phase 26's 0.200852-second median.
That small increase is the measured price of rebuilding 503,411,584 bytes of
row arrays instead of assuming those arrays already exist.

## Boundary and assumptions

The measured graph begins with these objects already resident:

- 149 shared raw skim arrays;
- compact chooser values and constant values;
- compact exact start/end-slot tables;
- compact dictionaries for repeated chooser-by-alternative response patterns;
- CSR chooser offsets and one compact slot code per modeled row;
- compiled strict utility, nesting, scatter, scheduling, and timetable state;
- the qualified 57-entry device boundary-decision map from Phase 26.

It does **not** begin with the captured row-dense input matrices or captured
skim coordinate vectors. Their device pointers are recorded before
factorization and the proof fails if any pointer appears in the sealed runtime.

This phase still uses ActivitySim's prepared dense rows once, before sealing,
to discover and qualify the compact representation. It therefore removes the
dense arrays from replay, not from cold initialization. A production upstream
compiler must emit the same compact factors directly from resident household,
person, tour, land-use, and scheduling-alternative tables. That is the next
boundary; this report does not claim it has already been crossed.

## Exact factorization contract

`ResidentInputExpansionPlan.compile` examines every packed float column, every
packed integer column, and every row-sized skim coordinate. It accepts a
column only when its raw bits have one of four proven forms:

1. **Constant** — one value applies to every row.
2. **Chooser** — one value applies to every row owned by a tour.
3. **Exact slot** — one value applies to every observed exact start/end pair.
4. **Chooser-response pattern** — a chooser stores a small pattern id; the id
   selects a deduplicated vector indexed by that chooser's CSR-local
   alternative position.

The fourth form is necessary for real preprocessor fields such as
`daily_parking_cost`, which combines a destination/chooser parking rate with
an alternative duration. It is also necessary for ragged feasible sets:
different tours do not always have the same exact start/end alternatives.
The representation stores repeated response patterns, not one captured value
per modeled row, and is accepted only if it is smaller than the source column.

Any column outside these forms raises an error. There is no row-dense fallback.
After compilation, CUDA reconstructs every target and compares its bytes with
the original before the originals can be removed. NaN payloads, signed zero,
integer values, and floating-point last bits are therefore part of the gate.

## CUDA execution

Each batch stores CSR chooser offsets. A small kernel expands row ownership
into reusable workspace. A second generic kernel reconstructs matrices and
coordinate vectors. It copies aligned 64-, 32-, 16-, or 8-bit values from the
qualified compact sources. It performs no floating-point calculation, so the
result does not depend on CUDA versus NumPy arithmetic order.

The reconstructed workspace is 508,252,080 bytes. Workspace is not a second
input dataset: it is the exact ABI required by the already-qualified strict
utility kernels and is overwritten on every replay. The compact persistent
input is 25,042,522 bytes. Compared with 503,411,584 removed captured row
bytes, the persistent representation is **20.102x smaller**.

The sealed graph then executes the unchanged Phase 26 chain:

1. reconstruct strict row inputs and skim coordinates;
2. evaluate six 315-term, 21-alternative mode-choice programs;
3. evaluate nested logsums;
4. scatter logsums into compact tour caches;
5. generate feasible scheduling rows and CSR indices;
6. choose TDDs and mutate the device timetable;
7. adjudicate 57 known arithmetic-boundary rows with the resident qualified
   map; and
8. publish only final TDD labels.

## Correctness and residency gates

Across 15 complete measured replays:

- all 1,210,124 mode-logsum rows were processed each time;
- every logsum was bit-identical to the live reference;
- every one of 81,983 final TDDs was exact;
- all 57 delicate rows were adjudicated on the device;
- no captured row-array pointer was registered;
- post-seal modeled host-to-device bytes were zero;
- intermediate modeled device-to-host bytes were zero; and
- modeled CPU fallbacks were zero.

The aggregate report hashes each of its three source reports and the Phase 26
comparison report. The complete test suite passes 131 tests.

## Interpreting the speed numbers

The **168.52x** number is a matched boundary comparison. NumPy and CUDA both
materialize the same 503.4 MB of exact ABI arrays from the same compact
factors; setup, factor discovery, downloads used to build the CPU control, and
the later utility/scheduling work are outside that timer.

The **0.205337-second** number is the complete sealed graph from compact input
state and resident raw skims through final timetable mutation. It is not a
cold ActivitySim runtime, a full-model CPU comparison, or a file-I/O result.
For those questions, retain the earlier 1.257x paired live component result,
24.516x calibrated resident vertical-slice result, and other explicitly
matched phase reports.

## What should come next

Phase 28 should eliminate factor discovery from captured dense rows. The
upstream compiler should name each compact source and generate it directly:

1. compile constant and chooser fields from resident household/person/tour and
   land-use columns;
2. compile exact scheduling-slot values from the resident alternative table;
3. replace anonymous response-pattern dictionaries with generated expressions
   such as parking-rate times duration where identical arithmetic can be
   specified, retaining dictionaries only as an explicit compatibility form;
4. run the same byte-exact reconstruction gate against ActivitySim during
   qualification but do not require ActivitySim's dense arrays in production;
5. repeat the three-process complete-graph proof; and
6. begin the same coverage expansion for non-mandatory and joint tours.

The independent arithmetic project also remains: a shared Sharrow/CUDA
definition for exponential, summation, normalization, and probability search
would replace the fixed 57-entry public-benchmark decision map.

## Reproduction

Start each process from a fresh copy of the frozen reference pipeline and run:

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
  --resident-generated-input-report <phase27-process-report.json> `
  --resident-replay-runs 5
```

Aggregate the three process reports:

```powershell
./.venv-phase8/Scripts/python.exe scripts/summarize_phase27_generated_inputs.py `
  --input benchmark-results/phase27-resident-input-1.json `
  --input benchmark-results/phase27-resident-input-2.json `
  --input benchmark-results/phase27-resident-input-3.json `
  --phase26 benchmark-results/phase26-resident-schedule-summary.json `
  --output benchmark-results/phase27-generated-input-summary.json
```
