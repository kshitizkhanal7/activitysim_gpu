# Phase 37: fused trip-utility runtime

## Outcome

Phase 37 removes the large temporary GPU tables that remained after Phase 36.
One generated CUDA kernel now reads the 84-byte compact trip packet, gathers
resident land-use and skim values, derives the complete strict input contract
in registers, evaluates all utility expressions, and writes only the 21 final
mode utilities. It does not materialize the 11-float/45-integer ABI or four
groups of skim-coordinate vectors.

The authoritative public 50,000-household run completed all 34 ActivitySim
steps. All 30 trip-destination utility programs used the fused backend over
4,188,312 directional rows, with zero fallback. Relative to the Phase 36
device pipeline, it eliminated exactly 1,692,078,048 bytes of dense ABI arrays
and 268,051,968 bytes of coordinate arrays: 1,960,130,016 bytes of temporary
device writes and allocation demand in total. An independent verifier found
all seven published CSVs byte-for-byte identical to Phase 36, zero changed
decision cells, and zero difference in the declared logsum diagnostics.

This phase proves semantics, coverage, and memory-traffic removal. It does not
claim a new speedup because the available GPU was not quiet enough for a fair
matched timing experiment.

## Why Phase 36 was not the endpoint

Phase 36 removed a 1.79 GB CPU data factory by sending 351.8 MB of compact raw
facts. Its preparation kernel then expanded those facts on the GPU into:

- 11 float32 ABI values per row: 44 bytes;
- 45 int64 ABI values per row: 360 bytes; and
- four grouped int64 coordinate pairs: 64 bytes.

The next CUDA kernel immediately read those 468 bytes per row. Across the full
benchmark, that intermediate representation required 1.960 GB of device
storage and writes. It was a useful compatibility boundary, but not a useful
final data product.

## The fused execution path

The generated kernel preserves the reviewed strict expression IR and its
ordered FP32/FMA policy. For each candidate-destination row it now:

1. reads fourteen int32 facts, two float64 facts, and three float32 controlled
   wait values from the compact packet;
2. derives all aliases and availability rules in local CUDA variables;
3. gathers parking, terminal, density, and topology values from resident
   land-use arrays;
4. computes grouped skim indices directly from origin, destination, direction,
   and period;
5. evaluates the same 379 expression terms and ordered utility accumulation;
6. writes the 21 final utility values for nested logit.

The nested-logit kernel and ActivitySim orchestration remain unchanged. The
compact packet is still built and uploaded for each of the 30 programs so that
controlled random draws and current table semantics remain exact.

## Compiler and runtime changes

The canonical CUDA generator gained three deliberately narrow extension
points for direct kernels:

- typed row-source references can be replaced by generated register values;
- grouped skim coordinates can be replaced by inline coordinate expressions;
- additional kernel parameters and a per-row prelude can be emitted.

These extensions are rejected for the tiled generator, where their execution
semantics have not been reviewed. Unknown source overrides also fail closed.

The native ABI bootstrap can now allocate only one dummy row of legacy ABI and
coordinate storage while retaining the correctly sized utility output. The
production Phase 37 run used only 468 legacy bootstrap bytes per program. This
is an intentional dependency detector: any accidental legacy row read would
see a one-row buffer rather than a hidden full-size safety net.

The generated source is also scanned before compilation. Production is
rejected if it still contains a row-indexed read from `float_inputs`,
`int_inputs`, or any grouped coordinate vector.

## A useful failure found during qualification

The first minimal-bootstrap production attempt exposed an unresolved integer
alias. Multiple semantic names shared one ABI slot, but the first fused-source
map overrode only the canonical name. A remaining alias therefore attempted a
legacy read. The minimal bootstrap made the error visible, and the source guard
then turned it into a deterministic pre-compilation failure.

The compiler was corrected to override every binding alias by its shared slot.
The shadow run, minimal production run, and complete 50,000-household run were
then repeated successfully. This failure is retained in the report because it
shows why minimal state and source inspection are replication controls, not
just memory optimizations.

## Replication ladder

1. **Generator and bootstrap tests.** Tests verify source and coordinate
   substitution, rejection of unknown/tiled fusion, and one-row legacy state.
2. **500-household shadow run.** Phase 36 and Phase 37 utility matrices were
   generated in the same process for all 30 programs. The maximum difference
   was zero and complete published outputs were exact.
3. **500-household production run.** Phase 37 ran with only 468 bytes of legacy
   row state per program. All published outputs remained exact.
4. **50,000-household production run.** The full public model passed every
   Phase 37 runtime gate and the independent final-output comparison.

The shadow comparator treats paired NaNs as equal and rejects any other utility
difference above `1e-5`. The final verifier compares modeled decisions and
declared diagnostics and also records byte identity for every published CSV.

## Full public benchmark evidence

| Measure | Phase 37 result |
|---|---:|
| households | 50,000 |
| zones | 1,454 |
| ActivitySim model steps completed | 34 of 34 |
| fused utility programs | 30 of 30 |
| directional utility rows | 4,188,312 |
| compact host/device input | 351,818,208 bytes |
| dense device ABI eliminated | 1,692,078,048 bytes |
| grouped coordinate vectors eliminated | 268,051,968 bytes |
| total device intermediates eliminated | 1,960,130,016 bytes |
| legacy bootstrap state | 468 bytes/program |
| compact packet build time, summed | 4.4714 s |
| compact upload time, summed | 0.0805 s |
| fused utility kernel time, summed | 2.1572 s |
| nested-logit kernel time, summed | 0.0787 s |
| CUDA fallback calls | 0 |
| changed decision cells | 0 |
| maximum destination-logsum difference | 0 |
| maximum mode-logsum difference | 0 |
| byte-identical published CSVs | 7 of 7 |
| all-model-step time, descriptive only | 216.9 s |
| resident GPU bytes released after last consumer | 6,200,027,648 |

The 30 bootstrap allocations total 14,040 bytes, but only one 468-byte program
allocation is live at a time. The fused timing includes lazy compilation where
it occurred and has not been cleaned after the fact.

## What is and is not proved

Proved:

- the fused path covers every public-model trip-destination utility program;
- the legacy dense ABI and coordinate vectors are not production inputs;
- precisely 1.960 GB of device intermediate traffic/allocation demand is
  removed for the qualified workload;
- all modeled decisions and declared diagnostics match Phase 36; and
- every published output file is byte-identical.

Not yet proved:

- a Phase 37 incremental stopwatch win over Phase 36;
- lower hardware peak memory, which requires a dedicated device-memory sampler;
- changed-world generality beyond the frozen public benchmark; or
- a GPU-only ActivitySim model. Table orchestration, packet construction,
  controlled random-number ownership, retry control, and output writing remain
  on the CPU.

The full Phase 37 artifact recorded 216.9 seconds, while an older Phase 36
artifact recorded a lower time. They are not a matched quiet pair and cannot
establish either a speedup or a regression. Fusion trades global memory traffic
for more register work, so a clean experiment must measure both runtime and
occupancy rather than assume fewer bytes automatically means faster execution.

## Reproduction

Run the complete candidate:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\run_phase22_integrated_scheduling.py `
  --full-model `
  --phase37-fused-trip-utility `
  --households-sample-size 50000 `
  --native-abi-live
```

Run alternating Phase 36/37 processes on a quiet machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase37_fused_trip_ab.ps1
```

The wrapper defaults to three 50,000-household pairs, verifies complete outputs
for every pair, and reports component and whole-model timings. Accept timing
only if GPU utilization, temperature, clocks, and competing processes are
controlled for the entire experiment.

## Next major opportunity

Phase 38 should stop rebuilding the same compact packet 30 times. A normalized
resident trip/tour/person/household store should upload stable facts once and
have each purpose program provide only row selection, changed values, and its
controlled wait draws. It should add hardware peak-memory and kernel occupancy
measurement, changed-world mutation tests, and three clean Phase 36/37 timing
pairs before promoting a speed claim.

The following ambitious boundary is the trip scheduler: advance different tour
chains in parallel on the GPU while preserving trip order, retries, and the
controlled random ledger inside each tour. That is where structural memory
success can begin to remove a larger share of complete-model wall time.
