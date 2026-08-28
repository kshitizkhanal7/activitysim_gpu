# Phase 36: device-generated trip utility ABI

## Outcome

Phase 36 removes the largest measured CPU data factory left by Phase 35. The
trip-destination runtime no longer has to construct 11 floating-point and 45
integer utility-input columns on the CPU. It uploads a compact packet of raw
trip, tour, person, household, and wait-time facts; one generated CUDA kernel
combines those facts with resident land-use arrays and writes the complete
11/45 ABI, availability flags, and skim coordinates directly on the GPU.

The authoritative public 50,000-household run exercised all 30 programs over
4,188,312 directional utility rows. It completed all 34 ActivitySim model
steps, used no fallback, and passed every runtime proof gate. An independent
comparison with the Phase 35 control found zero changed decision cells, zero
logsum difference, and all seven published CSVs byte-for-byte identical.

This is a replication-qualified architectural and data-motion improvement. It
is not yet a clean incremental stopwatch claim because an unrelated process
was actively using the GPU during qualification.

## The problem Phase 35 exposed

Phase 35 compiled and ran trip-mode utility arithmetic on CUDA, but the host
still produced a materialized ABI for every candidate destination. For each
row it built:

- 11 float32 values: 44 bytes;
- 45 int64 values: 360 bytes; and
- three int64 skim-coordinate vectors: 24 bytes.

The measured legacy packet was 428 bytes per row. Across 4,188,312 rows that
meant 1,792,597,536 bytes of dense host-side construction before the GPU could
begin useful work.

## The new contract

The Phase 36 packet is deliberately small and reviewable:

- fourteen `int32` columns (56 bytes): origin, destination, period, direction,
  first/last-trip flags, free parking, tour and parent modes, at-work-subtour
  flag, auto ownership, age, participants, and household size;
- two `float64` columns (16 bytes): duration and value of time; and
- three `float32` controlled wait draws (12 bytes).

That totals 84 bytes per row. Values that do not fit in `int32` fail closed.
Mode membership is compiled into explicit `uint64` bit masks. The three wait
draws stay tied to ActivitySim's controlled random stream; Phase 36 moves only
their final compact values, not the ownership or order of random draws.

Four land-use columns used by the model (`TERMINAL`, `PRKCST`, density, and
`TOPOLOGY`) are uploaded once and cached as a 46,528-byte resident device
alias. Skim cubes remain part of the already-qualified native resident store.

## Device preparation kernel

The kernel is generated from the reviewed ABI contract. For every row it:

1. reads the compact packet;
2. gathers the needed origin/destination land-use values;
3. writes all 11 float and 45 integer ABI slots in their fixed order;
4. applies every mode-availability and ferry rule; and
5. writes every grouped origin/destination/period skim coordinate.

The existing Phase 35 utility and nested-logit CUDA kernels consume these
outputs without an intervening host materialization. Kernel generation is lazy,
so the Phase 35 reference path remains available as an independent oracle and
rollback boundary.

## Replication strategy

Phase 36 has three levels of evidence:

1. **Unit contract tests.** They prove exact byte accounting, mode-mask
   construction, and fail-closed `int32` conversion.
2. **Shadow run.** On 500 households, Phase 35 and Phase 36 both generated the
   utility matrix in the same process. The gate compared device values before
   the Phase 36 result became authoritative. The complete resumed model then
   produced all seven output CSVs exactly.
3. **Production runs.** A 500-household run with the reference disabled and a
   complete 50,000-household run both produced exact published outputs.

The shadow comparator treats paired NaNs as equal, counts every other unequal
cell, computes a maximum absolute difference, and stops if that maximum exceeds
`1e-5`. Unknown sources, missing resident data, overflow, or a CUDA failure do
not silently switch to CPU.

## Full public benchmark evidence

| Measure | Phase 36 result |
|---|---:|
| households | 50,000 |
| zones | 1,454 |
| ActivitySim model steps completed | 34 of 34 |
| trip ABI programs | 30 of 30 |
| utility rows | 4,188,312 |
| dense host ABI construction eliminated | 1,792,597,536 bytes |
| compact input transferred | 351,818,208 bytes |
| net bytes removed at this boundary | 1,440,779,328 bytes |
| input-size reduction | 80.37% |
| resident Phase 36 land state | 46,528 bytes |
| device preparation kernel time, summed | 0.3060 s |
| compact host packet build time, summed | 4.0081 s |
| compact upload time, summed | 0.0368 s |
| CUDA utility time, summed | 1.9529 s |
| CUDA nested-logit time, summed | 0.0560 s |
| CUDA fallback calls | 0 |
| changed decision cells | 0 |
| maximum destination-logsum difference | 0 |
| maximum mode-logsum difference | 0 |
| byte-identical published CSVs | 7 of 7 |
| resident GPU bytes released after last consumer | 6,200,027,648 |

The 0.3060-second preparation sum contains lazy first-use compilation and is
reported as observed, not cleaned up after the fact.

## Performance claim boundary

The Phase 35 control artifact recorded 224.58 seconds for the complete model
and 28.9 seconds for trip destination. This Phase 36 run recorded 173.97 and
22.6 seconds respectively. Those numbers are descriptive only: a separate
GPU workload was active, so this is not a controlled matched pair and must not
be presented as a Phase 36 speedup.

What is proved without a stopwatch assumption is the smaller contract: the
same model answers are generated while eliminating a precisely counted 1.793
GB dense CPU materialization and replacing it with a 351.8 MB compact transfer.
The repository includes a three-pair Phase 35-versus-Phase 36 runner for a
future quiet-machine timing qualification.

## Assumptions and limits

- The public MTC Extended model and its reviewed strict ABI define the current
  semantic scope; a changed expression contract must be recompiled and gated.
- ActivitySim still owns tables, control flow, sampling, retries, controlled
  randomness, and output writing.
- Wait-time draws are generated on the CPU to preserve the exact random ledger.
- The compact packet is still rebuilt and uploaded for each program. Phase 36
  removes dense ABI construction, not every host boundary.
- The GPU stores the finished ABI temporarily because the already-qualified
  utility kernel consumes that shape. Fusing preparation and utility is a
  separate optimization with a larger verification surface.
- Exact output evidence covers the frozen public benchmark plus two smaller
  authoritative paths. Changed-world qualification remains necessary before a
  general-purpose upstream claim.

## Reproduction

Run the complete authoritative path:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\run_phase22_integrated_scheduling.py `
  --full-model `
  --phase36-device-trip-abi `
  --households-sample-size 50000 `
  --native-abi-live
```

Run the direct matched-pair harness on a quiet machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase36_device_trip_ab.ps1
```

The wrapper alternates Phase 35 and Phase 36 fresh processes, verifies each
candidate's complete outputs, and writes a summary. Timing evidence is valid
only if the machine has no competing GPU workload and the paired-run gates pass.

## Next major opportunity

Phase 37 should keep compact trip/tour/person/household state resident across
all 30 purpose programs and fuse ABI preparation with utility evaluation where
register pressure permits. It should also move trip-scheduling host indexing
into a resident tour-chain state machine: different tours advance in parallel,
while trips inside one tour retain ActivitySim's order, retry behavior, and
controlled random ledger. Qualification must include changed households,
coefficients, land use, and skim worlds, followed by clean matched timing pairs.
