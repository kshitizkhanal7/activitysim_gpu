# Phase 20: exact variable-length tours and calibrated scheduling

## Outcome

Phase 20 crosses the fixed-table boundary that limited Phase 19. A mandatory
person can create one tour or two, so the GPU must now create a new table whose
length is not known until after the calibrated choice. It must also preserve
ActivitySim's row order, type numbering, schedule order, destinations, and
stable IDs.

On the public 50,000-household Prototype MTC Extended checkpoint, the new GPU
path expands 78,900 mandatory persons into 81,983 tours. Every value in all 12
generated columns matches ActivitySim:

- `tour_id`
- `person_id`
- `tour_type`
- `tour_type_count`
- `tour_type_num`
- `tour_num`
- `tour_count`
- `tour_category`
- `number_of_participants`
- `destination`
- `origin`
- `household_id`

The generated IDs then link exactly to all 81,983 choosers in a fresh capture
of the calibrated mandatory-tour-scheduling component. Across six real
work/school/university batches and 15,242,743 feasible tour-time interaction
rows, both the independent CPU compiler and CUDA compiler reproduce every
ActivitySim time-of-day choice. The GPU result has **zero TDD mismatches**
against the saved public scheduling checkpoint and is bit-repeatable across
nine runs.

Nine measured repetitions on the NVIDIA RTX A4000 give:

| Qualified boundary | CPU median | GPU median | GPU speedup |
|---|---:|---:|---:|
| Variable-length tour expansion, resident inputs | 0.006904 s | 0.000601 s | **11.496x** |
| Tour expansion, upload plus `tour_id` download | 0.006904 s | 0.001101 s | **6.272x** |
| Calibrated scheduling kernel, resident compact inputs | 0.173748 s | 0.009601 s | **18.097x** |
| Scheduling kernel, compact upload plus choice/logsum download | 0.173748 s | 0.059196 s | **2.935x** |

The machine-readable evidence is
[`phase20-tour-chain.json`](../benchmark-results/phase20-tour-chain.json). The
restart/audit manifest is
[`phase20-device-checkpoint.json`](../benchmark-results/phase20-device-checkpoint.json).

## Exact claim boundary

This phase proves two connected boundaries:

1. exact, GPU-native mandatory-tour table expansion from the frequency choices
   already proved by Phase 19; and
2. exact execution of the calibrated mandatory-scheduling utility,
   probability, and choice kernel from compact real ActivitySim inputs.

It does **not** claim that the complete scheduling component is GPU-native.
ActivitySim still prepares the scheduling inputs used by the replay:

- time-dependent mode-choice logsums derived from skims;
- feasible tour departure/duration alternatives;
- timetable availability and adjacency primitives; and
- the first/previous-tour state used by later tours.

Those prepared values are captured once and hashed. The CUDA calculation that
consumes them is real and calibrated, but their CPU preparation time is not
included in the kernel timing. This distinction prevents a 2.935x kernel win
from being mislabeled as a 2.935x whole-component or whole-model win.

## How variable-length expansion works

The five mandatory-frequency alternatives are:

| Choice | Work tours | School tours | Output rows |
|---|---:|---:|---:|
| `work1` | 1 | 0 | 1 |
| `work2` | 2 | 0 | 2 |
| `school1` | 0 | 1 | 1 |
| `school2` | 0 | 2 | 2 |
| `work_and_school` | 1 | 1 | 2 |

The GPU first turns each choice into a row count. A prefix sum gives every
person a non-overlapping output range. One fused CUDA kernel then writes the
complete table. Physical rows remain in ActivitySim's work-then-school order.
For a non-worker who selected `work_and_school`, only `tour_num` is swapped:
school is scheduled first even though the work row remains first in physical
storage. That subtle rule is covered by both a focused unit test and the full
checkpoint comparison.

### Stable tour IDs

ActivitySim creates IDs from a canonical universe of every possible tour label
in the configured model. The public configuration has 41 labels. The mandatory
positions are:

| Label | Canonical position | ID formula |
|---|---:|---|
| `school1` | 31 | `person_id * 41 + 31` |
| `school2` | 32 | `person_id * 41 + 32` |
| `work1` | 39 | `person_id * 41 + 39` |
| `work2` | 40 | `person_id * 41 + 40` |

The implementation does not treat this table as self-proving. A public-data
test derives the offset of every saved mandatory tour and requires all 81,983
IDs to satisfy the formulas.

## The full scheduling capture

The capture harness resumes the exact public pipeline after
`mandatory_tour_frequency`, runs only `mandatory_tour_scheduling`, and records
six batches:

| Batch | Choosers | Interaction rows |
|---|---:|---:|
| First work tour | 54,483 | 10,351,770 |
| First school tour | 21,271 | 4,041,490 |
| First university tour | 3,146 | 597,740 |
| Second work tour | 1,748 | 130,605 |
| Second school tour | 562 | 49,989 |
| Second university tour | 773 | 71,149 |
| **Total** | **81,983** | **15,242,743** |

Format 3 is compact-only. It stores chooser values once per tour, alternative
attributes once per TDD, and only genuinely row-varying values per feasible
interaction row. Pure expressions remain reviewed syntax for the CPU/CUDA
compiler. Expanded term, utility, and probability matrices are not saved; the
capture code also avoids rebuilding the expanded term matrix in memory.

The six compressed replay files total roughly 12 MB. Their union of chooser
IDs equals the GPU-generated tour-ID set, and their ActivitySim-selected TDDs
equal the saved scheduling checkpoint before the new kernel is judged.

## Precision error discovered and fixed

The first 50,000-household gate rejected the implementation: three GPU choices
and one CPU choice differed among 81,983 tours. All were draws extremely close
to a cumulative-probability boundary.

The old scheduling kernel had two arithmetic mistakes:

1. it rounded ActivitySim's float64 random draw to float32; and
2. it compared that rounded draw with unnormalized exponential weights.

ActivitySim instead produces normalized float32 probabilities, then subtracts
them in alternative order from the original float64 draw. The corrected CUDA
kernel preserves the draw as a double, normalizes each float32 weight, and
performs the ordered subtraction in double precision. The independent CPU
path now reconstructs ActivitySim's dense float32 probability boundary using
NumPy's reduction semantics before making the same ordered choice.

After the correction, all six batches have zero CPU/ActivitySim,
GPU/ActivitySim, and CPU/GPU choice mismatches. A focused regression test uses
a draw one float64 unit above 0.5; casting it to float32 would select the wrong
alternative and now fails the test.

The maximum CPU/GPU scheduling logsum difference is `3.814697265625e-06`.
Choices, which are the checkpointed behavioral output, are exact. The logsum
difference is reported rather than hidden because the CUDA and NumPy float32
reduction trees are not bit-identical.

## Random streams and restart evidence

Phase 19 supported only the first ActivitySim MT19937 draw for each entity.
Phase 20 adds an exact arbitrary-offset kernel. Tests cover offsets on both
sides of the 312-double MT19937 twist boundary (`311`, `312`, and `313`) and
match NumPy bit for bit. Offset zero keeps its faster specialized kernel.

`ActivitySimRandomLedger` reserves and advances draw positions by channel and
step. Its sorted snapshot can be stored in a checkpoint and restored. The
Phase 20 manifest records:

- completed component names;
- row count, dtype, and SHA-256 for every generated tour column;
- the selected TDD hash;
- the scheduling random-channel offset; and
- the source capture-manifest hash.

This is a hash-complete audit/restart manifest, not yet a self-contained model
checkpoint: the compact scheduling-preparation arrays remain external replay
inputs.

## Reproduce

The compact capture can be regenerated from the public checkpoint with:

```powershell
$env:PYTHONPATH = "src;tmp/activitysim-phase8-source"
./.venv-phase8/Scripts/python.exe scripts/capture_phase2_activitysim.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <a-copy-containing-pipeline.parquetpipeline> `
  --capture benchmark-results/phase20-scheduling-replay `
  --resume mandatory_tour_frequency `
  --only-next-model `
  --compact-only
```

Do not point `--output` at the preserved public baseline because ActivitySim
opens a resumed pipeline for update. Use a copy, as the qualification did.

Then run:

```powershell
$env:PYTHONPATH = "src"
$env:NUMBA_NUM_THREADS = "8"
./.venv-phase8/Scripts/python.exe -m pytest -q
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase20_tour_chain.py `
  --repetitions 9
```

The benchmark exits unsuccessfully unless upstream Phase 19 is exact, all 12
tour columns match, all generated IDs link to scheduling, every CPU/GPU
schedule choice matches ActivitySim, repeated GPU choices are bit-identical,
and both resident and transfer-inclusive scheduling timings beat the CPU.

## What Phase 20 changes, and what comes next

Phase 20 removes two major blockers:

- calibrated GPU choices can now grow a dependent table exactly; and
- a full-population calibrated scheduling kernel is exact and materially
  faster, including compact transfers.

The next decisive phase is **not** another scheduling replay. It is moving the
preparation boundary onto the GPU: read cached skim tensors, compute all
time-dependent mode-choice logsums, generate feasible alternatives from the
live timetable, update that timetable after each `tour_num` group, and feed the
result directly to the already-qualified kernel. Once that is exact, measure
the whole mandatory-scheduling component and the expanded Phase 19-20 chain.

After that, the same proof must be repeated on another NVIDIA architecture and
a second public model before making a portable superiority claim.
