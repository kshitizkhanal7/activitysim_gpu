# Phase 31: persistent verified native skim store

## Outcome

Phase 31 removes the largest remaining cold-start dependency from the native
mandatory-scheduling path. A fresh process no longer asks ActivitySim/Sharrow
to open 826 OMX matrices, construct a 6.452 GB `SkimDataset`, or provide skim
wrappers to the six mode-logsum calls. It reads one versioned native payload,
checks every byte, uploads 149 deduplicated physical cubes, and binds all 209
logical skim sources directly to the reviewed CUDA ABI.

On the public 50,000-household, 1,454-zone MTC workload, the conservative cold
component boundary—including the Phase 31 scheduler and frozen boundary-map
initialization—fell from the Phase 30 median of 30.739373 seconds to 27.085396
seconds. That saves 3.653978 seconds, reduces this measured interval by
11.886962%, and is a 1.134906x speedup. Phase 31 won in all three fresh
processes. All 81,983 schedules and every bit of all 1,210,124 generated
logsums remained exact.

This is a material component-level cold-start improvement. It is not a claim
that an entire ActivitySim regional model is 1.135x faster.

## The bottleneck and design decision

Phase 30 stopped building 1,210,124 dense chooser-alternative rows, but cold
time stayed near 30.74 seconds. Measurement showed that a fresh run still
spent roughly 12 seconds loading and arranging Sharrow's 6.452 GB skim
dataset. The utility programs require only 209 logical skim bindings. Reverse
directions and repeated program references share data, reducing the true set
to 149 physical cubes: three static and 146 five-period cubes.

The Phase 31 design therefore treats skims as a compiled model asset:

```text
public OMX + land-use TAZ order + reviewed utility IR
                         |
                  one-time builder
                         |
       manifest.json + contiguous payload.f32
                         |
        validate manifest / IR / zones / budget
                         |
       double-buffered read + SHA-256 + CUDA upload
                         |
        149 physical cubes -> 209 logical ABI bindings
                         |
   utility -> nested logit -> cache -> scheduling -> calendar
```

The source OMX is compressed to 733,866,957 bytes. The GPU-ready float32
payload is 6,198,588,112 bytes, 8.446 times larger on disk. That is a deliberate
space-for-startup trade: build once, then avoid HDF5 decompression, matrix-name
inventory, xarray/Sharrow assembly, repeated direction views, and dtype/layout
conversion in every model process.

## Artifact contract

`choiceforge.native-skim-store.v1` contains:

- a contiguous float32 payload in stable physical-key order;
- 149 cube entries with key, source name, rank, shape, byte offset, byte count,
  and content SHA-256;
- the ordered five MTC time periods;
- the public 1,454-TAZ identity/order hash;
- a hash of the 209-binding skim contract derived from reviewed IR;
- source OMX and land-use hashes;
- one SHA-256 over every payload byte; and
- one canonical hash over the complete manifest.

The builder refuses to overwrite an existing artifact. This makes replacement
an explicit operation and prevents a partial new build from silently changing
a qualified store. The 6.20 GB local payload is generated benchmark data and
is intentionally not committed to Git; the builder, build report, source
hashes, and reproduction command are committed.

## Loader and performance optimization

The first correct implementation copied each cube into pinned memory, hashed
the cube and the aggregate separately, uploaded it, and synchronized. It passed
all gates but loaded in 11.148 seconds and produced a 33.503-second cold run,
which was not a performance success.

The final loader keeps two reusable pinned buffers. An unbuffered sequential
file read writes directly into one buffer while the other buffer uploads on a
nonblocking CUDA stream. One ordered aggregate SHA-256 covers every byte; the
hashed manifest fixes cube boundaries, shapes, and keys. Per-cube hashes remain
available for build provenance and are evaluated on the exceptional corruption
diagnostic path. This removes a redundant success-path hash and overlaps I/O,
verification, and transfer without weakening the aggregate integrity gate.

The final three verified loads were 4.229013, 4.042573, and 4.319567 seconds,
for a 4.229013-second median. The median ordered read-and-hash time was
4.117254 seconds. Because uploads overlap those reads, the separately exposed
upload wait/enqueue accounting median was only 0.073760 seconds; it must not be
misread as standalone 6.20 GB PCIe bandwidth.

## Fail-closed behavior

Loading stops before the store is returned if any of these differ:

- format version or canonical manifest hash;
- reviewed skim-contract hash;
- public zone identity or order;
- payload file size or aggregate payload hash;
- ordered physical keys, offsets, float32 byte counts, shapes, or ranks; or
- the declared 8 GiB device budget.

ActivitySim's saved pipeline uses zero-based row positions although the public
source uses TAZ IDs 1 through 1454. The adapter only restores `position + 1`
when the live index is exactly the complete consecutive range `0..1453`; the
result must then pass the artifact's public-zone hash. It does not guess for an
arbitrary zone index.

Payload corruption, manifest/contract mismatch, zone mismatch, overwrite, and
budget-overrun tests are included in `tests/test_native_skim_store.py`.

## Removing the final hidden Sharrow call

Skipping `skims_for_logsums` was not sufficient. ActivitySim's `Network_LOS`
constructor still inventoried the OMX matrices, and its `load_data` injectable
still constructed the Sharrow dataset. Phase 31 replaces both operations only
inside this one-zone native runner. The live reports require exactly one
inventory bypass, one data-load bypass, and six native skim stubs.

The live scheduler also used Sharrow for 57 numerically ambiguous choices in
earlier phases. Phase 31 uses the already-qualified sparse decision map on the
GPU. The conservative initialization map can contain 58 candidate positions;
exactly 57 were exercised in each live process, all were adjudicated on CUDA,
and zero boundary logsum bytes were downloaded. A newly ambiguous position
without a qualified entry still fails closed.

## Measured result

All times are seconds. “Cold with initialization” includes the scheduler and
boundary-map setup that the older ActivitySim interval started after, making
the Phase 31 comparison conservative against Phase 30's reported interval.

| Measurement | Process 1 | Process 2 | Process 3 | Median |
|---|---:|---:|---:|---:|
| ActivitySim checkpoint-to-result interval | 26.434294 | 26.570047 | 26.350100 | **26.434294** |
| Cold interval plus scheduler/map initialization | 27.085396 | 27.166764 | 26.933112 | **27.085396** |
| Verified native-store load | 4.229013 | 4.042573 | 4.319567 | **4.229013** |
| Complete resident raw-source-to-calendar graph | 0.217578 | 0.217801 | 0.224729 | **0.217801** |

Relative to Phase 30:

- conservative cold boundary: 30.739373 -> 27.085396 seconds, **3.653978
  seconds saved**, **11.886962% lower**, **1.134906x faster**;
- ActivitySim-only interval: 30.739373 -> 26.434294 seconds, 14.005% lower;
- resident graph: 0.226712 -> 0.217801 seconds, 3.931% lower.

The resident change is welcome but small; Phase 31's principal performance
claim is the repeated cold-boundary reduction.

## Replication evidence

The aggregate report fails unless all of these hold:

- three fresh processes and 15 complete resident replays finish;
- all live and resident proof gates pass;
- every replay has zero logsum bit mismatch and zero final schedule mismatch;
- all 6,198,588,112 payload bytes are verified in every process;
- 209 logical bindings consistently map to 149 physical cubes and 1,454 zones;
- payload and skim-contract hashes remain stable;
- no Sharrow dataset or OMX inventory is materialized in the live path;
- all 57 exercised boundary decisions remain on the GPU with zero download;
- the Phase 31, Phase 30 native, and Phase 30 legacy six-program logsum hashes
  are byte-identical; and
- every Phase 31 cold process beats the Phase 30 cold median.

The shared logsum aggregate SHA-256 is
`41ea4ab90d0b47595a6ad59b1598a050a09a01db5d775dd3d1ad9f5be79e1322`.
The payload SHA-256 is
`67c689cf833357fb61556d6cb96f4adf2a1135c6207ee702cb04392e5aa2539a`.
`benchmark-results/phase31-native-skim-store-summary.json` hash-chains all
three live reports, all three resident reports, the immutable build report,
the independent Phase 31/30 logsum hashes, and the Phase 30 summary.

## Reproduction

Build the local artifact once from the frozen public data and reviewed mode
choice specification:

```powershell
./.venv-phase8/Scripts/python.exe scripts/build_phase31_native_skim_store.py `
  --omx benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full/skims.omx `
  --land-use benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full/land_use.csv `
  --spec benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/tour_mode_choice.csv `
  --output benchmark-data/phase9-mtc-full/prototype_mtc_extended/native_skim_store_v1_phase31 `
  --report benchmark-results/phase31-native-skim-store-build.json
```

For each proof process, copy the frozen reference pipeline to a fresh output
directory and run `scripts/run_phase22_integrated_scheduling.py` with
`--native-abi-bootstrap-report`, `--native-skim-store`, and five resident
replays. Add `--qualification-logsum-hash-report` to one untimed proof run.
Then run `scripts/summarize_phase31_native_skim_store.py` with the three live,
three resident, build, Phase 31 hash, Phase 30 native/legacy hash, and Phase 30
summary inputs.

## Assumptions, limits, and next major phase

This proof assumes the frozen public MTC one-zone model, its 1,454 ordered TAZs,
five time periods, the reviewed mandatory tour-mode IR, float32 skim storage,
an NVIDIA GPU with enough memory, and a reusable local artifact. Changed
network skims require a new artifact and new qualification. The result is from
one RTX A4000 and one local storage/cache environment; it is not a cross-device
performance guarantee.

Phase 31 accelerates one resumed mandatory-scheduling component, not all model
steps. Python, ActivitySim orchestration, checkpoint restore, raw tables,
configuration, and final pipeline publication remain on the CPU. The payload
also consumes 6.20 GB of local disk and GPU memory.

The next ambitious phase should turn this component-specific store into a
model-wide compiled asset registry and extend the same native ABI/store/runtime
contract to non-mandatory and joint tours, destinations, trip mode choice, and
trip scheduling. It should measure end-to-end model wall time, not just this
checkpointed component. In parallel, a shared specified exponential/reduction
implementation plus changed-scenario boundary fuzzing is still required before
the frozen ambiguity map can be removed universally.
