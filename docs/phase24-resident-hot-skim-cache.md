# Phase 24: budgeted resident hot-skim cache

## Outcome

Phase 24 completes the memory and raw-data foundation needed to bring the
Phase 22 mode-logsum producer into the sealed Phase 23 runtime. It reads the
reviewed 315-term tour-mode IR, discovers exactly 209 logical skim bindings,
deduplicates them into 149 physical arrays, converts only that hot set from
the public OMX source to float32, uploads it once, and registers the existing
CUDA arrays in the versioned runtime without copying them again.

The public hot set occupies 6,378,932,500 bytes (5.940 GiB), below the declared
8 GiB cache budget and well below the RTX A4000's 16,376 MiB capacity. Its
corresponding uncompressed float64 source values occupy 12,757,865,000 bytes.
The `.omx` file is only 733,866,957 bytes on disk because HDF5 compression is a
storage property, not the in-memory model representation.

Three independent Python/CUDA processes each performed five measured CPU and
GPU repetitions over the real mandatory-scheduling OD/period workload:

| Measure | Process 1 | Process 2 | Process 3 | Replicated middle |
|---|---:|---:|---:|---:|
| Valid raw-skim rows | 1,204,594 | 1,204,594 | 1,204,594 | 1,204,594 |
| Logical reads per run | 251,760,146 | 251,760,146 | 251,760,146 | 251,760,146 |
| CPU hot-cache median | 5.6914 s | 5.6616 s | 5.6690 s | 5.6690 s |
| GPU resident median | 0.0295 s | 0.0688 s | 0.0276 s | 0.0295 s |
| Resident speedup | 193.114x | 82.323x | 205.639x | **193.114x** |
| Upload-inclusive one-run speedup | 1.813x | 1.851x | 1.802x | **1.813x** |
| Ten-run upload-amortized speedup | 16.717x | 15.396x | 16.702x | **16.702x** |

The conservative minimum resident speedup is 82.323x. GPU memory-clock state
caused visible process variation, so the report preserves every process value
rather than presenting only the fastest run.

The canonical evidence is
[`phase24-resident-skim-cache-summary.json`](../benchmark-results/phase24-resident-skim-cache-summary.json).

## What a skim cache is

A skim is a network measure between an origin zone and a destination zone,
such as distance, driving time, toll, transit wait, or fare. Time-dependent
skims add a third axis for EA, AM, MD, PM, and EV periods. ActivitySim's public
MTC OMX collection has 826 stored matrices. Loading the complete uncompressed
float64 collection would consume about 13.39 GiB before model state and
workspaces.

The strict IR provides a better authority than a manually maintained list. Its
tour-mode expressions reference 209 directional meanings through six wrapper
types. Shared source cubes and reverse-direction views allow those logical
bindings to use only 149 physical allocations.

The cache refuses to load if:

- the IR contains a dynamic or unsupported skim reference;
- a required OMX matrix is absent;
- an OMX matrix is not square;
- the calculated float32 hot set exceeds the declared byte budget; or
- a sealed runtime is asked to attach new device data.

Each physical float32 cube receives a dtype/shape-aware SHA-256. The source
report records those hashes, the OMX file hash, strict-IR hash, workload hash,
cache implementation hash, and benchmark hash.

## Public workload and exact proof

The workload comes from all six captured mandatory-scheduling mode-logsum
batches: 1,210,124 tour/period rows. The proof links their stable tour IDs to
the public pipeline's real origin and destination zones. It excludes 5,530
rows whose tours have destination zone 0, the model's missing-destination
sentinel, leaving 1,204,594 valid OD/period rows.

For every valid row, the CPU and GPU independently read all 209 logical
bindings—251,760,146 raw skim values per run. Downloading that full matrix
would itself be a large artificial cost, so each implementation folds every
float32 bit pattern, in the same declared order, into two 64-bit row hashes.
This is an exact read-integrity proof rather than a floating tolerance check.

Across all three processes:

- CPU/GPU hash-word mismatches: **0**;
- repeat hash-word mismatches: **0** across 15 measured GPU runs;
- post-seal modeled host-transfer bytes: **0**;
- modeled CPU fallbacks: **0**;
- final publications per process: **1**; and
- public result hashes: identical in all processes.

The runtime now distinguishes ordinary host ingress from zero-copy attachment
of an already-resident CUDA table. The cache loader owns the one permitted OMX
read and upload; the sealed graph owns the arrays afterward.

## Timing interpretation

The resident timing measures the all-binding CUDA probe after disk loading,
upload, allocation, and JIT compilation. Its CPU comparator starts from the
same hot float32 arrays in RAM and performs the same ordered reads and hashes
with NumPy. This is a cache-layer throughput comparison, not a complete model
comparison.

The upload-inclusive measure charges the 2.89-to-3.15 second one-time transfer,
one GPU probe, and one final 19.3 MB publication to a single execution. It does
not charge the roughly 26-second OMX decompression/read to one side only,
because both CPU and GPU need the source loaded. Even under that conservative
boundary, GPU won all three processes.

The very large resident ratio must not be substituted for Phase 23's 24.405x
calibrated chain result or Phase 22's 1.257x live component result. Phase 24's
CPU loop and fused CUDA probe isolate the network-data access layer; they do
not evaluate 315 utility terms, nested logit, or scheduling choice.

## Exact claim boundary and next phase

Phase 24 proves that the real public hot skim set fits this GPU, can be selected
from the expression contract instead of hand-maintained names, stays resident
inside the fail-closed runtime, returns every tested raw value bit exactly, and
has large measured bandwidth headroom.

It does **not** yet remove the compact 5-by-5 logsum cache from Phase 23's
ingress. The all-binding hash probe is a qualification kernel, not the actual
mode-choice utility and nested-logit engine. It also does not cover sparse MAZ
overlays, destination sampling, shadow pricing, or the remaining ActivitySim
components.

The next phase should use these resident cube pointers as the skim bindings of
the already-qualified strict CUDA expression plan. It must then:

1. construct each batch's dense chooser and OD/period indices on the GPU;
2. run the real 315-term, 21-alternative mode utility kernel;
3. reduce utilities through the ActivitySim-compatible nested-logit tree;
4. scatter the device logsum vector into the 5-by-5 scheduling cache;
5. define GPU arithmetic for the 57 near-boundary scheduling rows or retain an
   explicitly measured adjudication boundary;
6. replace Phase 23's precomputed cache ingress with that producer; and
7. repeat exactness, restart, residency, and performance gates on the connected
   calibrated chain.

Only step 6 closes the raw-skim-to-resident-model gap. Phase 24 makes it
practical and measurable; it does not claim it prematurely.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
./.venv-phase8/Scripts/python.exe -m pytest -q -p no:cacheprovider
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase24_resident_skim_cache.py `
  --repetitions 5 `
  --output benchmark-results/phase24-resident-skim-cache.json
```

Repeat in two new Python processes using `-2` and `-3` output names, then run:

```powershell
./.venv-phase8/Scripts/python.exe scripts/summarize_phase24_resident_skim_cache.py `
  --input benchmark-results/phase24-resident-skim-cache.json `
  --input benchmark-results/phase24-resident-skim-cache-2.json `
  --input benchmark-results/phase24-resident-skim-cache-3.json `
  --output benchmark-results/phase24-resident-skim-cache-summary.json
```

Both the per-process benchmark and replicated summarizer fail closed when a
required correctness, residency, repeatability, or performance gate fails.
