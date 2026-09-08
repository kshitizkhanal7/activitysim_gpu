# Phase 54: device-owned destination packet

## Outcome

Phase 54 removes another CPU staging boundary between destination sampling and
the previously qualified CUDA destination-logsum service. Sampling now
publishes a versioned GPU lease containing group offsets and sampled
destination IDs. The logsum service consumes those exact buffers, generates
the six controlled normal draws with a compatible MT19937 CUDA kernel, and
turns them into taxi/TNC wait tables on the GPU. It no longer rebuilds those
three packet sections through pandas and NumPy.

On the public Prototype MTC Extended workload at 50,000 households and 1,454
zones, three matched Phase 53/54 shard pairs produced:

| Measurement | Phase 53 median | Phase 54 median | Result |
|---|---:|---:|---:|
| packet preparation | 1.300 s | 0.306 s | 4.252x; 76.48% lower |
| instrumented destination service | 4.989 s | 3.713 s | 1.344x; 25.58% lower |
| five destination components | 12.6 s | 11.3 s | 1.115x; 10.32% lower |

Phase 54 won all three pairs at all three boundaries. Each pair covered all 19
calls, 201,390 owners, and 4,696,676 sampled destinations.

The former Phase 53 monolithic allocation problem was also resolved by using
one BLAS/OpenMP worker for this GPU benchmark. Three fresh complete runs took
134.267, 133.264, and 133.714 seconds including startup/resume bookkeeping: a
133.714-second median. That is **1.536x faster and 34.90% lower** than the
separately measured 205.4-second regular ActivitySim CPU median, and 1.025x
faster than the 136.991-second Phase 52 median.

## Architecture

The Phase 54 contract is version 5 and has three new device-owned sections:

1. The exact destination sampler packs contiguous owner groups as before, but
   now publishes its CUDA offset and destination buffers through a generation-
   numbered lease.
2. The logsum adapter must consume the lease in the same synchronous call,
   with the same dataframe object, index, destination column, row count, owner
   count, and owner-ID sequence. A stale or mismatched lease raises an error.
3. A CUDA MT19937 kernel fast-forwards each ActivitySim seed by the declared
   offset and generates the requested normal values with the same polar
   Box-Muller algorithm as legacy `RandomState`.
4. A second CUDA kernel maps those normals and resident density bands to the
   exact float32 taxi/TNC wait table required by the destination formula.
5. The existing hash-verified four-row utility, nested-logit, probability,
   choice, and device handoff remain unchanged.

The normal implementation follows NumPy's legacy distribution source and the
project deliberately targets the legacy compatibility contract, not the newer
`Generator` API. See the [NumPy legacy distribution source](https://github.com/numpy/numpy/blob/main/numpy/random/src/legacy/legacy-distributions.c)
and [NumPy random compatibility policy](https://numpy.org/devdocs/reference/random/compatibility.html).

## Replication and fail-closed guarantees

The formal evidence proves, in every matched pair and every complete run:

- all 19 calls used sampler-owned GPU leases;
- 1,208,340 controlled normal values were generated on CUDA for 201,390
  owners, with no host normal generation;
- all wait tables were generated on CUDA;
- exact public cardinality: 4,696,676 rows and 201,390 owners;
- the same source hash,
  `599a9704be0992d2863320390cbca0028c7a578ecacf72d69de2e658a5d79906`;
- exact school, workplace, and tour destination decisions;
- zero generic or CPU fallback;
- all complete-model proof gates passed;
- maximum observed cached diagnostic difference `5.7220458984375e-6`, below
  its declared tolerances.

Unit tests additionally compare GPU normals bit for bit with NumPy legacy
`RandomState` across multiple seeds and offsets, and compare GPU wait tables
exactly with the float32 host reference. Unsupported shape, index, generation,
source-object, or column contracts fail closed.

Later audit (Phase 58): a wider seed/step/offset matrix found 12 tiny normal-value
differences among 3,168 comparisons, despite exact ledger positions. The unit
tests above established equality for their tested cases, not a universal
bit-identical normal generator. Phase 58 therefore does not reuse this generator
as its general trip RNG. The fixed-workload Phase 54 output qualification remains
separate; see [the audit and retained CPU boundary](phase58-live-trip-runtime.md).

## Traffic and memory accounting

Across the 19 calls, Phase 54 still eliminates 1,953,817,216 bytes of dense
device ABI allocation. The compact path uploads 14,500,080 bytes of remaining
owner state, reuses 20,397,976 bytes of sampler device buffers, and generates
12,083,400 wait-table bytes directly on the GPU.

The speed has a deliberate memory cost. The persistent service reached
462,283,900 bytes (about 441 MiB) of workspace because it keeps separate
MT19937 state for the sampling stream and normal stream plus reusable packet
buffers. This fits comfortably on the qualified 16 GiB GPU, but another device
must pass its own capacity and timing qualification.

## Benchmark method and claim boundary

The incremental claim uses three deterministic Phase 53/54 pairs split around
the already verified mandatory-scheduling checkpoint. Identical inputs and
cardinalities make the destination comparison matched. Phase 54 wins each
pair; medians are not hiding a losing trial.

The complete-model claim comes from three fresh monolithic Phase 54 runs. Its
regular ActivitySim and Phase 52 comparators are separately measured historical
three-run medians on the same public workload and machine, not interleaved
Phase 54 pairs. Therefore 1.536x is a measured cumulative comparison, while
the 1.344x destination-service result is the stronger matched incremental
claim.

The first monolithic attempt used 16 OpenBLAS/OpenMP workers and crashed in an
unchanged pandas allocation during mandatory scheduling. Limiting host math to
one worker made three consecutive runs complete. The reproducible runners now
set OpenBLAS, OpenMP, MKL, and Numba worker counts to one. This is part of the
benchmark configuration, not a GPU arithmetic change.

## Reproduction

Run the matched sharded experiment:

```powershell
.\scripts\run_phase54_sharded_qualification.ps1 -Repetitions 3 -RunTag p54final
```

Run a fresh complete candidate series against an explicitly selected baseline:

```powershell
.\scripts\run_phase32_full_model_ab.ps1 -Repetitions 3 -Households 50000 `
  -RunTag p54fresh -Baseline phase53 -CandidatePhase 54
```

Rebuild the committed qualification (the long argument lists identify every
source report and prevent accidental evidence substitution):

```powershell
.\.venv-phase8\Scripts\python.exe scripts\build_phase54_qualification.py `
  --output benchmark-results\phase54-p54final-qualification.json `
  --phase53-qualification benchmark-results\phase53-p53final-qualification.json `
  --baseline-pre benchmark-results\phase53-p53premandatory-gpu-2.json benchmark-results\phase53-p53premandatory-gpu-3.json benchmark-results\phase53-p53premandatory-gpu-4.json `
  --baseline-post benchmark-results\phase53-p53postmandatory-gpu-2.json benchmark-results\phase53-p53postmandatory-gpu-3.json benchmark-results\phase53-p53postmandatory-gpu-4.json `
  --candidate-pre benchmark-results\phase54-p54final-pre-gpu-1.json benchmark-results\phase54-p54final-pre-gpu-2.json benchmark-results\phase54-p54final-pre-gpu-3.json `
  --candidate-post benchmark-results\phase54-p54final-post-gpu-1.json benchmark-results\phase54-p54final-post-gpu-2.json benchmark-results\phase54-p54final-post-gpu-3.json `
  --full-runs benchmark-results\phase54-p54monolithic2-gpu-1.json benchmark-results\phase54-p54full-gpu-2.json benchmark-results\phase54-p54full-gpu-3.json
```

Run the automated suite:

```powershell
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

## Assumptions and limits

The result covers the public model, scale, GPU, reviewed expression program,
zone mapping, time-period system, skim directions, current sampler ordering,
and current ActivitySim random-ledger semantics. The lease is safe in the
qualified synchronous flow because its producer and consumer share the same
sample object and validated metadata; it is not a general durable cache and
must not cross an asynchronous mutation boundary.

The normal kernel supports the bounded offsets and draw counts used by this
model. A different random distribution, a switch from legacy `RandomState`, a
different sample layout, or another device needs a new compatibility scan and
qualification. Floating diagnostics remain tolerance-bounded; the model's
published choices are exact.

## Next major opportunity

Phase 55 should replace the remaining per-call Python planning and owner-state
upload with a versioned device entity store plus ahead-of-time linked execution
plans. Person, tour, household, and land-use columns should remain authoritative
on the GPU across all destination families; a hash-addressed plan should launch
the packet, wait, utility, nesting, and choice graph without rebuilding Python
objects. Success should require exact decisions, zero fallback, three matched
wins, destination service below 2.5 seconds, five destination components below
10 seconds, and a measured complete-model median below 130 seconds.
