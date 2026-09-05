# Phase 52: persistent, prewarmed, four-row destination service

## Outcome

Phase 52 turns the Phase 51 fused destination kernel into a reusable service.
It preserves Phase 51's exact arithmetic and dense-device-ABI elimination, but
adds checked-in precompiled source, four-row owner-local tiles, cached semantic
and native plans, reusable device workspaces, and an explicit early-release
boundary.

On the public Prototype MTC Extended model at 50,000 households, Phase 52 won
all three matched comparisons against the already accelerated Phase 51 system:

| Incremental measurement | Phase 51 median | Phase 52 median | Result |
|---|---:|---:|---:|
| fused kernel pipeline | 2.030 s | 1.657 s | 1.225x; 18.36% lower |
| instrumented destination service | 9.069 s | 7.846 s | 1.156x; 13.48% lower |
| five destination components | 16.9 s | 16.2 s | 1.043x; 4.14% lower |
| complete 34-step lifecycle | 139.0 s | 136.527 s | 1.018x; 1.78% lower |

Every line won three of three matched pairs. These are Phase 52's isolated
incremental gains; they are not inferred from a CPU comparison.

Against regular pinned ActivitySim with Sharrow required, the complete system
through Phase 52 achieved:

| Cumulative measurement | ActivitySim median | Phase 52 median | Result |
|---|---:|---:|---:|
| complete 34-step lifecycle | 205.4 s | 139.086 s | 1.477x; 32.29% lower |
| five destination components | 43.2 s | 16.3 s | 2.650x; 62.27% lower |

Those larger figures are cumulative gains from Phases 1-52, not a claim that
Phase 52 alone caused the complete difference.

## What changed

### A checked-in, hash-verified CUDA program

The qualified public kernel is
`src/choiceforge/kernels/phase52_public_destination_tile4.cu`. Its SHA-256 is:

`599a9704be0992d2863320390cbca0028c7a578ecacf72d69de2e658a5d79906`

The runtime reads and hashes this source, asks CuPy to compile it before
ActivitySim component timing begins, and relies on CuPy's content-addressed
on-disk binary cache across fresh processes. Each call records the source hash.
The qualification rejects a run if a call uses another program or compiles the
fused kernel inside the timed destination service.

### Four sampled rows per owner-local tile

Phase 51 assigned one CUDA block to one sampled destination row. Phase 52
assigns a block to four adjacent rows. The block loads the compact owner fields
once into shared memory, loads the four row-specific destination/time states,
then evaluates the same 315 terms and 21 alternatives independently for each
row. This improves owner-data locality without serializing an entire chooser's
sample.

Two-row tiling was implemented and tested first. Four rows produced the better
live kernel time on this GPU and was promoted into the checked-in program. The
tile width is part of the proof contract rather than a silent tuning default.

### Persistent plans and workspaces

The service caches two kinds of immutable plan:

- semantic plans map the reviewed expression specification and purpose to its
  exact compact inputs;
- native ABI plans preserve ordered CUDA argument layout, scalar constants,
  and skim-coordinate groups.

It also grows and reuses device buffers for compact packets, row-owner maps,
and output utilities. Capacities use a grow-only power-of-two policy during the
19 calls. Across every qualified run this produced 9 semantic-plan hits, 9
native-plan hits, 16 utility-workspace hits, 152 compact-packet workspace hits,
and 16 row-owner workspace hits.

### Minimal bootstrap and explicit lifecycle release

The strict ABI bootstrap now supports a one-row output allocation. The actual
4,696,676-row utility output comes from the reusable Phase 52 workspace, so the
compatibility object retains only 7,904 bytes across all 19 calls.

After `atwork_subtour_destination`, the runtime releases its plans and device
workspaces, runs Python garbage collection, and returns unused CUDA and pinned
allocator blocks. The live shared skim store is retained until the final GPU
skim consumer. Each qualified run freed 101,630,976 bytes at this early
boundary. This hardening was added after a complete 19-call experimental run
encountered a later host-allocation failure in ActivitySim trip destination.

## Preserved contracts

Phase 52 did not weaken the Phase 51 proof boundary:

- 19 destination-logsum calls;
- 201,390 owners and 4,696,676 sampled alternative rows per run;
- exactly 10 floating row sources, 31 integer row sources, and six directed
  skim-coordinate groups;
- 1,953,817,216 bytes of dense device ABI allocation eliminated across calls;
- 1,953,817,216 bytes of dense host packing avoided;
- only 47,787,016 compact bytes uploaded, avoiding 1,906,030,200 net bytes;
- one int32 row owner per sampled row (18,786,704 aggregate bytes);
- no generic generator, dense preprocessor, or CPU fallback.

The implementation fails closed on changed source hashes, expression schema,
coordinate direction, zone assumptions, noncontiguous owner groups, or int32
range violations.

## Accuracy and replication

Two independent three-pair experiments were run in fresh processes:

1. Phase 51 control followed by Phase 52 candidate, repeated three times.
2. Regular ActivitySim control followed by Phase 52 candidate, repeated three
   times.

All six independent output verifiers found zero changed modeled decision cells.
In the incremental experiment, five output CSVs were byte-identical. Floating
logsum diagnostics are intentionally treated as bounded diagnostics: maximum
observed destination difference was about `3e-6`, below the `1e-4` gate; school
and workplace maxima were about `1.91e-6`, below their `1e-5` gates. Final
choices did not change.

Machine-readable evidence:

- `benchmark-results/phase52-p52final-qualification.json`
- `benchmark-results/phase52-p52final-summary.json`
- `benchmark-results/phase52-p52cpu-summary.json`
- `benchmark-results/phase52-p52final-{base,gpu,exact}-1.json` through `-3.json`
- `benchmark-results/phase52-p52cpu-{gpu,exact}-1.json` through `-3.json`

Rebuild the consolidated proof with:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\build_phase52_qualification.py `
  --incremental-summary benchmark-results\phase52-p52final-summary.json `
  --cpu-summary benchmark-results\phase52-p52cpu-summary.json `
  --output benchmark-results\phase52-p52final-qualification.json
```

Re-run the two timing campaigns with:

```powershell
.\scripts\run_phase32_full_model_ab.ps1 -Repetitions 3 -Households 50000 `
  -RunTag p52final -Baseline phase51 -CandidatePhase 52

.\scripts\run_phase32_full_model_ab.ps1 -Repetitions 3 -Households 50000 `
  -RunTag p52cpu -Baseline activitysim -CandidatePhase 52
```

Run all automated tests with:

```powershell
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

## Assumptions and claim limits

The qualification covers the public Prototype MTC Extended workload at 50,000
households and 1,454 zones on this machine and GPU. It assumes the reviewed
315-term, 21-mode destination-logsum expression; five skim periods; current
direction mappings; contiguous stable chooser groups; no chunking of these
location-logsum calls; and current land-use and compact integer contracts.

Timing is measured in matched control/candidate order, not randomized order.
Three pairs reduce but do not eliminate operating-system, thermal, and cache
noise. The all-three-pair rule, low-level service metrics, and exact-output
verification make the incremental claim much stronger than a single wall-clock
run. The result does not claim universal support for arbitrary ActivitySim
models or hardware.

## Next major opportunity

The kernel is no longer the dominant Phase 52 service cost: the median tiled
pipeline is 1.657 seconds, while compact packet preparation remains roughly
4.3 seconds. Phase 53 should therefore build a model-wide device-resident
destination data plane, not merely tune another kernel constant.

The sampling stage should emit canonical owner IDs, sampled destination IDs,
and row offsets directly into persistent device tables. The destination service
should gather owner and land-use facts on-device, retain purpose-invariant
packets across related calls, and feed logsum, probability, and choice without
returning through pandas-shaped intermediate packets. Capturable fixed-shape
segments can then use CUDA Graph replay. A successful Phase 53 should retain all
Phase 52 exactness and hash gates, eliminate most of the remaining 4.3-second
host preparation cost, win three pairs, and target the five destination
components below 14 seconds before claiming success.
