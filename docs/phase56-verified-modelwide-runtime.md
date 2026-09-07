# Phase 56: verified model-wide resident runtime

## Outcome

Phase 56 qualifies the first sub-130-second complete public benchmark in this
project. Three fresh-process Phase 55/56 pairs run the same 34-step Prototype
MTC extended model for 50,000 households on the RTX A4000. Phase 56 wins every
pair, preserves every published modeled decision, and passes every inherited
and new proof gate.

| Measure | Phase 55 median | Phase 56 median | Result |
|---|---:|---:|---:|
| complete lifecycle | 128.300 s | 118.836 s | 1.080x; 7.38% lower |
| `initialize_landuse` | 14.100 s | 0.600 s | 23.500x; 95.74% lower |
| Phase 56 cache validation | - | 0.528 s | charged to candidate total |

The complete lifecycle median saves 9.464 seconds. Individual pair savings
are 8.264, 9.261, and 10.263 seconds. Candidate totals include the Phase 55
prewarm lifecycle cost and full Phase 56 runtime validation; the comparison
does not hide those costs.

For larger-project context, the separately measured three-run regular pinned
ActivitySim median is 205.4 seconds on the same machine and workload. Compared
with Phase 56's 118.836 seconds, the cumulative accelerated stack is **1.728x
faster**, saves 86.564 seconds, and uses 42.14% less time. That is historical
same-benchmark context, not an interleaved Phase 56/CPU experiment; the matched
Phase 55/56 pairs above are the evidence for Phase 56's isolated contribution.

## Why the phase changed direction

The Phase 55 plan proposed a broader device-native entity store. Before
implementing it, a new whole-model profile showed that the already accelerated
destination service consumed only about 2.25 seconds, while every fresh
process spent roughly 14-19 seconds expanding the same OMX skim data into a
6.452 GB Sharrow memory image. Perfecting the destination kernel could not
recover that startup time.

Phase 56 therefore attacks the largest measured removable cost first. It is a
model-wide data-plane optimization around the GPU runtime, not a claim that a
new CUDA arithmetic kernel is faster. The planned general entity store remains
a later opportunity.

## Implementation

ActivitySim and Sharrow already contain a file-backed NumPy memmap mechanism,
but the public Windows environment did not have a production replication
contract around it. Phase 56 adds that contract:

1. An explicit developer build expands the public skims once into
   `cache_sharrow/phase56_taz.mmap`.
2. The builder hashes all 6,452,305,336 cache bytes, Sharrow's layout metadata,
   `skims.omx`, `land_use.csv`, and `network_los.yaml`.
3. A canonical manifest receives its own SHA-256 artifact digest.
4. Normal processes fail closed unless cache size and modification time,
   metadata digest, all source sizes/times/digests, contract version, and
   manifest self-digest agree.
5. ActivitySim receives the verified file-backed store only for the TAZ skim
   tag. All other tags retain their original behavior.
6. The Phase 55 AOT cubin, plan atlas, device sample leases, exact random
   streams, output checks, and zero-fallback rules remain active.

The local 6.452 GB cache is deliberately not committed. The small manifest and
the deterministic build command are committed, so another installation can
rebuild and verify the same artifact from the public inputs.

## Upstream Sharrow compatibility finding

The first build exposed a Sharrow 2.13 defect. After successfully writing the
memmap and metadata, `from_shared_memory` evaluated a multi-element NumPy
memmap as a Boolean. NumPy correctly raises an ambiguous-truth-value error.

Phase 56 installs a narrow process-local bridge. Only when the ownership
argument is a NumPy memmap does it convert that argument to Sharrow's documented
`True` ownership flag; Sharrow then reopens the same file read-only and executes
its unchanged reconstruction logic. The original method is restored after the
ActivitySim process. A unit test proves that the bridge changes only this case.

## Replication design

The benchmark driver launches alternating fresh Phase 55 control and Phase 56
candidate processes. Each candidate is independently compared with the fixed
public reference pipeline. Output directories are deleted only after timing
CSV files and exactness reports are saved.

| Trial | Phase 55 | Phase 56 including validation | Saved | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 127.100 s | 118.836 s | 8.264 s | 1.070x |
| 2 | 128.300 s | 119.039 s | 9.261 s | 1.078x |
| 3 | 128.400 s | 118.137 s | 10.263 s | 1.087x |

All candidates initialize land use in 0.6 seconds. Runtime cache validation is
0.526-0.531 seconds and is included in each Phase 56 total. All three exact
verifiers report zero changed decision cells. All candidate proof gates pass.

## Acceptance gates

The consolidated qualifier refuses release unless all of these are true:

- the out-of-band build exits successfully and verifies the full 6.452 GB hash;
- the checked-in manifest self-digest is valid;
- build and all three runtime processes use the same artifact digest;
- all three runtime cache contracts validate;
- all three `initialize_landuse` measurements are below 3 seconds;
- all inherited candidate proof gates pass;
- all three independent output comparisons are exact;
- Phase 56 wins all three complete-model pairs; and
- the Phase 56 lifecycle median, including validation, is below 130 seconds.

All nine gates pass in
`benchmark-results/phase56-p56formal-qualification.json`.

## Reproduce

Build and seal the local cache once:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\run_phase22_integrated_scheduling.py `
  --project benchmark-data\phase9-mtc-full\prototype_mtc_extended `
  --data benchmark-data\phase9-mtc-full\prototype_mtc_extended\data_full `
  --output benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-p56-cache-build-50000 `
  --config-overlay benchmark-data\configs_phase33_choiceforge `
  --config-overlay benchmark-data\phase9-mtc-full\prototype_mtc_extended\configs_sh `
  --full-model --stop-after-model initialize_landuse --households-sample-size 50000 `
  --native-abi-live `
  --reference-pipeline benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-p17modeproof16-baseline-50000-1\pipeline.parquetpipeline `
  --report benchmark-results\phase56-cache-build.json `
  --checkpoint benchmark-results\phase56-cache-build-checkpoint.json `
  --kernel-reports benchmark-results\phase56-cache-build-kernels `
  --phase56-modelwide-resident-runtime --phase56-build-skim-cache
```

Run the replicated comparison and consolidated validator:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p56formal `
  -Baseline phase55 -CandidatePhase 56 -CleanupOutputs
.\.venv-phase8\Scripts\python.exe scripts\build_phase56_qualification.py
```

The build mode refuses to overwrite an existing cache. Remove or archive the
three exact cache artifacts (`.mmap`, `.meta.pkl`, and `.choiceforge.json`)
before deliberately rebuilding.

## Claim boundary and assumptions

The result applies to this fixed public dataset, configuration, software lock,
Windows host, RTX A4000, 50,000-household sample, and current cache contract.
Changing any hashed source invalidates the cache. A different model, zone
system, Sharrow layout, or GPU environment requires its own qualification.

The 1.080x number is Phase 56 versus the already GPU-accelerated Phase 55 stack.
The separately measured cumulative GPU-versus-regular-ActivitySim context is
1.728x; neither number is an isolated kernel speedup. Phase 56 makes repeated
complete runs faster by safely reusing invariant data. The initial one-time cache build takes about 23 seconds
for ActivitySim through `initialize_landuse`, plus about 4.7 seconds to seal and
fully recheck the artifact. That cost is amortized across later runs.

The benchmark also shows normal component noise outside initialization; several
unchanged components were slightly faster or slower in the candidates. The
replicated claim is based on the predeclared complete lifecycle, not on claiming
that every individual row improved.

## What should come next

Phase 57 should pursue the deferred general device-native entity store where
the profile supports it, while treating Phase 56 as the standard launch path.
The largest remaining medians are mandatory tour scheduling (about 14.0 s),
trip mode choice (9.9 s), trip destination (9.8 s), non-mandatory tour
frequency (7.2 s), and non-mandatory scheduling/write matrices (about 6.6 s
each). The next major phase should fuse one complete semantic component at a
time, remove its pandas expansion and Python orchestration boundaries, retain
the exact random contract, and demand three fresh matched wins. A meaningful
next target is below 105 seconds complete lifecycle, not another tiny kernel
micro-optimization.
