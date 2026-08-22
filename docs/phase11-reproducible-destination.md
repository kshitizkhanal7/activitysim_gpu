# Phase 11: reproducible destination superiority

## Claim

ChoiceForge's production configuration accelerates trip destination choice
while preserving byte-identical final model outputs on the public full
Prototype MTC Extended geography. This claim applies to the explicit
destination backend only. Tour scheduling remains on ActivitySim.

## Experimental design

The benchmark ran 50,000 sampled households over all 1,454 zones in an
interleaved fresh-process sequence: A1/B1/A2/B2/A3/B3. A is pinned ActivitySim
with Sharrow; B adds only ChoiceForge trip-destination batching and the
canonical MTC-21 CUDA nested-logsum reducer. All six runs used the same public
data hash, source patch hash, environment-lock hash, configuration hashes,
Python version, GPU model, and driver recorded in the run manifest.

## Result

| Metric | Baseline median | ChoiceForge median | Result |
| --- | ---: | ---: | ---: |
| All 34 model steps | 202.492 s | 190.380 s | 1.064x; 12.112 s saved |
| Trip destination | 39.7 s | 28.6 s | 1.388x; 11.1 s saved |

Every optimized whole-model run beat every baseline run. The conservative
worst-optimized versus best-baseline whole-model speedup is 1.038x. The three
paired whole-model savings were 12.172, 12.112, and 7.596 seconds; the
deterministic bootstrap interval for their median is 7.596--12.172 seconds.
Three repetitions are evidence of repeatability on this workstation, not a
general population-performance guarantee.

The seven substantive `final_*.csv` files are byte-identical across all six
runs: accessibility, households, joint-tour participants, land use, persons,
tours, and trips. The machine-readable result is
`benchmark-results/phase11-50k-replicated-summary.json`.

## Rebuild guarantee

`integration/activitysim-current-choiceforge.patch` now exactly reproduces
the tracked changes against pinned ActivitySim commit
`16ab11180a26912987eb902daf945e268f3efc11`. Verify it without modifying the
working checkout:

```powershell
.\scripts\verify_activitysim_patch.ps1
```

`requirements-phase11-lock.txt` records the installed environment used for
the result. Future manifests include the hashes of this lock, the integration
patch, and each configuration tree, as well as sampled GPU utilization and
memory. The runner's old launcher-memory values remain diagnostic only; the
ActivitySim high-water markers and GPU samples are the useful resource record.

## Destination profile

The CUDA nested reducer was instrumented in the three optimized runs. Across
90 reductions it processed 12,564,936 rows (2.111 GB of utility matrices):
1.623 s host-to-device, 0.603 s GPU kernel, and 0.060 s device-to-host. This
shows that reducer fusion alone cannot recover the 11.1-second destination
gain again. The main remaining opportunity is to move utility and skim
evaluation upstream of the host DataFrame, not merely to tune the reduction.

The reducer now also accepts device-resident utility matrices without an upload,
which is the clean integration boundary for a future Sharrow/CuPy evaluator.
ActivitySim currently supplies a host pandas matrix, so no unsupported claim of
full GPU utility evaluation is made.

## Scheduling safety

The shadow guard now fails closed if ActivitySim RNG offsets cannot be
snapshotted and rewinds offsets before ActivitySim fallback after a GPU
exception. It protects replication but is intentionally not enabled for
performance reporting. The 50k guarded finding from Phase 10 still stands:
the unguarded 32-bit scheduler has observed real near-boundary disagreements.

## Phase 12 gate

Do not enable unguarded GPU scheduling. The next kernel project should be a
device-side destination utility/skim evaluator with a byte-identical A/B gate.
It must first reproduce one trip-purpose batch against ActivitySim, then the
complete 50k A1/B1/A2/B2/A3/B3 benchmark. A 100k benchmark belongs on a
higher-memory host; generic chunking on this workstation has already shown
that it trades memory safety for unacceptable throughput.
