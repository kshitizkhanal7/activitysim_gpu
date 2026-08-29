# Phase 38: normalized directional trip state

## Result

Phase 38 removes most of the remaining repeated host-to-device input for the
trip-destination utility runtime without changing a modeled answer. On the
complete public Prototype MTC benchmark, 30 CUDA programs evaluated 4,188,312
directional candidate rows using a 64,171,392-byte normalized packet. The same
work used 351,818,208 bytes in Phase 37, so Phase 38 removes 287,646,816 bytes,
or 81.76%, at this boundary.

The 50,000-household, 1,454-zone model completed all 34 ActivitySim steps with
zero CUDA fallback. Independent verification against Phase 37 found zero
changed decision cells, zero destination-logsum difference, zero mode-logsum
difference, and seven of seven published CSV files byte-for-byte identical.

## Why Phase 37 still repeated data

Phase 37 fused input preparation and utility calculation, eliminating the
large device ABI. Its compact packet was still 84 bytes for every sampled
destination row. A trip with 50 candidate destinations therefore repeated its
tour mode, person age, household vehicles, value of time, duration, and other
stable facts 50 times in each direction.

Phase 38 splits that packet into two normalized tables:

| Table | Contract | Full-run bytes |
|---|---:|---:|
| candidate row | two `int32` coordinates plus one `int32` state selector = 12 bytes/row | 50,259,744 |
| directional state | twelve `int32` facts, two `float64` facts, three `float32` waits = 76 bytes/state | 13,911,648 |
| **total** | exact sum of both contracts | **64,171,392** |

There are 4,188,312 candidate rows, 91,524 unique trips, and 183,048 state
rows. The state count is exactly twice the trip count because ActivitySim
advances three controlled wait draws independently for the outbound and inbound
mode-choice frames. Merging those directions would be smaller but wrong.

## Implementation

`trip_logsum_native.py` now validates the paired directional layout before
normalizing it. For each trip it verifies that all candidate rows agree on the
stable trip, tour, person, and household facts. For each trip direction it also
verifies that all candidate rows carry identical controlled wait draws. A
changed fact, changed draw, unexpected row ordering, missing pair, or invalid
layout raises an error; it never silently uses the older evaluator.

The generated fused kernel receives candidate coordinates, a state selector,
and normalized directional state. It uses the selector to reconstruct strict
inputs in registers, gathers resident land-use and skim values, evaluates the
379 utility terms, and writes only 21 utilities. No dense 11-float/45-integer
ABI or grouped coordinate arrays are materialized.

Five grow-only GPU workspaces hold the two normalized tables' arrays. They are
allocated to the largest size seen so far and filled again for later purpose
programs rather than allocated from scratch. The full run recorded 120 reuse
hits across the 30 calls. The existing post-`trip_mode_choice` release point
still clears native GPU state exactly once.

## Full public-benchmark evidence

| Measure | Phase 38 result |
|---|---:|
| households | 50,000 |
| zones | 1,454 |
| ActivitySim steps | 34 of 34 |
| normalized CUDA programs | 30 of 30 |
| candidate rows | 4,188,312 |
| unique trips | 91,524 |
| directional state rows | 183,048 |
| Phase 37 compact packet | 351,818,208 bytes |
| Phase 38 normalized packet | 64,171,392 bytes |
| packet bytes removed | 287,646,816 (81.76%) |
| normalized packet build time, summed | 1.3616 s |
| normalized upload time, summed | 0.0117 s |
| fused utility kernel time, summed | 2.4671 s |
| nested-logit kernel time, summed | 0.0569 s |
| device ABI arrays still eliminated | 1,692,078,048 bytes |
| coordinate arrays still eliminated | 268,051,968 bytes |
| reusable GPU workspaces | 5 |
| workspace reuse hits | 120 |
| largest compatibility bootstrap | 468 bytes/program |
| CUDA fallbacks | 0 |
| changed decision cells | 0 |
| maximum destination-logsum difference | 0 |
| maximum mode-logsum difference | 0 |
| byte-identical final CSVs | 7 of 7 |
| all-model-step time, descriptive only | 209.3 s |

The measured Phase 37 artifact built packets in 4.4714 seconds and uploaded
them in 0.0805 second; Phase 38 recorded 1.3616 and 0.0117 seconds. That is a
descriptive 3.28x build-time and 6.89x upload-time reduction. Phase 38's fused
kernel itself recorded 2.4671 seconds versus 2.1572 seconds in Phase 37. These
runs were not an alternating matched experiment, so none of those comparisons
is promoted as a causal speed claim. The byte counts and exact outputs are the
qualified claims.

## Proof ladder

1. Unit tests prove correct representatives/selectors and exact 12/76-byte
   contracts, and prove that unstable facts or directional waits fail closed.
2. A 500-household shadow run evaluated Phase 37 and Phase 38 utilities in the
   same process: zero utility mismatches and maximum absolute difference zero.
3. A full fresh 50,000-household run passed every Phase 38 structural gate.
4. The independent output verifier compared Phase 38 with the frozen Phase 37
   output: zero changed decisions and seven byte-identical published tables.
5. The complete repository regression suite passed: 158 tests.

An earlier 500-household shadow attempt completed and matched internally, but
its report used a 50,000-household reference pipeline and therefore failed the
global harness gate. It is not qualification evidence. The corrected shadow
run used the matching 500-household reference and is the cited artifact.

## Reproduction

Run the full candidate:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\run_phase22_integrated_scheduling.py `
  --project benchmark-data\phase9-mtc-full\prototype_mtc_extended `
  --data benchmark-data\phase9-mtc-full\prototype_mtc_extended\data_full `
  --output benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-phase38 `
  --config-overlay benchmark-data\configs_phase33_choiceforge `
  --config-overlay benchmark-data\phase9-mtc-full\prototype_mtc_extended\configs_sh `
  --full-model --households-sample-size 50000 --native-abi-live `
  --phase38-normalized-trip-state `
  --reference-pipeline benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-p37full1-gpu-50000\pipeline.parquetpipeline `
  --report benchmark-results\phase38-local-gpu.json `
  --checkpoint benchmark-results\phase38-local-checkpoint.json `
  --kernel-reports benchmark-results\phase38-local-kernels
```

Run exact final-output verification:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\verify_phase15_outputs.py `
  --reference benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-p37full1-gpu-50000 `
  --candidate benchmark-data\phase9-mtc-full\prototype_mtc_extended\o-phase38 `
  --output benchmark-results\phase38-local-exact.json
```

Run three alternating Phase 37/38 pairs on a quiet machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase38_normalized_trip_ab.ps1
```

The timing wrapper verifies complete output after every pair. Stopwatch results
should be accepted only when competing GPU work, temperature, clocks, power,
and operating-system activity are controlled.

## Claim boundary and next major phase

Phase 38 proves normalization, exact replication, full public-model coverage,
and an 81.76% reduction in the compact packet. It does not prove a new matched
whole-model speedup, hardware peak-memory reduction, or a GPU-only ActivitySim
model. Pandas orchestration, sampling, retry control, most model components,
and file output still run on the CPU.

The next large opportunity is not another tiny packet optimization. Phase 39
should make trip scheduling and destination iteration one resident GPU service:
advance different tour chains in parallel, preserve exact trip order within a
tour, preserve retry behavior and the controlled random ledger, and feed the
normalized utility runtime without rebuilding pandas frames between attempts.
Qualification must include changed-world mutations, hardware memory/occupancy
counters, exact complete outputs, and quiet alternating timings. That boundary
can remove orchestration and repeated sampling work large enough to affect the
complete model materially.
