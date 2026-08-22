# Phase 8 results: current ActivitySim at 50,000 households

## Outcome

Phase 8 closes the most important scale and currency gaps in the earlier
evidence. ChoiceForge now runs through an opt-in patch against pinned current
ActivitySim commit `16ab11180a26912987eb902daf945e268f3efc11` and the official
public `sharrow-contrast/mtc_mini` workflow at 50,000 households and 190 zones.

In three fresh-process interleaved A/B pairs, median whole-model runtime falls
from 147.225 to 131.812 seconds: **1.117x faster**, saving 15.413 seconds. The
four tour-scheduling components are **1.270x faster** together, and complete
trip destination is **1.417x faster**. All three optimized runs are faster than
all three baselines. Seven substantive final CSVs are byte-identical across all
six runs.

This remains evidence for one public model, one Windows workstation, and one
GPU. It is much stronger evidence than a synthetic kernel timing, but it is not
a claim that every ActivitySim model or every GPU will improve.

## Pinned system

| Item | Phase 8 value |
|---|---|
| ActivitySim | `1000.dev1+g16ab11180` |
| Git commit | `16ab11180a26912987eb902daf945e268f3efc11` |
| Workflow | `sharrow-contrast/mtc_mini` |
| Data | Public `prototype_mtc_sf` |
| Scale | 50,000 households, 190 zones |
| Python | 3.11.14 |
| NumPy / pandas | 2.4.6 / 2.3.3 |
| CuPy / CUDA runtime | 14.1.1 / 12.9 |
| GPU | NVIDIA RTX A4000, 16 GB |
| NVIDIA driver | 571.59 |
| Host | 48 logical CPUs, 63.9 GB RAM |
| OpenBLAS threads | 24 in all measured production trials |

CuPy 14 requires CUDA runtime headers for runtime kernel compilation. The GPU
extra therefore uses `cupy-cuda12x[ctk]>=14,<15`; after installation, the A4000
executed a compiled smoke-test kernel and all 33 project tests passed.

## Compile and pilot gates

ActivitySim's official workflow first performs a 500-household `sharrow: test`
compile pass. That pass completed all 34 models, used about 15.8 GB resident
memory, and accumulated 1,755.5 seconds of model timers. Its largest entries
were trip destination at 630.1 seconds, trip mode choice at 244.1 seconds, CDAP
at 177.0 seconds, and mandatory scheduling at 132.5 seconds. These are compile
and comparison costs, not warmed production costs, and are excluded from every
A/B result.

A launcher audit caught an initial direct run that inherited `sharrow: test`.
It was stopped before completion and excluded. All valid production trials use
the workflow's intended `sharrow: require` overlay and the prebuilt cache.

Before the large experiment, a warmed 5,000-household gate compared fresh
baseline and ChoiceForge processes. All seven substantive CSVs matched byte for
byte. Trip destination fell from 18.5 to 11.2 seconds and whole-model time from
67.839 to 59.686 seconds. This gate also exposed and fixed the current API's
removal of the historical `los.THREE_ZONE` symbol before large trials began.

## Interleaved 50,000-household proof

The order is A1/B1/A2/B2/A3/B3. A is pinned current ActivitySim with Sharrow
required. B changes only the explicit ChoiceForge scheduling and destination
overlays. Every run is a fresh process; ActivitySim's own timers are used, and
no setup time is subtracted.

| Boundary | Baseline samples (s) | ChoiceForge samples (s) | Median speedup |
|---|---|---|---:|
| All 34 models | 155.190, 147.225, 146.863 | 133.410, 131.812, 131.384 | **1.117x** |
| Four scheduling components | 30.4, 30.1, 29.9 | 24.4, 23.7, 23.7 | **1.270x** |
| Mandatory scheduling | 20.4, 20.5, 20.4 | 17.3, 17.1, 17.0 | **1.193x** |
| Joint scheduling | 1.0, 0.9, 0.9 | 0.8, 0.7, 0.7 | **1.286x** |
| Non-mandatory scheduling | 7.8, 7.6, 7.5 | 5.4, 5.1, 5.2 | **1.462x** |
| At-work scheduling | 1.2, 1.1, 1.1 | 0.9, 0.8, 0.8 | **1.375x** |
| Trip destination | 35.1, 30.9, 30.9 | 21.9, 21.8, 21.6 | **1.417x** |

Whole-model paired savings are 21.780, 15.413, and 15.479 seconds. Even the
slowest optimized run is faster than the fastest baseline; the conservative
worst-optimized versus best-baseline comparison is 1.101x.

## Correctness gate

The summarizer calculates SHA-256 for every substantive final table and raises
an exception on a mismatch. `final_checkpoints.csv` is excluded because it
contains run timing metadata. The following seven files are byte-identical in
all six trials:

- `final_accessibility.csv`
- `final_households.csv`
- `final_joint_tour_participants.csv`
- `final_land_use.csv`
- `final_persons.csv`
- `final_tours.csv`
- `final_trips.csv`

The checked outputs contain 50,000 households, 111,130 persons, 142,761 tours,
and 350,751 trips. This is exact file equality, not an average-error or sampled-
row check.

## What the live telemetry says

Each optimized run makes 16 scheduling calls covering 21,045,645 feasible
alternative rows. The largest real call contains 9,561,750 rows for 50,325
choosers in a 327.920 MB compact package. Across the three trials, its measured
GPU/backend interval is 389.061, 424.513, and 410.728 milliseconds; complete
lowering, stateful preparation, packing, ActivitySim random-number retrieval,
GPU work, and result mapping take 3.247, 3.286, and 3.305 seconds.

This is an important optimization result: the kernel is no longer the largest
cost at that boundary. Stateful timetable preparation and random-number
retrieval on the host dominate. More kernel tuning alone cannot deliver the
next large scheduling gain.

Trip destination forms three trip-number batches with 2,381,612, 605,782, and
198,736 directional-logsum rows across ten purposes each. Sampling, final
simulation, and random streams remain ActivitySim-owned and purpose-specific.

## Independent large-boundary replay

An observer captured the evaluated 21-mode utilities immediately before the
nested-logit reduction in the 50,000-household optimized run. The capture has
30 real batches, 3,186,130 rows, and a 535.270 MB FP64 utility matrix. It is
about 29.5 times larger than the Phase 7 capture.

Eleven warmed trials alternate CPU-first and GPU-first order. ActivitySim's
pandas reducer has a 4.646627-second median. ChoiceForge, including host-to-
device transfer, CUDA kernels, and device-to-host results, has a 0.125627-second
median: **36.988x faster**, saving 4.521001 seconds at this exact captured
boundary. Every GPU trial is faster than every CPU trial. Maximum absolute
logsum error is `5.329070518200751e-15`.

The raw capture is under
`benchmark-results/phase8-nested-logsum-capture`; the machine-readable result
is `benchmark-results/phase8-nested-logsum-summary.json`. This isolated 36.988x
number is not a whole-model claim. Expression evaluation, sampling, table work,
and all other components explain why the measured full-model gain is 1.117x.

## Reproduction

The current integration patch is
`integration/activitysim-current-choiceforge.patch`. The exact environment
pins are in `requirements-phase8.txt`; explicit overlays are in
`benchmark-data/configs_phase8_choiceforge`.

```powershell
pwsh scripts\run_phase8_interleaved.ps1 -Households 50000 -Repetitions 3
.venv-phase8\Scripts\python.exe benchmarks\benchmark_phase8_activitysim.py
```

The machine-readable result is
`benchmark-results/phase8-activitysim-summary.json`. The run-order manifest is
`benchmark-results/phase8-interleaved-runs.json`. ActivitySim's official
workflow documentation explains the benchmark runner and compile/run split:
<https://activitysim.github.io/activitysim/develop/dev-guide/workflows.html>.

## Limits and next work

- The result covers one public model specification and one GPU.
- The MTC workflow uses 190 zones; it does not prove performance for a much
  larger zone system or a different nested-logit topology.
- Estimation, tracing targets, explicit error terms, three-zone destination
  path building, and unsupported expressions still take conservative
  ActivitySim fallbacks.
- The next high-return engineering work is persistent timetable encodings,
  faster keyed-random retrieval, and more compiled preprocessing.
- Independent reproduction on Linux and a second GPU generation is needed
  before making a broad portability claim.
