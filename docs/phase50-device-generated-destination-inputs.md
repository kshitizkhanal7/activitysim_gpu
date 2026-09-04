# Phase 50: device-generated destination inputs

## Result

Phase 50 replaces ActivitySim's dense pandas destination-logsum preprocessor
with a fail-closed compact-to-CUDA compiler. It sends each chooser's state
once, sends sampled destination IDs, retains the reviewed land-use and skim
tables on the GPU, and generates the exact 10 floating-point fields, 31
integer fields, and six skim-coordinate groups needed by the 315-term,
21-alternative public MTC tour-mode utility.

This is the first destination phase large enough to produce a repeated
multi-second component result. Three fresh-process 50,000-household Phase
49/50 pairs produced the following medians:

| Measurement | Phase 49 | Phase 50 | Result |
|---|---:|---:|---:|
| five destination components | 28.4 s | 16.9 s | 1.680x; 40.5% lower |
| complete 34-step model | 148.700 s | 136.518 s | 1.089x; 8.19% lower |
| target-component pairs won | - | 3 of 3 | replicated |
| complete-model pairs won | - | 2 of 3 | median gain, not every-pair superiority |
| changed modeled decision cells | - | 0, 0, 0 | exact |

All seven published CSV files are byte-identical in every incremental pair.
The same-machine cumulative comparison against regular pinned ActivitySim
with Sharrow required also ran three fresh pairs. Phase 50 won all three and
reduced the complete-model median from 209.300 to 134.118 seconds: 75.182
seconds saved, 35.92% lower, or **1.561x faster**. This cumulative comparison
measures all project improvements through Phase 50; it does not attribute all
75 seconds to Phase 50 alone.

The complete-model candidates changed by -0.518, +12.182, and +11.883
seconds relative to their paired controls. The first candidate lost because
unmodified steps were unusually slow; its five targeted components still
improved by 9.6 seconds. The promoted causal claim is therefore the repeated
five-component gain. The 8.19% lifecycle median is real observed evidence,
but not a claim that every full run will be faster.

## What changed

The Phase 49 path still did this before each mode-logsum kernel:

1. repeat chooser attributes for every sampled destination;
2. evaluate the pandas preprocessor, including availability expressions;
3. resolve 41 row-source bindings and six skim-coordinate groups;
4. pack the dense host arrays;
5. upload them to CUDA; and
6. execute the utility and nested-logit kernels.

Phase 50 intercepts the public `compute_location_choice_logsums` contract
before step 1. Its compact packet contains owner offsets, four owner float
fields, thirteen owner integer fields, origin, two time-period codes,
duration, sampled destination IDs, and an owner-by-five-density-band wait-time
table. A generated CUDA kernel reconstructs the reviewed dense ABI and all
skim coordinates directly on the device. The existing strict IR evaluator
and Phase 49 handoff consume those arrays without returning the 4,696,676 mode
logsums to the CPU.

The code is public-MTC-specific on purpose. It does not pretend that arbitrary
ActivitySim preprocessors are already supported.

## Compact controlled random numbers

The preprocessor uses six identity-keyed random values per chooser for taxi,
single-TNC, and shared-TNC waits at the origin and destination. ActivitySim's
`broadcast=True` implementation internally generates those values for unique
chooser IDs and then expands them back over all sampled alternatives.

Phase 50 calls the same registered random channel directly on that exact
unique-ID series with `broadcast=False`. It advances the same per-ID ledger by
the same six values, but never creates the dense 4,696,676 by 6 temporary. A
first complete implementation used the ordinary broadcast and already passed
the output proof. The compact call then passed the complete proof again and
reduced Phase 50 preparation time.

## Memory and execution accounting

Each qualified run covers 19 calls, 4,696,676 sampled rows, and 201,390
owners. The figures below are identical in all three production reports.

| Boundary quantity | Dense Phase 49 path | Compact Phase 50 path | Change |
|---|---:|---:|---:|
| row fields | 192,563,716 values | generated on CUDA | host expansion removed |
| host-to-device input bytes | 1,953,817,216 B | 58,259,296 B | 1,895,557,920 B avoided |
| upload reduction | - | - | 97.02% |
| binding-resolution calls | 19 | 0 | eliminated |
| dense host-pack calls | 19 | 0 | eliminated |
| fallback calls | - | 0 | fail closed |
| CUDA input generation, median | - | 0.262 s | all 19 calls |
| strict utility kernels, median | - | 1.607 s | all 19 calls |
| complete Phase 50 logsum service, median | - | 8.892 s | includes compile and compact preparation |

The dense byte count includes the 10 float32, 31 int64, and six grouped
coordinate surfaces required by the existing strict utility ABI. Phase 50
still materializes those arrays on the GPU; it removes their construction and
transport on the host. Eliminating the device materialization itself is the
next major optimization.

## Exactness and replication

Correctness is established at several levels:

1. Unit tests check contiguous owner topology, stable owner fields, period
   conversion, wait-time reconstruction, accounting, and fail-closed cases.
2. Every live call requires exactly 10 float sources, 31 integer sources, and
   six skim-coordinate groups. An unfamiliar source or direction stops the
   run.
3. All 19 calls must cover the exact public workload and finish with zero
   fallback, zero pending Phase 49 packets, and the expected compiler-cache
   lifecycle.
4. The integrated runner checks its pinned ActivitySim reference during every
   candidate run.
5. An independent verifier compares the published accessibility, household,
   participant, land-use, person, tour, and trip tables after each pair.

All three Phase 49/50 production pairs have zero changed modeled decision
cells and all seven published files are byte-identical. In the separate
regular-ActivitySim comparison, four published files are byte-identical and
floating logsum diagnostics are bounded: the maximum destination-logsum
difference is 1.0e-5 against a 1.0e-4 gate, and the maximum
mode-choice-logsum difference is 3.814e-6 against a 1.0e-5 gate. These are
inherited declared diagnostic tolerances; choices, destinations, modes, and
schedules remain exact.

## Assumptions and fail-closed boundaries

The qualified backend assumes:

- the public Prototype MTC extended model and its reviewed 315-term mode spec;
- the dense zero-based 1,454-zone universe;
- contiguous sampled alternatives for each stable chooser ID;
- the current no-transit-subzone-access-distance configuration;
- the five reviewed skim directions and five time-period labels;
- unchunked location-logsum calls;
- stable owner attributes inside each sample group; and
- the declared ActivitySim controlled-random channel.

Land-use content is hashed when first installed on the device. Changed land
use, missing fields, a reordered/noncontiguous owner, an unknown period, a new
row source, a changed skim direction, or an unsupported chunking contract
raises an error instead of falling back to a numerically different path.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path src)

.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p50final `
  -Baseline phase49 -CandidatePhase 50

.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p50cpu `
  -Baseline activitysim -CandidatePhase 50

.\.venv-phase8\Scripts\python.exe scripts\build_phase50_qualification.py `
  --incremental-summary benchmark-results\phase50-p50final-summary.json `
  --cpu-summary benchmark-results\phase50-p50cpu-summary.json `
  --output benchmark-results\phase50-p50final-qualification.json

.\.venv-phase8\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Primary evidence:

- `benchmark-results/phase50-p50final-summary.json`
- `benchmark-results/phase50-p50cpu-summary.json`
- `benchmark-results/phase50-p50final-qualification.json`
- `benchmark-results/phase50-p50final-{base,gpu,exact}-{1,2,3}.json`
- `src/choiceforge/destination_input_supergraph.py`
- `tests/test_destination_input_supergraph.py`

## Next major phase

Phase 51 should fuse input reconstruction with strict utility evaluation. The
current generator writes roughly 1.95 GB of dense device ABI, and the utility
kernel immediately reads it. A compact-source expression compiler can load
owner, destination, land-use, and skim values only when a term needs them,
accumulate the 21 utilities directly, and feed nested logit without the
intermediate arrays.

That phase should also precompile the ten purpose-specific plans and keep
compact owner tables resident across related calls. Its success gate should
be another three-of-three win at the five-component boundary, zero changed
modeled decisions, bounded diagnostics, zero fallback, and a target median
below 12 seconds. If achieved, it would remove another roughly five seconds
from the destination chain and make a further visible lifecycle dent.
