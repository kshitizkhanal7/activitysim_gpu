# Phase 28: named semantic input generation on CUDA

## Outcome

Phase 28 replaces every anonymous chooser-response dictionary in the six real
mandatory-tour mode-choice programs with a declared CUDA expression. One
floating input, `daily_parking_cost`, is generated from one compact rate per
tour and the exact scheduling duration. Fourteen integer availability inputs
are generated directly from resident raw skim cubes, compact coordinates, and
auto ownership.

The public 50,000-household proof used three fresh Python/CUDA processes, five
complete resident replays per process, and five additional changed synthetic
populations. All proof gates passed.

| Metric | Process 1 | Process 2 | Process 3 | Median of medians |
|---|---:|---:|---:|---:|
| Complete semantic-input-to-calendar graph | 0.211799 s | 0.210766 s | 0.216390 s | **0.211799 s** |
| Semantic input generation | 0.008788 s | 0.008805 s | 0.008802 s | **0.008802 s** |
| ActivitySim checkpoint-to-result run | 31.853 s | 32.148 s | 31.170 s | **31.853 s** |

All 15 complete public replays reproduced every bit of 1,210,124 logsums and
all 81,983 final TDD labels. No modeled host transfer or CPU fallback occurred
after sealing.

## What changed from Phase 27

Phase 27 encoded 15 chooser-by-alternative fields with repeated response
dictionaries. Those dictionaries were exact and compact, but they described
observed answers rather than the rules that produced the answers.

Phase 28 carries semantic names from the shared strict IR into the resident
invocation and accepts only these declared sources:

- `daily_parking_cost`;
- SOV-toll and HOV2-toll availability;
- local, commuter, express, heavy-rail, and light-rail/ferry walk-transit
  availability; and
- the corresponding drive-transit availability fields.

The compiler emits one CUDA kernel containing the same comparisons, sums, and
public MTC transit scale factor of 100 as the preprocessor. It addresses raw skim cubes with the already
qualified origin, destination, and time coordinates. Drive-transit formulas
also read compact per-tour auto ownership. Parking cost reads a compact
double-precision rate and computes `rate * (end - start)` before casting to the
strict input dtype.

Any response-pattern source outside this registry fails compilation. The
generic reconstruction kernel uses an explicit semantic-generated marker and
cannot silently fall back to a dictionary.

## Exact floating-point parking rates

ActivitySim evaluates parking multiplication in double precision and then the
strict CUDA input pack converts the result to float32. Dividing one rounded
float32 cost by one duration does not necessarily recover a rate that
reproduces every other duration.

Phase 28 solves this with a rounding-interval qualification. For every observed
duration and float32 result, it computes the interval of double values that
would round to those exact output bits. It intersects all intervals for one
tour, selects a valid double rate, regenerates every cost, and rejects the plan
unless every output bit matches. This is a qualification mechanism; a later
raw-table producer should read the unrounded parking rate directly from land
use and tour free-parking state.

## Changed-scenario proof

The fixed public benchmark alone cannot prove that a formula generalizes. A
separate qualification creates five independent inputs with different:

- chooser identities and auto ownership;
- parking rates and start/end alternatives;
- origins, destinations, and time indices; and
- raw road and transit skim values, including positive, zero, and negative
  availability cases.

Each scenario contains 1,600 rows. An independent NumPy implementation creates
the expected 15 semantic columns; the Phase 28 compiler regenerates them on
CUDA. All 8,000 rows are bit-exact, all 15 formulas are exercised, and all five
output hashes differ. This proves that the named generators respond to changed
inputs rather than memorizing the public row patterns. It does not qualify the
entire ActivitySim model under five new policy scenarios.

## Memory and performance

Phase 28 retains 20,258,882 bytes of compact input state for 503,411,584 bytes
of removed captured row arrays, a **24.849x reduction**. This is 19.102% less
persistent input state than Phase 27. The 5,439,864 bytes of response
dictionary storage are gone. Tiny exact start/end tables and one parking-rate
vector per batch replace them; the existing per-row compact slot index is
reused.

Generating availability from raw skims performs real cube gathers, so it is
slower than looking up Phase 27's dictionaries. The full graph median is 3.147%
above Phase 27 and 5.450% above Phase 26. That is the measured cost of replacing
memorized response forms with scenario-responsive formulas. No whole-model CPU
speedup should be inferred from the 0.212-second resident result.

## Proof gates

The hash-chained aggregate requires:

- three independent public processes and 15 complete resident replays;
- all six real 315-term programs;
- zero anonymous response-pattern columns;
- all 15 semantic formulas present across the six programs;
- bit-identical logsums and exact final schedules;
- no original captured pointer in the timed runtime;
- zero modeled post-seal H2D, intermediate D2H, or CPU fallback; and
- five distinct changed-scenario outputs with all 15 formulas exercised.

The complete test suite passes 132 tests.

## Honest boundary

Phase 28 removes anonymous response dictionaries, not every cold-start use of
ActivitySim. During qualification, ActivitySim still creates dense inputs so
constant, per-tour, and exact-slot factors can be discovered and byte-checked.
The compact parking rate is also recovered from qualified parking outputs
rather than read directly from raw land-use state. The timed sealed graph needs
neither the dense arrays nor the response dictionaries.

The next production boundary is therefore a raw-table source compiler: create
constant and tour factors directly from household/person/tour/land-use tables,
read parking rates from land use plus free-parking state, and use ActivitySim's
dense rows only as an independent test oracle. Separately, the 57-entry frozen
boundary-decision map still needs a shared Sharrow/CUDA arithmetic definition.

## Reproduction

Run three fresh copies of the frozen checkpoint with:

```powershell
./.venv-phase8/Scripts/python.exe scripts/run_phase22_integrated_scheduling.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <fresh-output-copy> `
  --reference-pipeline benchmark-data/phase9-mtc-full/prototype_mtc_extended/o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline `
  --report <phase28-live.json> `
  --checkpoint <checkpoint.json> `
  --kernel-reports <empty-report-directory> `
  --resident-semantic-input-report <phase28-semantic-input.json> `
  --resident-replay-runs 5
```

Run changed scenarios and aggregate:

```powershell
./.venv-phase8/Scripts/python.exe scripts/qualify_phase28_semantic_scenarios.py `
  --output benchmark-results/phase28-changed-scenario-qualification.json

./.venv-phase8/Scripts/python.exe scripts/summarize_phase28_semantic_inputs.py `
  --input benchmark-results/phase28-semantic-input-1.json `
  --input benchmark-results/phase28-semantic-input-2.json `
  --input benchmark-results/phase28-semantic-input-3.json `
  --live benchmark-results/phase28-live-1.json `
  --live benchmark-results/phase28-live-2.json `
  --live benchmark-results/phase28-live-3.json `
  --scenarios benchmark-results/phase28-changed-scenario-qualification.json `
  --phase27 benchmark-results/phase27-generated-input-summary.json `
  --phase26 benchmark-results/phase26-resident-schedule-summary.json `
  --output benchmark-results/phase28-semantic-input-summary.json
```
