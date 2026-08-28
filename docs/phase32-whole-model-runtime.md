# Phase 32: whole-model GPU runtime proof

## Outcome

Phase 32 makes the first replicated, measurable dent in the complete public
34-step ActivitySim runtime from the native mandatory-scheduling work. It
combines three previously qualified GPU consumers in one fresh model process:

1. native strict-ABI mandatory tour mode logsums and GPU scheduling;
2. generated-CUDA trip-destination mode logsums; and
3. generated-CUDA trip mode-choice utilities.

The immutable CUDA skim cubes stay resident across those consumers and are
released once, after `trip_mode_choice`. Later model steps keep their existing
ActivitySim implementations.

Three fresh, interleaved 50,000-household pairs compared Phase 32 with the
already GPU-accelerated Phase 17 runtime. Phase 32 won every pair:

| Pair | Phase 17 current | Phase 32 | Saved | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 200.0 s | 188.4 s | 11.6 s | 5.800% | 1.0616x |
| 2 | 196.2 s | 190.9 s | 5.3 s | 2.701% | 1.0278x |
| 3 | 197.0 s | 188.8 s | 8.2 s | 4.162% | 1.0434x |
| Median | **197.0 s** | **188.8 s** | **8.2 s** | **4.162%** | **1.0434x** |

Mandatory scheduling itself fell from 25.6, 24.3, and 24.7 seconds to 14.9,
15.0, and 15.2 seconds. The middle baseline and candidate values are 24.7 and
15.0 seconds: 9.7 seconds saved, 39.271% lower, or 1.647x faster.

This is an incremental whole-model result over Phase 17, not a comparison with
an unoptimized toy loop. Phase 17 already accelerates trip destination and trip
mode choice. The Phase 17 report separately establishes its replicated result
against the pinned Sharrow-required ActivitySim control.

## Direct comparison with regular ActivitySim

The follow-up experiment answers the more useful deployment question directly:
how much faster is the finished GPU runtime than a regular, pinned ActivitySim
run? The control uses ActivitySim's normal optimized Sharrow expression backend,
all 34 configured model steps, and no ChoiceForge overlay or GPU hooks. This is
not a naive Python substitute.

Three new interleaved, fresh-process pairs used the same 50,000 households and
all 1,454 zones. The GPU runtime won every pair:

| Pair | Regular ActivitySim | GPU runtime | Saved | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 218.2 s | 192.3 s | 25.9 s | 11.87% | 1.135x |
| 2 | 214.3 s | 192.9 s | 21.4 s | 9.99% | 1.111x |
| 3 | 215.2 s | 200.3 s | 14.9 s | 6.92% | 1.074x |
| Median | **215.2 s** | **192.9 s** | **22.3 s** | **10.36%** | **1.116x** |

Every post-run output comparison passed its declared gates. Across all three
pairs there were zero changed modeled decision cells. Floating-point diagnostic
text is not byte-identical: 18,180 destination-logsum cells differ, with a
worst absolute difference of 0.0000100000000032 against a 0.0001 gate; the
worst mode-choice-logsum difference is 0.00000289586 against a 0.00001 gate.
Those same bounded differences repeat in all three pairs. This direct experiment
is now the primary whole-model performance claim. The Phase 17 comparison above
remains useful because it isolates the additional benefit of native GPU
mandatory scheduling.

### Every ActivitySim component

The table reports the median ActivitySim timer for each named step. Negative
savings mean the candidate was slower in that step. Untargeted rows mostly show
normal run-to-run noise and must not be described as GPU speedups. The two large
direct gains are mandatory scheduling and trip destination; trip mode choice is
currently neutral at this scale.

| Component | Regular ActivitySim | GPU runtime | Saved | Reduction | Speedup | GPU role |
|---|---:|---:|---:|---:|---:|---|
| initialize_landuse | 16.8 | 18.2 | -1.4 | -8.33% | 0.923x | not directly GPU-targeted |
| initialize_households | 5.7 | 5.3 | 0.4 | 7.02% | 1.075x | not directly GPU-targeted |
| compute_accessibility | 2.0 | 2.0 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| school_location | 9.6 | 9.8 | -0.2 | -2.08% | 0.980x | not directly GPU-targeted |
| workplace_location | 14.6 | 14.4 | 0.2 | 1.37% | 1.014x | not directly GPU-targeted |
| auto_ownership_simulate | 0.9 | 0.9 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| free_parking | 1.1 | 1.1 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| cdap_simulate | 6.8 | 6.9 | -0.1 | -1.47% | 0.986x | not directly GPU-targeted |
| mandatory_tour_frequency | 1.5 | 1.6 | -0.1 | -6.67% | 0.938x | not directly GPU-targeted |
| mandatory_tour_scheduling | 24.6 | 15.3 | 9.3 | 37.80% | **1.608x** | Phase 32 native GPU |
| joint_tour_frequency | 1.3 | 1.3 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| joint_tour_composition | 0.7 | 0.8 | -0.1 | -14.29% | 0.875x | not directly GPU-targeted |
| joint_tour_participation | 2.4 | 2.5 | -0.1 | -4.17% | 0.960x | not directly GPU-targeted |
| joint_tour_destination | 3.4 | 3.8 | -0.4 | -11.76% | 0.895x | not directly GPU-targeted |
| joint_tour_scheduling | 1.4 | 1.4 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| non_mandatory_tour_frequency | 3.4 | 3.6 | -0.2 | -5.88% | 0.944x | not directly GPU-targeted |
| non_mandatory_tour_destination | 13.9 | 14.3 | -0.4 | -2.88% | 0.972x | not directly GPU-targeted |
| non_mandatory_tour_scheduling | 10.1 | 10.0 | 0.1 | 0.99% | 1.010x | not directly GPU-targeted |
| tour_mode_choice_simulate | 5.5 | 5.6 | -0.1 | -1.82% | 0.982x | not directly GPU-targeted |
| atwork_subtour_frequency | 1.1 | 1.1 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| atwork_subtour_destination | 4.2 | 4.0 | 0.2 | 4.76% | 1.050x | not directly GPU-targeted |
| atwork_subtour_scheduling | 1.7 | 1.7 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| atwork_subtour_mode_choice | 1.1 | 1.2 | -0.1 | -9.09% | 0.917x | not directly GPU-targeted |
| stop_frequency | 3.7 | 3.9 | -0.2 | -5.41% | 0.949x | not directly GPU-targeted |
| trip_purpose | 1.1 | 1.2 | -0.1 | -9.09% | 0.917x | not directly GPU-targeted |
| trip_destination | 42.6 | 27.6 | 15.0 | 35.21% | **1.543x** | Phase 17 generated GPU retained |
| trip_purpose_and_destination | 0.6 | 0.7 | -0.1 | -16.67% | 0.857x | not directly GPU-targeted |
| trip_scheduling | 7.4 | 7.5 | -0.1 | -1.35% | 0.987x | not directly GPU-targeted |
| trip_mode_choice | 10.2 | 10.2 | 0.0 | 0.00% | 1.000x | Phase 17 generated GPU retained |
| write_data_dictionary | 1.7 | 1.8 | -0.1 | -5.88% | 0.944x | not directly GPU-targeted |
| track_skim_usage | 0.4 | 0.5 | -0.1 | -25.00% | 0.800x | not directly GPU-targeted |
| write_trip_matrices | 7.0 | 7.0 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |
| write_tables | 1.9 | 2.0 | -0.1 | -5.26% | 0.950x | not directly GPU-targeted |
| summarize | 4.8 | 4.8 | 0.0 | 0.00% | 1.000x | not directly GPU-targeted |

The per-step timers are rounded to tenths by ActivitySim, so small differences
should not be treated as precise causal estimates. The 22.3-second whole-model
median saving is not the sum of independently rounded median rows.

## Correctness result

Every candidate run passed all live gates:

- all six native mode-logsum programs ran on CUDA over 1,210,124 rows;
- all 81,983 mandatory tours had exact TDD, start, and end values;
- no CUDA fallback occurred;
- no bulk modeled logsum vector returned to the host;
- the 57 exercised near-boundary choices used the qualified device map;
- zero boundary logsum bytes returned to the host;
- all 34 model steps completed; and
- 6,198,614,528 resident CUDA bytes were released once after trip mode choice.

After each matched pair, `verify_phase15_outputs.py` compared the complete
published model outputs. All three pairs had zero changed decision cells and
zero changed diagnostic logsum values. The seven substantive `final_*.csv`
tables therefore represent the same modeled population, tours, trips, choices,
and numerical diagnostics. All non-trip files are also byte-identical.

The proof does not rely on an in-process GPU self-check. The reference output
is produced by a separate fresh baseline process, and final publication is
compared after both processes exit.

## Architecture and lifecycle

The full model still needs ActivitySim's Sharrow skim dataset for components
that have not been ported. Loading the separate 6.20 GB Phase 31 native store
at the same time would duplicate the road-network payload in host memory.
Phase 32 therefore uses a transitional hybrid:

```text
ActivitySim loads the public skim dataset once
                     |
                     v
native mandatory ABI uploads/reuses immutable CUDA cubes
                     |
                     v
generated trip destination reuses the same CUDA cube cache
                     |
                     v
generated trip mode choice reuses the same CUDA cube cache
                     |
                     v
release 6.199 GB once; finish remaining CPU/ActivitySim steps
```

This is deliberate. The Phase 31 native store remains the faster independent
choice for a resumed mandatory-only process. The full model should not keep two
host representations until every remaining skim consumer can use the native
store.

The scheduling-specific function hooks are removed immediately after mandatory
scheduling. Only immutable skim data survives. This prevents later joint,
non-mandatory, and at-work scheduling components from accidentally entering a
backend whose contracts they have not yet qualified.

## Failures found and corrected

The final result required three integration corrections. They are part of the
evidence because a whole-model runtime must survive consumers outside the
optimized component.

### Scheduling metadata leaked into trip destination

The first full run asked ordinary trip-destination rows for scheduling-only
`start`, `end`, and time-period metadata. The strict candidate failed closed
with `KeyError: 'start'` and returned to Sharrow. Metadata capture is now
conditional on a real device-logsum sink. A regression test proves that a
destination frame without scheduling columns is not read.

### Releasing the skim cache too early

Releasing 6+ GB immediately after mandatory scheduling made trip destination
rebuild the same cubes. One diagnostic full run completed exactly but spent
275.9 seconds in trip destination. The release boundary now follows the last
qualified GPU skim consumer, `trip_mode_choice`. Normal trip destination then
returns to 26.7--27.2 seconds, and every candidate releases exactly
6,198,614,528 bytes once.

### Qualification work contaminated production timing

An early full run also enabled the strict CPU/CUDA oracle for every destination
batch. An interrupted stack proved that time was being spent in
`evaluate_strict_cpu`, not in the candidate CUDA kernel. Phase 32 disables that
shadow only in the timed full-model boundary. The strict oracle remains in the
qualification suite, while complete final-output comparison supplies the
independent correctness gate for each performance pair. This separates
measurement from qualification without weakening replication.

## Reproduction

On the pinned RTX A4000 environment:

```powershell
.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 `
  -Households 50000 `
  -RunTag p32cpu1 `
  -Baseline activitysim
```

The runner refuses to overwrite any output. It executes fresh processes in
`Phase17-current 1 / Phase32 1 / Phase17-current 2 / Phase32 2 / ...` order,
records external wall and ActivitySim model timings, requires every candidate
proof gate, and runs complete-output verification after each pair.

Primary evidence:

- [`phase32-p32cpu1-summary.json`](../benchmark-results/phase32-p32cpu1-summary.json)
- [`phase32-p32cpu1-gpu-1.json`](../benchmark-results/phase32-p32cpu1-gpu-1.json)
- [`phase32-p32cpu1-gpu-2.json`](../benchmark-results/phase32-p32cpu1-gpu-2.json)
- [`phase32-p32cpu1-gpu-3.json`](../benchmark-results/phase32-p32cpu1-gpu-3.json)
- the three `phase32-p32cpu1-exact-*.json` output comparisons

Incremental Phase 17 comparison evidence:

- [`phase32-p32proof1-summary.json`](../benchmark-results/phase32-p32proof1-summary.json)
- [`phase32-p32proof1-gpu-1.json`](../benchmark-results/phase32-p32proof1-gpu-1.json)
- [`phase32-p32proof1-gpu-2.json`](../benchmark-results/phase32-p32proof1-gpu-2.json)
- [`phase32-p32proof1-gpu-3.json`](../benchmark-results/phase32-p32proof1-gpu-3.json)
- the three `phase32-p32proof1-exact-*.json` output comparisons

## Claim boundary and next major opportunity

Phase 32 proves a repeatable **10.36% median reduction (1.116x speedup)** in all
34 model steps on this public 50,000-household benchmark relative to regular
pinned ActivitySim, with complete output replication. It also proves the native
mandatory scheduler adds a 4.162% median reduction beyond the already-GPU
Phase 17 runtime. It does not prove the same percentage on another GPU,
population size, network, or model configuration. It does not make ActivitySim
CPU-free.

The next major opportunity is no longer another scheduling micro-phase. It is
a native model-wide skim and expression service for the largest remaining
consumers: non-mandatory destination/scheduling, tour mode choice, and trip
scheduling. The performance gate should remain whole-model time. The memory
gate should allow one authoritative host/network representation, not a native
store beside a second Sharrow store. Changed-scenario qualification and an
upstream shared arithmetic contract are still required before the small
boundary map can become universal rather than benchmark-qualified.
