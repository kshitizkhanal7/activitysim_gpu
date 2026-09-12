# Phase 59: qualified scenario runtime; 100-second target missed

## Final outcome (September 12, 2026)

All six balanced Phase 58/59 pairs improve both measured clocks. Independent
checks preserve modeled decisions, all 115 matrices and all 24 summary-report
values. The three changed scenarios also pass on the same implementation
fingerprint. The qualification status is
`replicated_improvement_wall_target_not_met`.

| Clock | Fresh regular CPU | Matched Phase 58 | Phase 59 | CPU / Phase 59 |
|---|---:|---:|---:|---:|
| Launch-to-exit median | 314.90 s | 112.52 s | 109.84 s | 2.867x |
| Charged median | 308.40 s | 104.82 s | 102.69 s | 3.003x |

The incremental elapsed-time reduction is **2.38% (2.68 seconds)**; charged
time falls **2.03% (2.13 seconds)**. Every candidate takes 109.34-110.81 seconds
elapsed, so none reaches 100 seconds. The first control's 123.71-second elapsed
time remains in the evidence; no favorable subset replaces the six-pair series.
The historical Phase 58 median in the objective below is not substituted for
the newly matched control. Two fresh regular CPU runs follow the paired series;
that cumulative comparison is unpaired and limited to the stated configuration.

[Every model step and both full-model clocks](phase59-component-comparison.md)
are published, including regressions and unchanged CPU steps. Mandatory tour
scheduling rises from 7.15 to 10.80 seconds as the live, stronger correctness
path replaces the narrower artifact-dependent path. Trip scheduling falls
from 3.70 to 2.30 seconds and matrix writing from 6.60 to 2.50 seconds.
These component medians do not add exactly to full-run medians. Matrix writing
is a CPU storage improvement; the full-model ratio is not pure GPU hardware
attribution. The strongest equivalent-algorithm CPU controls appear below.

Evidence: `benchmark-results/phase59-formal-qualification.json`,
`phase59-complete-comparison.json`, the complete `phase58-p59formal-summary.json`
and `phase58-p59regular-summary.json`, and the three final-source scenario
summaries. The following sections preserve the development chronology,
including rejected attempts; their provisional language describes that stage,
not an unresolved current qualification.

The final full repository test run passes **413 tests** in 43.88 seconds.
Its 48 warnings are upstream Pydantic/pandas deprecations and a non-fatal
pytest cache-write permission warning. No tests fail or skip in that run.
The updated explainer retains the earlier phases and adds the new algorithms,
assumptions, rejected boundary adapters, stronger CPU controls and target miss.

## Objective and acceptance criteria

The approved goal is a larger, scenario-ready device runtime, with a target of
less than 100 seconds **from process launch to exit** on this machine. Phase 58's
qualified baseline is 111.02 seconds wall time and 103.81 seconds charged model
time. Those are different clocks; reaching one is not reaching the other.

Completion requires a live versioned data contract, complete departure retries,
removal of mandatory scheduling's captured input/answer dependency, changed
scenario comparisons, strong CPU algorithm controls, six balanced full-model
pairs, explicit correctness gates, and an updated plain-English explainer/PDF.
The final result establishes repeated improvement, but not the sub-100-second
target. Generality is limited to the supported contracts and tested scenarios.

## Findings and implementation decisions

- Profiling the original matrix step found 0.87 seconds in two groupby reductions
  versus 4.88 seconds in the writer (including 3.10 seconds closing compressed
  datasets). These are instrumented diagnostic timings, not performance evidence.
  GPU aggregation cannot plausibly eliminate all 6.5 seconds of normal matrix time.
- The new matrix verifier compares every OMX dataset's shape, dtype and byte hash,
  including zone mappings. Different compression is allowed; changed values are
  not. The Phase 58 profile run's five OMX files match the regular CPU reference.
- Complete departure retries now run within one CUDA invocation. One thread owns
  a person-tour. Failed outbound and inbound legs remain active independently.
  An outbound leg that succeeded earlier is absent from the inbound lower-bound
  calculation on later retries, exactly as in upstream ActivitySim logic version 2.
- Random uniforms are generated speculatively for at most 100 attempts per active
  trip. Only the actually consumed count advances the live ActivitySim ledger.
  This exchanges extra GPU computation/memory for removal of repeated CPU joins
  and retry coordination. It is not permission to consume unused random draws.
- Nine complete-controller tests cover three seeds, shuffled rows, nonzero initial
  random offsets and retry limits of 1, 3 and 100. Final departures and complete
  random ledgers match actual upstream execution. Eight earlier chain tests also
  pass after extracting shared input validation/packing.
- Development run `p59retry1` passes all previous full-model implementation and
  published-output gates. Scheduling takes about 1.9 seconds. Overall charged
  time is 109.43 seconds and launch-to-exit is 117.24 seconds: this is **not** a
  whole-model improvement over the qualified Phase 58 median. It is a development
  run, not a matched comparison; matrix auditing overlapped briefly with this run.
- A live mandatory scheduler is being tested. It compiles the current model spec,
  uses current tours/timetables and live random draws, and has no constructor input
  artifact. Its seven supported timetable expressions are checked structurally.
  Numerical boundary rows must be recalculated live with all feasible alternatives,
  not merely the compact period representatives. Historical inline artifact error
  counters are compatibility fields, not evidence of comparisons in this new path.
- The first complete live-mandatory run passes the external modeled-decision
  oracle with 57 live boundary rows and no captured scheduling artifact. It takes
  143.65 seconds charged / 152.28 seconds wall; mandatory scheduling alone is
  42.4 seconds while new Sharrow comparison code is generated. Cache-reuse run
  `p59live2` takes 108.04 seconds charged / 115.62 seconds wall and also passes
  all five OMX files exactly. Neither is a balanced performance qualification.
- The shared keyed device store now owns numeric person/tour/trip snapshots and
  retry working columns. Equal columns can be reused; changed values and changed
  row order invalidate leases. CPU publication remains authoritative at step
  boundaries. This is an ownership foundation, not a claim that every component
  now consumes the store or that snapshot checks are free.
- Compiled CPU controls sweep 1, 4, 12, 24 and 48 threads. For the public trip/tour
  structure with fresh seed-17 uniforms, the compact retry kernel is faster on the
  best CPU control (about 2.1 ms) than GPU (about 7.4 ms). A warp-counter experiment
  does not reverse this. Removing Python/pandas retry coordination is the primary
  explanation for the component gain; GPU hardware superiority is **not** proven
  for this algorithm. These kernel controls exclude packing, RNG and transfers.
- The mode control uses synthetic utilities at the public trip count, not captured
  public mode utilities. GPU reduction wins these tests, but its observed advantage
  against the best CPU thread count varies between runs (about 1.3x to 1.9x). This
  is not an end-to-end component ratio. Raw samples and source hashes are retained
  in the two CPU-algorithm-control JSON files.

## Development checklist (subsequently completed)

Qualify the live mandatory path; integrate/test versioned ownership where there
are actual consumers; test changed scenarios; run compiled CPU controls; evaluate
matrix work against its measured ceiling; perform balanced end-to-end comparisons;
document rejected candidates and update the explainer and PDF. If the 100-second
target is missed, report that plainly without weakening output or timing gates.

## Scenario debugging and output optimization evidence

The 10,000-household seed-991 candidate `p59scenario10k4` completes all 34
steps and matches its independently generated regular CPU control in every
modeled decision and all five OMX files. It contains 87,922 trips. Earlier
attempts exposed (and failed closed on) directory-ordinal-sensitive config
hashing and two inherited alternative-width whitelists. The replacement
supports actual widths 1 through 32, with tests of every width. It does not
pad rows or change arithmetic order. The scenario overlay exception is limited
to seed/sample settings; original specs and coefficients remain hash checked.

This diagnostic used a new cache on the hard disk because the SSD previously
had insufficient space. It took 768.44 seconds charged / 792.75 seconds wall,
dominated by first-use compilation and slow storage. It is **not timing
evidence**. No earlier outputs were deleted to make room. A later read-only
space check found ample SSD space; subsequent experiments use fresh output
directories there. Formal comparisons use identical storage/cache policy.

Two writer-only sweeps use the actual published morning-period OMX values:

| Writer | First sweep median (s) | Larger-tile sweep median (s) |
|---|---:|---:|
| Upstream | 0.9612 | 0.9436 |
| Sparse 16 | 1.0146 | - |
| Sparse 32 | 0.5303 | - |
| Sparse 64 | 0.2212 | 0.2567 |
| Sparse 128 | - | 0.1912 |
| Sparse 256 | - | 0.2615 |

Every output is logically exact; setup, aggregation and verification are
excluded from this writer-only clock. The selected default is 128, with four
CPU compression workers. This is explicitly CPU storage optimization, not GPU
hardware superiority. Full-model comparisons must still qualify it across all
115 matrices. Evidence: `phase59-matrix-writer-controls.json` and
`phase59-matrix-writer-large-controls.json` in `benchmark-results`.

The entity contract additionally tests signed zero, copied ID ownership and
explicit deletion invalidation. Live mandatory scheduling checks that every
feasible TDD's compact cache slot exists and is finite. The runtime records
speculative random-buffer, generator-state and retry-workspace bytes. Existing
legacy telemetry fields that are not instrumented must not be interpreted as
zero memory or zero cost.

An input-only capture mode supports the stronger mode-reduction CPU comparison.
It saves live utilities, random uniforms, chooser IDs and the current nest, not
expected choices. Capture runs are explicitly excluded from timing evidence.
The first capture attempt found a serialization mismatch between a test dict
and ActivitySim's `LogitNestSpec`; the new regression test covers the actual
model object. No modeled arithmetic was changed to resolve that diagnostic bug.

## Seed-17 failure: a stronger boundary reference is required

The final-source 10k rerun passes, but the first 50k seed-17 test rejects the
candidate. Mandatory tour frequency is identical. Mandatory scheduling changes
two rows; downstream tour scheduling expands this to three final tour rows
and ten modeled cells. `phase59-seed17-boundary-failure.json` retains the exact
diagnostic observations. In both initial divergences the GPU's first TDD is
the CPU reference TDD; the weak CPU boundary guard then changes it incorrectly.
Their boundary distances are about 9.15e-9 and 5.91e-8. These IDs are diagnostic
evidence only, not runtime correction rules.

The replacement rechecks complete live CPU logsum inputs as well as the final
scheduling equation. It receives the already-generated six standard normals,
current period rows, tours, settings and skims. It does not load expected
choices. The random ledger must be unchanged afterward, and patched functions
are restored even on exceptions. Fourteen focused tests cover three seeds,
three starting offsets, repeated/reordered IDs, hook restoration and failure.

An implementation detail matters: the pinned upstream `assign_variables`
preprocessor batches its six lognormal expressions into one six-column normal
draw. Six separate scalar calls are **not equivalent**, even when they end at
the same offset. An intermediate attempted scalar adapter was rejected by the
tests/integration check; production retains the upstream batched contract.
The first compilation of the complete CPU boundary logsum evaluator takes
100.08 seconds for one work segment. This is an explicitly disclosed cold
compilation cost, not eligible cached-run timing evidence.

The first complete-logsum adapter also exposed an integration bug: it derived
cache keys from representative hours. In the public school/university table,
the EV representative starts at 18, which would map to PM if treated as an
actual departure hour. The corrected adapter uses the original categorical
`out_period`/`in_period` keys. A regression test specifically covers this case.
The erroneous adapter run is rejected, not silently accepted.

With both corrections, `p59seed17fix3` passes every modeled decision and every
matrix against the same fresh CPU reference. It performs 63 complete boundary
rechecks covering 936 period rows, consumes zero additional random draws and
observes at most 9.54e-7 GPU/CPU logsum difference in those inputs. Its 105.35
seconds charged / 112.92 seconds wall are one development observation, not a
balanced speed qualification and not the under-100-second target.

Final run fingerprints now also include the pinned ActivitySim Python source
and the three static CUDA/compiled-plan assets, in addition to ChoiceForge,
the runner/verifiers and configuration-file hashes. Three changed scenarios
and the old/new timing series must use the same final implementation snapshot.

## Final-source changed scenarios and stronger CPU controls

The following three final-source candidates pass the independent decision and
OMX oracles. Their purpose is input-domain qualification, not a balanced speed
comparison. Every candidate uses the same implementation fingerprint as the
formal old/new series.

| Changed case | Households | Charged (s) | Launch-to-exit (s) | Evidence tag |
|---|---:|---:|---:|---|
| Seed 991 and smaller sample | 10,000 | 61.75 | 68.79 | p59qualified10k |
| Seed 17 | 50,000 | 103.83 | 111.04 | p59qualifiedseed17 |
| Seven departure attempts | 50,000 | 103.13 | 110.43 | p59qualifiedretry7 |

The corresponding fresh regular CPU outputs remain available independently.
The 10k baseline was generated in `p59scenario10k1`; the seed-17 baseline in
`p59finalseed17`; the seven-attempt baseline in `p59qualifiedretry7`. Earlier
failed candidate attempts do not invalidate those separately completed CPU
references and are not counted as successful candidates.

The stronger mode control now uses **actual live public-model inputs**, not
synthetic utilities: 442,682 rows across ten purpose-specific nested-logit
segments. The input-only capture contains utilities, random uniforms, IDs and
the nest definition, never expected choices. Its instrumented capture run is
excluded from performance claims. With seven balanced measurements per thread
count (1, 4, 12, 24, 48), the GPU reducer is **1.5985x** faster than the best
compiled CPU reducer. This includes allocation and resident computation, but
excludes utility expression evaluation, JIT, packing, random generation and
transfers. It is not a complete trip-mode component speedup.

The final public-structure retry control gives CPU/GPU time ratio **0.2963**:
the best compiled CPU is approximately **3.37x faster** at that same narrow
resident algorithm boundary. The synthetic mode control gives 1.7745x GPU
advantage, but the real-input 1.5985x result is the more relevant mode evidence.
Both favorable and unfavorable controls are retained with raw measurements.

For the main 50k run, retry telemetry records 69,690,400 bytes of speculative
draw buffers, 217,434,048 bytes of generator state and 20,181,928 bytes of
retry working arrays. Entity snapshots require additional storage. These are
instrumented allocations, not a claim about whole-process peak GPU memory.

## Published reports: exact values, explicit formatting contract

The independent report verifier also checks the entire `summarize/` inventory.
The public output has 24 CSV reports. Four time-of-day reports can spell a
departure key as `5` instead of `5.0`: non-mandatory tours, work tours, school
tours, and trip purpose by time of day. Only their named `depart` column may
differ in decimal spelling. Decimal arithmetic establishes exact equality with
**no tolerance**; all other cells, headers, row order and shapes must match.
Other files must be byte-identical. Missing/extra reports or changed counts
fail. Both raw file hashes and every representation-only cell difference are
published in the qualification evidence. This is not a claim that every CSV
file has identical bytes.

## Reproduction procedure and measurement boundary

Use the pinned ActivitySim checkout and reproduction patch documented in the
repository setup, the public extended MTC project, and the existing Python
environment. The tested workstation has a Threadripper PRO 5965WX (24 cores,
48 threads) and RTX A4000 (16 GB). Ordinary numerical libraries use one thread
in full-model runs; the matrix candidate explicitly uses four compression
workers. Separate algorithm controls sweep up to all 48 CPU threads.

Run serially from the repository root, choosing fresh tags/output paths:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRO --modes gpu,candidate --repetitions 6 --phase59 --live-mandatory --sparse-matrices
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPROCPU --modes regular --repetitions 2 --phase59
```

Despite the harness filename, `gpu` denotes the Phase 58 control and
`candidate` denotes Phase 59 when `--phase59` is supplied. Odd pairs run the
old version first; even pairs run the candidate first. Every process writes
a fresh directory on the same SSD and reuses the established compiler caches.
No model steps or outputs are suppressed. Model/configuration fingerprints
are checked before and after each child; do not edit them during a series.

These are substantial inherited prerequisites, including the Phase 56 native
skim artifact; this is not a clean-machine one-command installation claim.
The pinned ActivitySim commit is `16ab11180a26912987eb902daf945e268f3efc11`.
The measured environment uses Python 3.11.14, NumPy 2.4.6, Numba 0.66.0,
CuPy 14.1.1 and Sharrow 2.16.2. The arithmetic ABI is version-sensitive.
Upstream component timings are rounded to tenths of a second; extra displayed
decimals do not create additional timing precision. The live CPU boundary
adapter temporarily replaces upstream function hooks and is scoped to the
tested serial single-process model, not concurrent model execution in threads.

Charged time includes all 34 steps plus recorded runtime validation and
prewarming. Launch-to-exit includes process startup and shutdown as well.
Independent post-run oracle comparisons are outside both clocks; they are
additional qualification costs, not part of operational model execution.
The fresh regular CPU controls run after the paired series and provide an
unpaired cumulative comparison, not a randomized CPU/GPU whole-model trial.

Re-run each changed scenario against its own fresh CPU reference, using the
three retained overlays `benchmark-data/configs_phase59_seed991`,
`benchmark-data/configs_phase59_seed17` and `benchmark-data/configs_phase59_retry7`. Do not use
the seed-0 reference as an answer key for changed-seed runs. The final
qualifier consumes the balanced summary and three scenario summaries:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRO10K --modes regular,candidate --repetitions 1 --phase59 --live-mandatory --sparse-matrices --households 10000 --scenario-overlay benchmark-data/configs_phase59_seed991
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPROSEED --modes regular,candidate --repetitions 1 --phase59 --live-mandatory --sparse-matrices --scenario-overlay benchmark-data/configs_phase59_seed17
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRORETRY --modes regular,candidate --repetitions 1 --phase59 --live-mandatory --sparse-matrices --scenario-overlay benchmark-data/configs_phase59_retry7
```

Substitute the newly generated summary paths in the qualification command
when reproducing, rather than overwriting the published original evidence.

```powershell
.venv-phase8/Scripts/python.exe scripts/qualify_phase59.py --comparison benchmark-results/phase58-p59formal-summary.json --scenarios benchmark-results/phase58-p59qualified10k-summary.json benchmark-results/phase58-p59qualifiedseed17-summary.json benchmark-results/phase58-p59qualifiedretry7-summary.json --output benchmark-results/phase59-formal-qualification.json
.venv-phase8/Scripts/python.exe -m pytest -q
```

Retain actual outputs until qualification: the report verifier reads them
again instead of trusting a previously asserted success flag. The qualifier
hashes its code, input summaries and individual implementation reports.

To regenerate the independent real-mode CPU control, first capture a new
instrumented full run, then benchmark the input-only arrays separately:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPROCAPTURE --modes candidate --repetitions 1 --phase59 --live-mandatory --sparse-matrices --capture-mode-inputs benchmark-data/REPRO-mode-inputs
.venv-phase8/Scripts/python.exe scripts/benchmark_phase59_live_modes.py --inputs benchmark-data/REPRO-mode-inputs --output benchmark-results/phase59-REPRO-live-mode-control.json
```

Do not include that capture run in timing qualification. Public datasets,
intermediate NPZ captures and large retained model outputs are not committed;
their generation paths and hashes are evidence, not claims that a checkout
alone contains every prerequisite.

## What remains beyond this phase

The practical next target is at least another 10 seconds of whole-process
reduction, not another impressive isolated-kernel ratio. Prioritize complete
mandatory scheduling (10.80 s), trip destination (8.90 s), non-mandatory tour
frequency (7.10 s) and CDAP (6.90 s), with live data preparation inside the
measurement boundary. Those are measured component ceilings, not additive
promised savings. Versioned snapshots only help when actual consumers replace
their old preparation paths; merely publishing extra copies is not residency.

An adaptive CPU/GPU retry path is worth evaluating because the matched compact
CPU algorithm wins, but its few milliseconds cannot by themselves save ten
seconds. Removing generator-state preparation, redundant tables and handoffs
is the larger hypothesis. Cold-start compilation also deserves separate
measurement. Broader geographic/configuration qualification and a production
upstream backend remain open; the three tested scenarios are not substitutes.
