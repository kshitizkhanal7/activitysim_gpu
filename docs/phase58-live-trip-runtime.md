# Phase 58: live trip runtime and honest CPU attribution

## Qualified outcome

**Phase 58 meets the charged sub-105-second target in all six candidates.**
Six balanced fresh-process pairs improve both charged total and process wall
in every pair. All implementation gates, exact published-decision checks, and
paired scheduling chooser/failure counts pass. The machine-readable status is
`replicated_improvement_target_met`.
All **264 repository tests pass**, including the six qualification-checker tests.

| Measurement | Regular CPU (2 runs) | Previous GPU (6 runs) | Phase 58 (6 runs) |
|---|---:|---:|---:|
| Charged complete-model median | 300.00 s | 110.97 s | 103.81 s |
| Launch-to-exit median | 306.36 s | 118.27 s | 111.02 s |
| Trip destination median | 66.65 s | 9.75 s | 9.00 s |
| Trip scheduling median | 6.75 s | 6.20 s | 3.65 s |
| Trip mode choice median | 10.55 s | 9.85 s | 6.05 s |

The incremental gain is **1.069x / 6.45% less charged time**, saving 7.16 seconds
at the medians. The three targeted component medians sum to 25.80 versus 18.70
seconds, a 27.52% reduction; their 7.10-second difference closely accounts for
the whole-model change. This sum is descriptive, not a separately timed median.

The cumulative new-control comparison is **2.890x** on charged totals and
**2.759x** on process wall. The latter compares 306.36 with 111.02 seconds and is
the closest measure of elapsed application wait. These are comparisons against
the regular CPU configuration described below, not the strongest conceivable
CPU rewrite or evidence that GPU hardware alone caused every saved second.

| Pair | Order | Previous GPU | Phase 58 | Charged seconds saved |
|---|---|---:|---:|---:|
| 1 | Old, new | 111.232 | 103.235 | 7.998 |
| 2 | New, old | 110.233 | 104.034 | 6.199 |
| 3 | Old, new | 110.615 | 103.815 | 6.801 |
| 4 | New, old | 111.418 | 104.015 | 7.404 |
| 5 | Old, new | 111.715 | 103.523 | 8.192 |
| 6 | New, old | 110.715 | 103.815 | 6.900 |

All six candidates are between 103.235 and 104.034 seconds. Displayed extra
decimals include validation accounting; model-step timing resolution remains
0.1 second. None of these runs establishes a launch-to-exit time below 105 seconds.

Evidence: [qualification](../benchmark-results/phase58-formal-qualification.json),
[all matched runs](../benchmark-results/phase58-formal-summary.json),
[clean CPU controls](../benchmark-results/phase58-controls2-summary.json), and
[normal-RNG rejection](../benchmark-results/phase58-normal-rng-rejection.json).

## Scope

Phase 58 targets complete trip components on the public MTC full-geography,
50,000-household workload. It adds a shared live keyed-uniform service, a CUDA
departure-chain executor, and device-resident nested mode probabilities and
selection. All 34 model steps, fresh matrices, summaries, input validation and
prewarm remain in the measured application. The complete application remains
a benchmark-specific CPU/GPU hybrid, not a general GPU-only ActivitySim release.

The development candidate passed every implementation and output gate at
103.632 seconds charged model total / 111.075 seconds process wall. This is
development evidence; the final matched qualification is the authority for
the repeatability claim.

## Why these changes, rather than another small kernel ratio?

The six clean control runs distinguish three questions:

1. Regular CPU ActivitySim: original pinned application, no ChoiceForge overlay.
2. Compact CPU control: the accelerated stack, but the identical Phase 57
   feasibility reduction runs as compiled Numba code with 48 threads.
3. Existing GPU stack: identical compact representation, using the CUDA reduction.

Regular CPU model-step totals were 300.2 and 299.8 seconds. The compact CPU
charged totals were 111.635 and 111.233 seconds; GPU totals were 111.833 and
111.035 seconds. The latter medians differ by less than 0.001 seconds, far below
the 0.1-second component timing resolution. There is no demonstrated material
whole-model GPU advantage for compaction alone. The prior 2.634x *kernel*
advantage remains a separate, valid boundary-specific measurement.

The regular CPU result is a new measurement, not the older 205.4-second result.
No cross-date ratio against that older number is used as fresh evidence. The
new regular controls precede the six-pair incremental experiment, rather than
being repeated inside every pair. Host/cache effects are not fully controlled.

Regular CPU means the pinned single-process ActivitySim/Sharrow configuration,
with numerical libraries limited to one thread, not a claim about the fastest
possible multiprocess CPU deployment. The 48-thread compact control applies
only to the identified feasibility kernel; it does not make every model step
48-threaded. Phase 58 has not independently compared every new trip kernel with
a hand-optimized parallel CPU rewrite. Its incremental evidence concerns the
complete application, including data-layout and orchestration improvements.

## Architecture and ownership

The diagnostic-only profile measured trip destination at 13.80 seconds, trip
scheduling at 8.86 seconds, and trip mode choice at 13.09 seconds. Mode choice
spent 7.34 seconds cumulatively inside nested-logit simulation, including 2.70
seconds of utility evaluation and 2.29 seconds of choice handling. Scheduling
spent 6.30 seconds in its per-leg driver. These nested timings overlap and must
not be added as independent savings. They motivated eliminating repeated CPU
coordination and probability tables, rather than optimizing the already-small
inner arithmetic alone. The profiled run is excluded from formal timing medians.
The [saved hotspot summary](../benchmark-results/phase58-profile-hotspots.json)
contains the underlying cumulative and self-time entries.

| Boundary | New work on GPU | Deliberately retained CPU work |
|---|---|---|
| Trip destination | Live MT19937 uniform generation through the shared service | Ordinary normal draws, existing compact final Sharrow evaluation, host interfaces |
| Trip scheduling | Within-tour bounds, outbound/inbound dependency, clipped probabilities and choices | Live key/index preparation, failed-cohort retry loop, final table publication |
| Trip mode choice | Strict FP32 utility output directly into FP64 nested probabilities, logsum and selection | Chooser annotation, ordinary normal draws, final labels and diagnostics; live near-boundary adjudication if needed |

An epoch is one active model step. The random service rejects stale step names;
ActivitySim still owns the keyed row seeds and offsets. A draw advances the
same row's offset once, regardless of frame order. Scratch memory is reused,
but modeled answers are not replayed. Context hooks are restored on exit.

For scheduling, each GPU thread executes one short person-tour chain. Independent
person-tours run in parallel. Outbound departures constrain later outbound trips;
their maximum then constrains the inbound leg, processed in reverse. At-work
and tour-origin trips retain their prescribed departures without consuming RNG.
The first schedulable trip is normalized using the same global trip-number
rule as upstream. Failure probabilities, failed-choice bounds and final-iteration
`choose_most_initial` corrections follow the pinned upstream algorithm.

ActivitySim retains its 100-iteration failed-cohort loop. A scheduling iteration
has one final choice-vector publication, rather than a CPU publication between
each dependent trip-number choice. This is **iteration-level residency**, not
an entire immutable trip table resident from destination through final output.

In the first formal pair, the old adapter made 548 scheduling calls; the chain
executor made 100 iteration calls. Both processed 210,110 random-consuming
chooser rows and recorded 66,899 failed choices before final corrections.
These aggregate counts are checked pair-by-pair in qualification, in addition
to exact final decisions. Full RNG-ledger equality is directly tested in the
changed-input unit tests, not independently captured for every full-model row.

For mode choice, canonical MTC alternative and nest order are validated. Utilities
stay in their already-qualified FP32 representation; exponentials, conditional
probabilities, ordered products, logsums and inverse-CDF selection use FP64 with
FMA contraction disabled. Unsupported nests and nonfinite probability domains
fail closed. The 21-mode utility matrix is no longer downloaded to make choices.

## Arithmetic and replication boundaries

The mode selector checks every cumulative-probability boundary with a 1e-9
engineering guard. Guarded rows are adjudicated live with actual upstream
probability and choice functions using the **same computed utility and already
consumed draw**. No saved choices or population identity list is consulted.
The guard is not a universal floating-point error theorem. Independent exact
published-decision verification remains mandatory for qualified full runs.

Changed-seed tests compare keyed uniforms, ordinary normals, subsequent draws,
row permutations and subsets, plus the complete RNG ledger. Chain tests compare
actual upstream scheduling on changed tour lengths/windows, reordered trips,
at-work subtours, impossible windows and final-iteration corrections. Mode tests
cover changed utilities and deliberately constructed CDF edges; a separate
adapter test exercises live boundary adjudication without consuming a second draw.

Unsupported chain configuration is rejected: relative/duration modes, a nonempty
preprocessor, logic other than version 2, alternative failure policies, incomplete
legs, non-monotonic within-leg trip IDs, invalid probabilities or invalid times.
Estimation and household tracing are outside the new adapter's supported scope.

The full pipeline still uses inherited mandatory-scheduling reference artifacts
and sparse adjudication contracts. New trip kernels do not read saved decisions,
but their changed-input tests do not establish unrestricted whole-model scenario
or seed support. Logsums may differ within published tolerances; exact travel
decisions do not imply byte-identical floating-point diagnostics or improved
behavioral prediction accuracy.

## Rejected and failed experiments

- `controls1`: the compact CPU arm failed because ActivitySim reset Numba's
  pool limit after initialization. The corrected CPU arm uses `--fast` only to
  prevent that CLI override; all ordinary compute remains masked to one thread,
  with 48 threads enabled only around the compact reduction and restored afterward.
  Other numerical-library thread limits remain one. Failed-run timing is excluded.
- `profile1`: cProfile traced the three trip steps to locate CPU RNG, joins and
  probability materialization. Its timings are diagnostics, not speed evidence.
- `smoke1`: live RNG and mode selection passed before the chain executor was
  connected. It is not the final candidate used for qualification.
- `chain1`: a fresh process caught an import-order bug hidden by unit-test module
  loading. An obsolete class-replacement hook was removed. This failed execution
  is excluded; `chain2` passed full verification afterward.
- General GPU normal RNG reuse was rejected: the reproducible audit compared
  3,168 values across changed seeds/steps/offsets and found 12 tiny numerical
  differences even though ledger offsets matched. Ordinary normals remain on CPU.
  Earlier Phase 54 fixed-workload exactness must not be read as a universal normal
  arithmetic guarantee. The rejection does not retroactively invalidate that
  fixed-workload result.

The zero-variance-normal algebraic shortcut is tested, but the complete public
development run did not use it. It receives no credit for measured savings.

## Measurement and reproduction

Use the pinned `.venv-phase8` environment and public MTC project/reference inputs
described in previous phase reports, including the verified Phase 56 skim image.
These prerequisites are substantial; this is not a clean-machine one-command
installation claim. No concurrent tests or heavy benchmarks may overlap timing.

```powershell
.venv-phase8\Scripts\python.exe -m pytest tests -q --disable-warnings
.venv-phase8\Scripts\python.exe scripts/audit_phase58_normal_rng.py --output benchmark-results/phase58-normal-rng-audit-repeat.json
.venv-phase8\Scripts\python.exe scripts/run_phase58_comparison.py --tag repeat-controls --modes cpu,gpu,regular --repetitions 2
.venv-phase8\Scripts\python.exe scripts/run_phase58_comparison.py --tag repeat-pairs --modes gpu,candidate --repetitions 6
.venv-phase8\Scripts\python.exe scripts/qualify_phase58.py --controls benchmark-results/phase58-repeat-controls-summary.json --matched benchmark-results/phase58-repeat-pairs-summary.json --output benchmark-results/phase58-repeat-qualification.json
```

Use new tags: existing outputs are protected from overwrite. The harness checks
all 34 timings, implementation gates and independent final outputs. Source SHA256
fingerprints must remain identical throughout matched measurements. It reverses
order on even repetitions: three old-first and three new-first pairs. These are
fresh processes reusing existing disk/compiler caches, not cold-machine tests.

Charged total means the sum of model-step timings plus validation and recorded
prewarm. Launch-to-exit process wall is separately measured and includes Python
startup and other orchestration. Component timings are rounded to tenths of a
second; extra decimals from validation arithmetic do not increase their accuracy.
Medians of component rows need not sum to the median complete-run total.

Independent final-output comparison runs after the timed child process: it is
an experimental audit, not simulated model work. The inherited implementation
report also contains checks inside the child, which process wall therefore pays.
Source fingerprints cover Python implementation and harness files; they are
not a substitute for the inherited configuration and input-artifact checks.

The chain executor records complete service time, kernel time, transfer totals,
row counts and failure counts. Some inherited per-call telemetry fields
(`host_index_seconds`, `random_ledger_seconds`, `peak_workspace_bytes`, and
`first_trip_calls`) are not instrumented for this new batched path and retain
their default zero values. They must not be interpreted as zero allocation,
zero indexing cost or absent first-trip normalization. The shared random
service's scratch memory and live chain allocations are real GPU memory costs.

The qualifier requires at least four balanced pairs with lower charged total
and process wall in every pair, exact outputs, frozen code and all implementation
gates. The sub-105-second candidate median is a separate gate. Repetition here
measures local performance variability, not six new populations or a universal
statistical guarantee.

## Remaining ambition

The next architectural boundary is a versioned person/tour/trip data store with
explicit mutation/invalidation rules and final publication only at required
interfaces. Current trip destination still crosses CPU interfaces, and failed
scheduling cohorts still return to the CPU once per iteration. Replacing those
boundaries must preserve the random ledger and dependency order demonstrated here.

Separately, remove inherited mandatory-scheduling benchmark artifacts and qualify
changed populations/seeds end-to-end. Neither a small kernel ratio nor this phase's
fixed-workload success establishes that broader portability claim.

## All 34 component medians

Seconds; regular CPU uses two preceding control runs, previous/latest GPU use
six balanced matched runs each. Ratios below one are retained, not hidden.
Small changes in untouched steps are timing variability, not new kernel claims.
Initialization gains include persistent input caching, not merely GPU arithmetic.

| Component | Regular CPU | Previous GPU | Phase 58 | CPU / Phase 58 |
|---|---:|---:|---:|---:|
| initialize_landuse | 14.35 | 0.60 | 0.60 | 23.92x |
| initialize_households | 4.70 | 4.40 | 4.40 | 1.07x |
| compute_accessibility | 1.65 | 1.60 | 1.60 | 1.03x |
| school_location | 15.15 | 3.90 | 3.95 | 3.84x |
| workplace_location | 30.45 | 2.00 | 2.00 | 15.22x |
| auto_ownership_simulate | 0.80 | 1.40 | 1.40 | 0.57x |
| free_parking | 0.90 | 1.00 | 1.00 | 0.90x |
| cdap_simulate | 6.90 | 6.90 | 6.90 | 1.00x |
| mandatory_tour_frequency | 1.40 | 1.50 | 1.50 | 0.93x |
| mandatory_tour_scheduling | 38.95 | 7.10 | 7.05 | 5.52x |
| joint_tour_frequency | 1.20 | 1.20 | 1.20 | 1.00x |
| joint_tour_composition | 0.60 | 0.70 | 0.70 | 0.86x |
| joint_tour_participation | 2.10 | 2.15 | 2.10 | 1.00x |
| joint_tour_destination | 3.10 | 2.00 | 2.00 | 1.55x |
| joint_tour_scheduling | 1.40 | 1.60 | 1.60 | 0.87x |
| non_mandatory_tour_frequency | 7.25 | 7.30 | 7.25 | 1.00x |
| non_mandatory_tour_destination | 30.50 | 2.00 | 2.00 | 15.25x |
| non_mandatory_tour_scheduling | 18.30 | 6.80 | 6.80 | 2.69x |
| tour_mode_choice_simulate | 5.60 | 5.40 | 5.40 | 1.04x |
| atwork_subtour_frequency | 1.00 | 1.00 | 1.00 | 1.00x |
| atwork_subtour_destination | 7.55 | 1.40 | 1.40 | 5.39x |
| atwork_subtour_scheduling | 1.60 | 1.70 | 1.70 | 0.94x |
| atwork_subtour_mode_choice | 1.00 | 1.10 | 1.10 | 0.91x |
| stop_frequency | 3.35 | 3.50 | 3.40 | 0.99x |
| trip_purpose | 1.00 | 1.10 | 1.10 | 0.91x |
| trip_destination | 66.65 | 9.75 | 9.00 | 7.41x |
| trip_purpose_and_destination | 0.55 | 0.60 | 0.60 | 0.92x |
| trip_scheduling | 6.75 | 6.20 | 3.65 | 1.85x |
| trip_mode_choice | 10.55 | 9.85 | 6.05 | 1.74x |
| write_data_dictionary | 1.50 | 1.55 | 1.50 | 1.00x |
| track_skim_usage | 0.40 | 0.40 | 0.40 | 1.00x |
| write_trip_matrices | 6.50 | 6.50 | 6.50 | 1.00x |
| write_tables | 1.80 | 1.80 | 1.80 | 1.00x |
| summarize | 4.50 | 4.50 | 4.50 | 1.00x |

Hardware: NVIDIA RTX A4000, 16,376 MiB reported VRAM, driver 571.59; 48 logical
CPU workers available on an AMD Ryzen Threadripper PRO 5965WX (24 physical
cores), Windows, pinned Python 3.11.14 environment. NumPy 2.4.6, Numba 0.66.0,
CuPy CUDA12x 14.1.1, ActivitySim 1000.dev1+g16ab11180, Sharrow 2.16.2. See the
saved per-run thread environment and source fingerprints for execution settings.
The source fingerprints are byte-sensitive, including line endings; cross-OS
Git checkout conversion can change them. Repeated pairs on a new checkout must
remain internally frozen rather than assuming cross-platform byte identity.
