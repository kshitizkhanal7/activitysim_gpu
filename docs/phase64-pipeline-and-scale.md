# Phase 64: whole-model pipeline, GPU attribution and scale

Status: output, memory and measurement qualification complete. The 50,000-household
hybrid takes **74.95 seconds** versus **208.93** for ordinary CPU: **2.787x**.
The paired previous-version median is 76.17 seconds, a **1.59%** reduction with
all six pairs improving both clocks. The under-70 and 65-second targets are
**not met**. Complete 100,000- and corrected 250,000-household rechecks pass.
All **663 tests pass**, with 91 dependency warnings. The 174-page explainer's
cover, opening pages, appendix transition and all new pages are visually checked.

## Approved scope and acceptance criteria

Phase 63 is the published baseline: 73.94 seconds fresh elapsed, 2.743x versus
ordinary CPU, and 2.528x in the separate matched-preparation worker comparison.
This phase targets a substantial complete-model reduction rather than another
isolated kernel headline. Under 70 seconds remains the minimum fresh-median
goal; 60-65 seconds is the ambition, not a promised outcome.

1. Profile the selected Phase 63 path, with special attention to mandatory
   scheduling, trip destination and trip mode choice. Distinguish arithmetic,
   preparation, transfers, allocation, validation and output costs.
2. Implement candidates supported by that profile. Keep live input binding,
   independent CPU arithmetic and borderline-choice checks. Reuse instructions
   and immutable inputs, never saved model answers or mutable scenario state.
3. Run controlled CPU/GPU calculations with identical live/captured inputs,
   precision and boundaries. Separate transfer-inclusive and device-resident
   timings. Existing full-system ratios are not pure GPU-hardware attribution.
4. Attempt complete 100,000- then 250,000-household models with independently
   generated CPU references. Preflight host/GPU memory and disk; retain failures.
   No silent precision change, disabled checks or synthetic-model substitution.
5. Freeze candidates before balanced qualification. Remeasure contemporaneous
   CPU/previous-hybrid controls, changed scenarios and sequential memory behavior.
   Publish exact output contracts, sample counts and all clock boundaries.
6. Run tests, update the beginner Markdown/PDF, visually inspect changed pages,
   and publish the source/evidence. Document target misses and scale limits.

## Experiment discipline

No simultaneous model runs, benchmarks, tests, profiling or PDF rendering.
Diagnostic profiles are not performance evidence. Every experiment has a new
tag; existing outputs are not overwritten or pruned. Source/configuration
digests must stay fixed during measured runs. Preserve Phase 63 evidence and
its same-machine clean reconstruction. Independent hardware replication is
still outstanding and is not implied by further local work.

## Initial workstation preflight

The RTX A4000 reports 16,376 MiB total and about 809 MiB already occupied by
the desktop at the initial check. C: has about 109.9 GB free before Phase 64
experiments. These are starting observations, not guarantees of later capacity.
Two unrelated Python GIS MCP servers are idle background services; neither is
a model benchmark. Do not stop unrelated user processes.

The initial diagnostic is `p64profile1`, using the complete selected Phase 63
options, all live proof gates and unique repository-local output paths.

## First diagnostic and candidates

The complete profiled model passes its output gates at 99.38 seconds; that is
instrumented time and is excluded from speed comparisons. The ten profiled
steps include about 283,000 Python `eval` calls (1.10 seconds self time and
5.43 seconds cumulative, including work performed by expressions). Mandatory
scheduling also spends significant time in the live independent CPU boundary
path. That correctness path is not removed. Trip-destination normalized GPU
utility work and preparation both remain visible; profiling overhead means
these costs cannot simply be subtracted from the published uninstrumented run.

The first candidate reuses bounded Python expression code objects, executing
them against current globals/locals every time. Fifteen initial regression
tests pass, including changed coefficients, signed zero, side effects, errors,
scope restoration, cross-thread fallback and bounded capacity. Both engines
will receive equivalent preparation options in matched comparisons. This
candidate is CPU preparation optimization, not a GPU arithmetic speed claim.

An explicit CPU mode-reducer ablation is also implemented for testing. It
retains GPU utility generation and random draws, changes only the nested
probability/choice reducer, and charges device-to-host and return transfers.
Its results cannot be labeled a complete all-CPU model. Strong standalone
controls will also start on host inputs to avoid making unnecessary transfers
look like an inherent CPU disadvantage.

`p64expr1` is a two-pair development comparison, not the final qualification.
The first pair passes at 76.29 seconds baseline versus 75.38 candidate. This
small observation is not yet a substantial or replicated speed claim.

A retained 50,000-household output occupies approximately 294 MB, so initial
disk capacity is sufficient to attempt larger models without deleting old
evidence. The Phase 63 sampled peak RSS was about 9.95 GB for its first large
hybrid scenario; this is not a linear prediction for 250,000 households.
The new scale harness keeps one-second host-memory observations and stops only
its own child tree if available RAM stays below 2 GiB for three samples or
free disk falls below 8 GiB. Monitored single pairs establish feasibility and
output correctness, not final repeated performance superiority.

## Verified raw-input representation

The numeric-input artifact stores the unmodified full public household, person
and land-use tables as uncompressed Feather, before household sampling or model
decisions. Preparation re-reads the artifact and compares every numeric column's
bytes with the upstream CSV parser, then rehashes the source to detect changes
during conversion. The initial preparation takes 4.47 seconds, outside warm
model timings and reported in `phase64-inputprep1-input-preparation.json`.

Every use hashes both the original CSV and the entire artifact. The identity
includes parser-source bytes, requested numeric dtypes and NumPy/pandas/Arrow
versions. Missing/unsupported artifacts fall back to the original reader;
corrupt prepared artifacts fail closed. Each return is a private frame. No
pickle or saved model decisions are loaded. This does not eliminate validation
cost or certify maliciously modified artifacts whose manifests were also edited.

The complete `p64inputs1` development model passes all output gates with three
artifact hits and no misses. Household initialization is 3.4 seconds; elapsed
79.85 seconds is not a paired improvement claim (different session/startup).
Approximately 297,716 expressions reuse code while 1,915 are compiled; all
expressions still execute with live inputs. The expression-only two-pair series
passes with 76.29/76.23-second old runs versus 75.38/75.41-second new runs. These
remain development evidence, not a claim that the phase's 60-65-second ambition
has been achieved.

The initial expanded regression subset passes 39 tests, including CPU reducer
agreement away from guarded boundaries, thread restoration, numeric artifact
corruption, same-size/timestamp input changes, dtype identity and private copies.
`p64s100k1` completes a new ordinary CPU reference and the complete hybrid at
100,000 households with resource guards and unchanged output tolerances.
The measured process clocks are 333.1108 seconds CPU and 105.2649 seconds
hybrid, a single-pair ratio of 3.1645x. Decision columns, all logical matrices
and report values pass their existing contracts. This is monitored feasibility
evidence, not a repeated performance claim. Maximum sampled individual child
RSS is 11,468,464,128 bytes; minimum host available RAM is 31,396,294,656 bytes.
These are sampled host measurements, not GPU peaks or aggregate process USS.

The first 250,000-household pair (`p64s250k1`) completes computation but FAILS
the external exact-output audit. Ordinary CPU elapsed time is 672.8207 seconds.
There is no qualified hybrid speed ratio for this failed pair. The failure is
one differing household cell, `hh_work_auto_savings_ratio`, traced to person
6230726 selecting workplace zone 752 on the hybrid versus 764 on CPU. Household
2480353 consequently has a ratio of 3.4674168 versus 4. The audit is not relaxed.
The individual identifiers are diagnostic locators in the synthetic public
population, never lookup keys for replacement answers.

Both outputs and the failure receipt are retained. Maximum sampled individual
child RSS is 20,251,910,144 bytes; minimum host available RAM is 26,023,841,792
bytes. No resource guard stopped execution. This establishes capacity to run
the workload, NOT correctness at that scale. A targeted live diagnostic now
captures utilities, probabilities and the existing random draw, stopping after
workplace location. It does not inject CPU answers or count as a full timing run.
The first diagnostic had a capture bug (device-only draws have no host array);
that failed attempt is retained and the revised capture downloads only the
requested diagnostic row. Independent CPU and GPU captures then confirm the
same draw (0.7311568634551859), but three of 30 padded utilities differ by one
float32 unit (about 0.000000954). The CPU cumulative probability at alternative
20 is about 0.73115679, while the GPU-derived value is about 0.73115691. The
draw lies between them, producing different choices despite the small error.

The existing guard DOES flag this row. Its limitation is that it recomputes CPU
probabilities from GPU-produced utilities, so an earlier utility/logsum rounding
difference survives the check. A new opt-in `location_boundary` path recomputes
the full upstream CPU logsum and final utility for guarded destination owners.
It retains only the already-consumed six normal values per owner until the
immediate final-choice consumer, borrows the guarded subset without advancing
the random ledger, and keeps the original padded segment width. A targeted
live diagnostic is running; full corrected-model qualification is still required.

The first live correction (`locfix1`) chooses the correct zone but still differs
in two utility cells because sampled-destination probabilities also enter the
sample-correction term. Fifteen of 28 captured sampled probabilities differ by
small rounding amounts, even though sampled destination IDs agree. The revised
path recomputes the sampling probabilities as well, using current raw chooser,
specification and full destination inputs for only the guarded owners. It does
not resample alternatives or draw new random numbers.

`locfix2` matches all 30 padded utilities AND all 30 probabilities bit-for-bit
with the independent CPU capture and selects internal zone 763 (published zone
764). The old GPU selected internal 751 (published 752). Both use the same
recorded draw. The corrected full 250,000-household candidate (`p64s250k2`) is
has passed the complete external output audit against the retained independent
CPU reference: decision columns, all matrices and summary reports match. Its
launch-to-exit time is 501.2547 seconds (489.9426 charged seconds), including
first-use compilation of the live CPU safeguard. Peak sampled individual-child
RSS is 20,630,368,256 bytes; no resource guard fired and source hashes remained
unchanged. This is correctness/scale feasibility, not warmed performance.
This iteration
does not rerun CPU, so it cannot supply a new contemporaneous CPU speed ratio.
The expanded pre-run regression subset passed 47 tests with two deprecation
warnings. Five additional boundary tests subsequently pass, including live
sample-probability correction and rejection of an advanced random ledger.
The complete test suite and final repeated campaign are still outstanding.

The larger benchmark also exposes an audit limitation worth making explicit:
the integrated run's selected live proof gates passed, but the separate full
CSV audit rejected the household value. Those are complementary checks. We
must retain the external all-output audit, not treat an internal successful
run or exact tour choices as proof that every modeled field is unchanged.
Existing Phase 63 claims remain restricted to their tested populations and
scenarios; they are not retroactively extended to 250,000 households.

## Predeclared final campaign

`run_phase64_campaign.py` defines 42 sequential complete model runs: six
reversed-order Phase 63/64 fresh pairs (12), two ordinary CPU controls, four
matched-preparation fresh processes per engine (8), and two ten-scenario
hybrid sequences (20). The matched fresh order reverses between trials.
The changed-scenario sequences use A,B,C,D,A,C,B,A,D,A, including changed
seeds, household count and coefficients, each with its own existing independent
CPU reference. This campaign does not remeasure persistent CPU sequences and
therefore makes no new persistent CPU/hybrid speed-ratio claim.

The memory criteria remain explicit: no active GPU arrays before each invocation
or after the final reset; fifth-to-tenth invocation growth at most 512 MiB USS
and 128 MiB active GPU allocation. Growth beyond those limits is reported as
unresolved, not silently excused by passing outputs. A finite sequence cannot
guarantee indefinite leak-free operation. The old CSV snapshot cache may now
retain zero bytes when verified Feather inputs bypass it; private copies,
artifact hashes and actual artifact use are checked separately for both engines.

CPU mode-reducer control telemetry explicitly reports its backend and charged
download/upload bytes. It no longer claims dense utility downloads were avoided
when the CPU control actually downloads those utilities. This is a reporting
correction for the new ablation, not a change to the GPU arithmetic contract.

## Same-input mode-reduction control

`p64capture1` completes the selected guarded 50,000-household model and passes
all output audits. Its 78.56-second clock includes input capture and is not
performance evidence. It captures 19 real tour/trip batches, 603,161 rows.
`phase64-mode-controls1.json` checks both reducers against upstream CPU nested
probabilities, sweeps 1, 4, 12, 24 and 48 CPU threads, and takes nine alternating
measurements per configuration. The strongest measured CPU setting is 48.

Against that setting, GPU-resident nested reduction is 1.4225x faster. Including
host-to-device inputs and device-to-host results gives a CPU/GPU ratio of 0.9262:
the GPU is slightly slower, not faster. This result supports residency for this
primitive and rejects a transfer-inclusive GPU-superiority claim. Both controls
exclude utility generation, RNG and live boundary adjudication. Exact primitive
choices exclude the union of their 1e-9 boundary guards; production adjudicates
those rows live. Logsum comparisons use 1e-12 relative and absolute bounds.
No primitive ratio is presented as a complete-model speedup.
"Resident" describes the large utility/draw/result arrays: the actual timed
API still uploads six nest coefficients and downloads a scalar invalid-row
count, synchronizing for its fail-closed domain check. Those costs are included.
It is not a no-CPU/no-transfer claim and not an isolated CUDA-event kernel time.

The full-model reducer ablation (`p64ab1`) passes all four output audits. In
reversed-order pairs, GPU-reducer models take 75.53 and 74.19 seconds; replacing
only tour/trip reduction with CPU takes 73.96 and 73.54 seconds. Medians are
74.8565 GPU versus 73.7517 CPU, a CPU/GPU elapsed ratio of 0.9852. This does not
demonstrate a whole-model win from this GPU reducer. Two pairs are insufficient
to assign every approximately one-second system difference to a primitive that
itself takes only milliseconds. The full-model control keeps GPU utilities and
RNG, charges transfers, and is not ordinary all-CPU ActivitySim. No production
backend switch is justified by this small experiment alone.

## Final-code scale repeats

`p64s100k2` passes all output checks using the final guarded candidate and the
retained independent 100,000-household CPU reference. Elapsed time is 109.63
seconds; peak sampled individual-process RSS is 11,370,442,752 bytes. No guard
fires. This is a single monitored candidate-only recheck, not a new paired CPU
ratio. `p64s250k3` also passes, completing 250,000 households in 299.09 elapsed
seconds (290.63 charged). Its peak sampled individual-process RSS is
20,055,449,600 bytes. The live safeguard handles 28 owners in 3.3064 seconds.
The first-use 501.25-second run and this warmed repeat are different setup
conditions, not a paired optimization ratio. The latter checks against the
retained independent CPU reference, not a freshly timed CPU run. Neither size
is a full 2.875-million-household calibrated-model qualification.

## Final campaign audit and retained measurement failure

All 42 models in `p64f1` finish and pass their output audits, but the final
qualifier rejects the ordinary-CPU thread contract: both CPU controls omitted
an explicit `OMP_WAIT_POLICY=PASSIVE`, while the hybrid and matched workers
used it. This is not silently interpreted as PASSIVE. The launcher now sets
the policy for every child and accepts an explicit new CPU-control tag when
resuming. The shared timed harness and production source remain unchanged.

The original two CPU controls are retained but excluded from qualification.
Replacement controls use `p64cpu2`; final selection therefore contains 42 models
from 44 executed campaign models. All twelve old/new hybrid runs, eight
matched-preparation runs and twenty changed-scenario runs remain valid. The
audit fix passes 21 focused tests, including rejection of a missing wait policy.
Both memory sequences have zero active default-pool GPU bytes after final reset;
fifth-to-tenth USS growth is 174.45 and 188.35 MiB, below the 512 MiB criterion.
Both replacement controls pass and the final qualifier reports
`outputs_and_memory_qualified`. The ordinary-CPU median is 208.93 seconds versus
74.95 for the hybrid (2.787x). Matched preparation has separate worker-clock
medians of 206.83 seconds CPU and 77.44 hybrid (2.671x), four processes per engine.
The final full suite passes 663 tests with 91 warnings. The consolidated result
is `phase64-complete-comparison.json`; all 34 component rows and scope caveats
are in `phase64-component-comparison.md`.

The prepared-workspace check also verifies all 98 pinned packages and five
public input files. This is a present-workspace integrity check, not a new
installation or a second-machine replication. Final delivery separately checks
staged Git bytes under both newline policies, all 34 documented component rows,
current production hashes and the actual visually reviewed PDF.
