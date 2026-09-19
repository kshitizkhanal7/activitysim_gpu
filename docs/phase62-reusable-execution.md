# Phase 62: reusable compiled execution - qualified

Final status: **replicated improvement; fresh-process time target not met**.
All 43 formal model runs completed on September 14, 2026. Delivery checks
and the explainer update were completed September 19, 2026. The selected
production source is unchanged from the timed series (412 fingerprinted files).

## Approved scope and acceptance criteria

Starting point: Phase 61's qualified 79.2211-second full-model median,
201.1664-second regular CPU/48-thread control, exact output contracts, and
545 passing tests. Target a fresh-process full-model median below 70 seconds,
with 65 seconds as a stretch; these are goals, not forecasts.

1. Measure and reduce repeated calculation-plan construction, including
   household activity coordination and the independent live CPU scheduling
   reference. Preserve its arithmetic and do not reuse saved answers.
2. Reduce trip-destination contract construction, packing and transfers.
3. Make shared-column publication demand-driven and connect additional live
   consumers only where measured costs justify it.
4. Implement safe repeated-scenario execution: reuse immutable compiled
   programs and unchanged skims, while resetting all mutable model state,
   identities, timetables and random ledgers. Give CPU controls equivalent
   reuse opportunities. Include initial setup in batch totals.

Keep single-run and repeated-scenario clocks separate. Qualification requires
six balanced full-model pairs, fresh strong CPU controls, all current output
and changed-scenario gates, explicit state-isolation tests, and equivalent
compiled CPU controls for new GPU calculations. Run and verify changes,
document misses, update the beginner Markdown/PDF, and publish tested code.
Do not change production source or model configuration during a timed series.

## Initial diagnostics

`p62profile1` profiles the final selected Phase 61 CPU-timetable/passive-worker
configuration. It is diagnostic only and excluded from performance claims.

The first preparation candidate (`p62dev1`) passed the full output audits at
76.3243 seconds process wall and 69.7020 seconds charged. This is development
evidence, not a replicated speed claim. Shared entity storage fell from
42,055,332 bytes to 1,726,514 bytes while preserving the two directly consumed
columns and all 19 segment consumers. No additional GPU arithmetic was added;
the prior equivalent CPU primitive controls remain relevant.

The first batch experiment exposed a real contract mismatch: the old memory
gate required more than 5 GB to be freed at the last GPU consumer, whereas a
bounded batch pool intentionally retained about 2 GB. The failed run is kept.
Fresh-run behavior and its original gate remain unchanged. The optional
content-cached batch mode instead
reports bytes freed plus bytes owned by its bounded, content-addressed pool,
and checks the retention limit. This is a changed lifecycle contract, not an
excuse to ignore failed checks.

The content-addressed pool hashes the actual shape, dtype and contents before
reuse; an in-place changed array cannot hit by an old pointer. Its 2 GB budget
is smaller than the working set: preliminary counts show four hits and 145
misses per scenario, essentially cache thrashing. Retaining skims will remain
an option only if measured costs support it; program-only reuse is tested
separately. Never call setup savings a faster GPU arithmetic kernel.

## Selected architecture and isolation boundary

The selected batch worker does **not** retain GPU skim arrays. It reuses
Sharrow's content-hashed generated program modules, two explicitly source-keyed
CUDA kernel dictionaries, and pure compiled CPU timetable/normal programs.
It discards and rebuilds the ActivitySim and ChoiceForge application module
graphs between scenarios. Each CLI run creates a new workflow State, with new
tables, timetable instances, entity IDs and RNG ledgers. Compiled functions take
current live arrays as arguments; no utility result, choice or random answer is
used as a cross-scenario cache entry.

The worker also retains private copies of numeric **raw input** tables, before
sampling, recoding or modeling. Every scenario receives deep copies, and dtype
requests are part of the key. Non-numeric/unsupported dtype requests fall back
to original parsing. The numeric snapshot budget is 1 GiB; this benchmark uses
668,561,212 bytes. These are input survey records, not a previous model's
households or persons. Both CPU and hybrid batches receive this optimization.
All public input files are SHA-256 checked at scenario boundaries. Generated
program files are also checked for changes. The selected GPU memory-release
gate is unchanged because no GPU skim pool is retained.

This is an exclusive, sequential worker for generated benchmark manifests,
not a general-purpose in-process server or concurrent hot-reload API. Library
internals may retain old application objects, but they are not installed in
the next application's module graph. A-B-A verifies the tested restart path,
not an unlimited-scenario memory bound or correctness on arbitrary networks.
The immutable-input assumption is enforced for this benchmark by input hashes;
detected changes at scenario boundaries fail closed. Concurrent file editing
during execution is outside the supported contract; this is not a file-locking
or adversarial tamper-detection mechanism. A different network requires its
own qualification. Programs may contain immutable model constants, but their
source/IR identities must match the new request.

## Replication design

Fresh-process qualification uses six Phase 61/62 pairs with order reversed
on alternate repetitions, then two newly run CPU controls each at one and
48 Numba threads, and three changed-scenario checks (10k/seed 991, 50k/seed 17,
and retry limit 7). All 34 steps, modeled decisions, matrices, report values,
existing diagnostic bounds and live boundary adjudication remain mandatory.

The batch experiment runs A-B-A for four strategies: fresh hybrid processes,
persistent hybrid process, fresh CPU processes, persistent CPU process.
It repeats the complete four-strategy sequence in reverse order. A is the
default 50k sample and seed; B uses seed 17. All 24 outputs are checked against
independently produced CPU references. The return to A must pass its original
reference again. Initial parsing, compilation, hashes, copies, resets and
process overhead stay in the total. Output auditing is outside the clock for
every strategy. Batch totals and single-run medians must not be conflated.

## Reproduction and evidence boundaries

The qualification entry point is:

```powershell
./scripts/run_phase62_qualification.ps1 -Tag p62formal
```

This is the command for the retained series, not an instruction to overwrite
its outputs. A new execution must use an unused tag; the runners fail if
scenario output directories or result files already exist. The aggregate
qualification/report filenames are fixed, so an independent reproduction
should use a separate checkout/workspace and retain its own report bundle.
Do not run a second series concurrently on the same machine.

Prerequisites are the pinned ActivitySim environment, the public full-MTC
data directory, calibrated configurations, sealed generated kernels and
independently produced reference outputs used by the earlier phases. The
public network contains 1,454 zones; these complete-model experiments use
50,000 sampled households, not the full 2.875-million-household population.
The smaller changed-scenario check uses 10,000 households. A fresh clone alone
does not contain the large input files or retained model-output directories.
See the earlier setup and public-data reports before running this command.

`run_phase58_comparison.py --phase62` distinguishes `gpu` (the final Phase 61
control) from `candidate` (Phase 62). Both remain hybrids. Their labels do not
mean GPU-only versus CPU-only. `regular` runs the original CPU ActivitySim CLI.
The CPU capacity controls use one or 48 Numba threads, with `--fast` required
for the 48-thread setting so the CLI does not reset it. Other numerical
libraries use one thread. Selected hybrid timetable work uses 24 CPU threads;
normal generation can use 48 and matrix compression retains four workers.
The selected waiting policy is PASSIVE. Capacity is not a claim that every
operation uses that many threads.

The formal source fingerprint includes the production modules, pinned
upstream source, execution harness, output verifiers and both batch programs.
Source and configuration identities are checked around the child processes.
The batch additionally records full hashes of the five public input files
and verifies them before every scenario and after execution. Generated
Sharrow source identities are checked across the persistent sequence.
Paths and hashes make the local experiment auditable; external replication
still requires locating the same public inputs and recreating references.

The first source/data seal and manifest preparation establish the experiment
outside the batch clock. Each strategy's per-scenario data validation is
inside it, as are the actual application's initialization, compilation,
parsing or private copying, execution and shutdown. Audits of the resulting
outputs are outside all strategy clocks. The separately reported fresh-model
clock does not add the batch-specific per-scenario input hashing. This is
another reason not to compare a batch average directly with a fresh-model
median as if their measurement boundaries were identical.

All measurements are warm-cache application experiments on the existing
workstation, not cold-install compilation tests. CPU controls receive
equivalent batch reuse, but they are not a claim to be the fastest imaginable
rewrite of ActivitySim on CPU. No new GPU arithmetic primitive was introduced
in Phase 62; the prior strong primitive controls are retained with source
hash checks. They include CPU wins and are not relabeled as new experiments.

## Final fresh-process results

| Configuration | Process wall median (s) | Charged median (s) | Runs |
|---|---:|---:|---:|
| Regular CPU, one Numba thread | 299.83 | 293.35 | 2 |
| Regular CPU, 48 Numba threads | 204.84 | 198.70 | 2 |
| Phase 61, same-series control | 80.63 | 73.83 | 6 |
| Phase 62 | 76.81 | 70.37 | 6 |

The wall-time ratio against CPU/48 is 2.666660x (62.50% less elapsed time).
The incremental Phase 61/62 ratio is 1.049627x, a 4.728% reduction or 3.8120
seconds. All six pairs improve on both clocks. New elapsed times span
76.2528-78.1447 seconds. The under-70-second target misses by 6.8140 seconds;
the under-65 stretch also fails. Charged time, a development minimum, or a
later warm scenario is not substituted for the specified fresh-process clock.

The matched Phase 61 baseline is 80.63 seconds here, not its earlier 79.22
median from another series. Use same-series controls for the incremental
claim, rather than selecting the historical comparison that looks best.
The [complete 34-step table](phase62-component-comparison.md) retains slower
components as well as faster ones. Notable reductions are CDAP 4.85 to 3.55,
mandatory scheduling 7.50 to 6.90, stop frequency 2.95 to 2.50 and trip
destination 7.90 to 7.55 seconds. At-work subtour frequency rises 0.60 to
0.80. Component medians are descriptive, rounded, and not an independently
randomized causal attribution of each code change.

Every formal output audit passes: checked decisions, all 115 logical matrices,
and values across all 24 reports. Four reports permit exact-equivalent
departure-key spelling (`5` versus `5.0`). Existing diagnostic-score bounds
remain; this is not blanket intermediate bit identity. All three changed
scenarios pass (10k/seed991, 50k/seed17 and 50k/retry7).

## Final repeated-scenario results

Two repetitions of each full A-B-A strategy, with the four-strategy order
reversed, give the following three-scenario totals:

| Engine/strategy | Repetition 1 (s) | Repetition 2 (s) | Median (s) |
|---|---:|---:|---:|
| Hybrid, three fresh processes | 234.84 | 236.45 | 235.65 |
| Hybrid, persistent worker | 212.08 | 212.61 | 212.34 |
| CPU/48, three fresh processes | 593.17 | 595.66 | 594.42 |
| CPU/48, persistent worker | 544.28 | 547.21 | 545.74 |

The hybrid batch saves 23.3017 seconds (9.888%); CPU saves 48.6762 seconds
(8.189%). Persistent CPU / persistent hybrid is 2.570086x. All 24 outputs
pass, including each return to A after B. The warmed hybrid scenarios are
about 65-67 seconds inside these batches, but neither those individual
scenario times nor the 70.78-second three-run average meet the fresh-process
target by definition. Two repetitions are limited evidence, not tight
cross-machine confidence bounds.

Both candidate workers retain 668,561,212 raw-input bytes, use six table-cache
hits across B and A2, and deliver independent copies. Subsequent candidate
scenarios reuse 23 CUDA programs and 19 CPU dispatchers. The selected skim
pool has zero retained bytes. All setup and input-hashing costs remain in
the batch clock as defined above. See the [batch report](phase62-batch-comparison.md).

## Verification, deliverables and limits

The pre-qualification suite passed 565 tests. A fresh delivery run on
September 19 also passed **565 tests, 92 warnings in 44.75 seconds**. Warnings
are not hidden by claiming a warning-free suite. Tests cover IR identity,
coefficient/tree mutation isolation, ordered source traversal, failed or
replaced cache directories, exception restoration, live-column mutation,
changed skim contents/LRU bounds, private input copies and batch-evidence
rejection for missing or inconsistent controls.

The measured environment is Python 3.11.14, NumPy 2.4.6, Numba 0.66.0,
CuPy 14.1.1 and Sharrow 2.16.2 on a 24-core/48-thread AMD Threadripper PRO
5965WX with NVIDIA RTX A4000 (16 GB). Upstream ActivitySim is pinned to
`16ab11180a26912987eb902daf945e268f3efc11` with the previously documented
integration hooks. This remains a prepared Windows workstation experiment.

Machine-readable evidence:

- `benchmark-results/phase62-formal-qualification.json`: six pairs, three
  scenarios, source/configuration identities and target status.
- `benchmark-results/phase62-complete-comparison.json`: fresh CPU controls,
  all 34 steps, unchanged primitive CPU/GPU controls and evidence hashes.
- `benchmark-results/phase62-batch-comparison.json`: eight strategy results,
  24 audited outputs, order checks and input/source hashes.
- `benchmark-results/phase62-delivery-verification.json`: final source,
  configuration, input and evidence hashes; published claim checks; explicit
  visual PDF review, which is separate from text extraction.

The beginner explainer adds sections 329-337 and updates its opening and PDF
cover. It preserves historical results rather than rewriting past claims.
The selected runtime delivers preparation reuse and a qualified sequential
worker; it does not deliver a general Sharrow backend, more shared consumers,
full GPU residency, arbitrary-configuration service isolation, bounded memory
over unlimited batches, a cold-install benchmark or full-population execution.
GPU skim retention was tested and rejected, not silently omitted from scope.

Next substantial work should measure a supported reusable live CPU reference
plan and remaining trip-destination preparation, then qualify longer workers
with changed coefficients, samples and networks plus memory telemetry.
Independent reproduction is needed before broad performance or replication
claims. No weakening of the live reference or reuse of modeled answers is
an acceptable route to the missed target.
