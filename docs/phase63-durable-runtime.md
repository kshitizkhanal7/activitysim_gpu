# Phase 63: durable scenario execution and remaining preparation costs

Status: completed qualification. All 94 measured models pass; the fresh target
is NOT met. Documentation below preserves the development failures as history.

## Approved objective

Starting from Phase 62 (76.8140-second fresh-process median; 212.3440-second
three-scenario hybrid batch), reduce complete-service costs without weakening
the live independent CPU reference or output contracts. Fresh-run target:
median below 70 seconds; stretch below 65 seconds. These are targets, not
promises, and batch scenario times cannot substitute for the fresh-run clock.

The phase combines:

1. Profile Phase 62's remaining Sharrow setup and trip-destination preparation.
   Reuse executable plans only when their instructions, coefficients, bindings
   and arithmetic contracts remain valid for current live input data.
2. Implement measured CPU/GPU preparation improvements with mutation and
   restoration tests; never cache modeled answers or bypass the live reference.
3. Qualify at least ten sequential scenarios with changed seeds, coefficients
   and sample sizes, including returns to earlier configurations. Compare every
   output against independently generated CPU references; measure host and
   device memory and give CPU equivalent reuse opportunities.
4. Package pinned environment, public-input retrieval, reference generation
   and one-command qualification. State remaining preparation or platform
   limitations honestly; a local rebuild is not an independent second-machine
   replication.
5. Freeze selected production source for balanced fresh-process comparisons
   and persistent/fresh CPU controls. Document failures and target misses,
   update beginner Markdown/PDF, run tests and publish verified code.

## Experiment discipline

Use unique output tags and preserve prior results. No simultaneous model
benchmarks, test suites, profiling or rendering. Diagnostic profiles are
excluded from timing claims. Establish a new frozen series after any change
to timed production code. Reuse programs and immutable raw input snapshots,
not mutable state, RNG ledgers, timetables, utilities or decisions. Preserve
all checked decisions, matrices, report values and existing diagnostic bounds.

## Initial diagnostic

`phase58-p63profile1-summary.json` profiles the committed Phase 62 selected
runtime. It is instrumentation evidence only, not performance qualification.

The profile passes all output gates. It identifies repeated Sharrow
`init_sub_funcs` work even when the generated Python module already exists:
mandatory scheduling spends 1.03 profiled seconds in that reconstruction,
while CDAP spends 1.98 seconds in total second-stage flow initialization.
These are nested instrumented costs, not additive promised savings.
Trip destination spends 1.26 seconds constructing its purpose contracts,
with repeated configuration parsing and binding traversal; normalized GPU
evaluation also remains substantial.

The first candidate adds a JSON-only expression-plan cache and private parsed
specification-file copies. Upstream Sharrow still builds the live tree and
computes its flow identity. Plan keys include definitions, compiler options,
Sharrow implementation bytes and extra-function code; cache entries must also
match the generated source bytes. Invalid entries rebuild upstream. No Flow,
DataTree, input array or modeled output is stored in this cache. Coefficient
CSV reads check actual contents on every request and return private copies.

The initial cache-fill run (`p63dev1`) passes all output audits at 80.6455
seconds wall / 73.3524 charged. It records zero expression-plan hits and
28 writes, plus 303 parsed-file hits. This is explicitly a cache-fill
development observation, not a speed claim. Warm reuse is tested separately;
the fill cost must be published and included in cold-start reporting.

Primary references consulted for integration and reproduction:

- [Sharrow DataTree implementation and flow-library interface](https://activitysim.github.io/sharrow/_modules/sharrow/relationships.html).
- [ActivitySim CLI and example creation](https://activitysim.github.io/activitysim/develop/users-guide/ways_to_run.html).
- [Canonical public MTC repository and full-data release](https://github.com/ActivitySim/activitysim-prototype-mtc).

Current online examples do not silently replace the pinned benchmark inputs.
Reproduction must validate the actual retained data/configuration fingerprints.

## Development observations (not formal qualification)

Warm plans/files (`p63dev2`) pass the complete model at 77.5051 seconds wall,
with 28 plan hits and no misses or rewrites. Optional RSS-only in-step tracing
(`p63dev3`) passes at 73.2166 seconds. This tracing option avoids expensive USS
page walks only with chunking and chunk training disabled. It is a diagnostic
tradeoff, not GPU arithmetic acceleration. CPU controls receive the same option.
USS is still measured explicitly at scenario boundaries; an unavailable in-step
USS value must never be described as zero actual memory.

An isolated CDAP coefficient overlay changes the full-time-worker mandatory
activity intercept from 1.378734579 to 1.5. `p63coeffdev1` independently generates
the CPU reference (216.0713 seconds) and verifies the candidate (74.6944 seconds)
against those changed outputs. These development clocks are not a balanced
performance comparison. Source changed afterwards to strengthen private copies.

The first ten-scenario durability experiment is `durable-dev1`, following
A,B,C,D,A,C,B,A,D,A: A is 50,000 households/default seed; B changes the seed to
17; C uses 10,000 households/seed 991; D changes the coefficient above. Early
boundary telemetry exposes approximately 0.9 GB of additional **active** CuPy
allocation per large scenario even after application-module reset. This is a
real retention defect, not merely unused pool reservation. The experiment is
diagnostic evidence, not a durable-runtime qualification. It must be fixed and
the complete sequence rerun before delivery.

The new sequence qualification contract requires two reversed-order trials of
four strategies (fresh/persistent, CPU/hybrid), all ten independent output audits
per strategy, identical preparation options, source/input/configuration seals,
and explicit boundary USS and GPU-pool measurements. Before formal measurement,
the retention limits are fixed at 512 MiB host USS and 128 MiB active device
growth between the fifth and tenth scenarios (both A). These finite tolerances
do not prove an infinite service cannot leak, and do not hide first-run memory.

## Replication preparation

The old `activitysim-current-choiceforge.patch` fails a reverse-application check
against the current pinned checkout. The corrected, separately versioned
`integration/activitysim-phase63.patch` passes that check against upstream
commit `16ab11180a26912987eb902daf945e268f3efc11` and its two modified files.
Historical patches and outputs are preserved.

`prepare_phase63_replication.py seal` records 158 exact configuration/license
files, five public-data digests, and 98 installed package pins. Its `check`
operation passed locally. `materialize` authenticates the public archive and
each selected member, refuses conflicting existing files, rejects path escapes,
and does not retrieve model answers. Archive extraction/idempotence/corruption
tests pass on synthetic fixtures; a fresh public download and clean-environment
rebuild still require execution before they can be claimed. The current pinned
environment is Windows x64/Python 3.11.14 with the tested NVIDIA sm86 path.

## Retention defect diagnosis and first fix

The completed `durable-dev1` experiment passes all ten output audits, but keeps
7,995,681,792 active CuPy bytes after its final application reset. Its 572.7267
seconds of worker process time therefore do **not** qualify a durable service.
Fifth-to-tenth scenario USS also grows by 685,678,592 bytes, above the predeclared
512 MiB retention tolerance. Correct outputs and a speed result do not excuse
unbounded retained resources.

`retention2` is an explicitly instrumented two-model diagnostic. Collector
inspection finds retained `Phase46DestinationService` objects, including the
module-wide destination service and the RNG service reachable through the trip
scheduling service. These own large random-state and calculation workspaces.
Deleting module names alone cannot guarantee destruction when compiled/library
objects still hold references into the old graph.

The worker now explicitly clears those invocation-service owners, input-bound
Sharrow Flows, and named device/input caches before discarding application
modules. Source-keyed compiled programs remain eligible for reuse. The isolated
`retentionfix1` diagnostic passes both complete-model output audits and measures
zero active CuPy array-pool bytes after each reset. This is not zero physical
GPU usage: the CUDA context, programs and unused pool reservation still exist.
The longer changed-scenario rerun is `durable-fix1` and must finish before this
fix is considered durable under the selected workload. That development rerun
has now completed: all ten scenarios pass their output audits, with 567.6909
seconds of worker-process time. Every scenario starts with zero active CuPy
pool bytes, and the final reset also leaves zero. Fifth-to-tenth USS growth is
185,577,472 bytes, within the predeclared 512 MiB threshold; active device growth
is zero. Final unused GPU pool reservation is still 2,244,674,560 bytes. These
are finite development checks, not the formal balanced qualification.

The full test suite passes 595 tests (92 existing/dependency warnings) after the
first teardown fix. RSS is additionally sampled every 0.1 seconds during each
scenario. This reports an observed high water, not a guaranteed instantaneous
peak; USS and CuPy active/reserved pool values are boundary measurements.

Subsequent safeguards restore hooks even when the last diagnostic sample fails,
and prevent CPU-only telemetry from initializing a CUDA context. A two-model
regular-CPU development batch passes both output audits using the same
plans/files/RSS options. Its 359.916-second process total is functional evidence,
not a formal speed comparison (environment preparation overlapped that test).

The final clean-replica suite passes 601 tests with one intentional historical
checkpoint skip; see [the executed reconstruction record](phase63-replication.md).
The pinned-version/newline fixes and fresh public download are now executed,
not merely proposed. Production code is frozen during the `p63formal` campaign:
four independently computed CPU references, raw-image preparation, six balanced
Phase 62/63 fresh pairs, two CPU48 controls, and 80 changed-scenario models with
equivalent CPU reuse. Formal results remain pending until all gates finish.

All four fresh CPU references in the reconstructed environment have now finished.
Their process times are A 1,880.8361 seconds, B 201.2611, C 111.4813 and D
205.1089. A includes first-use Sharrow compilation in the new directory; these
reference-generation clocks are preparation evidence, not the formal fresh
performance comparison. The changed coefficient alters 6,373 person-activity
choices relative to A while keeping the sampled person IDs the same. The new
raw-image build and all measured comparisons use these newly generated answers.

## Fresh-pair result (now joined by completed long-sequence qualification)

The complete six-pair `p63q-fresh` series passes all output audits. Fresh-process
medians are 82.1902 seconds for the contemporaneous Phase 62 control and 73.9384
for Phase 63. Charged medians are 75.2534 and 67.1509 seconds. Every pair improves
both clocks. The fresh median is still above the predeclared 70-second target
and 65-second stretch. These controls were measured in the reconstructed
environment; do not substitute the historical 76.8140-second Phase 62 result
from a different measurement session as this series' paired control.

The `p63formal` attempt was abandoned after a Windows 260-character checkpoint
path failure. Its outputs remain, but its one successful control is not mixed
into this series. Shorter output paths and a preflight guard fixed the harness
without changing any of the 416 production-source bytes. The new reporting and
path-regression subset passes 22 tests. Full delivery tests still follow the
complete campaign, and final qualification still requires all 80 scenario runs.

The two new ordinary CPU48 controls also pass. Their elapsed times are 203.4243
and 202.2168 seconds, giving a 202.82-second median versus the latest hybrid's
73.94 seconds. The exact ratio is computed in the completed comparison report;
it is approximately 2.74x. This is a complete-system comparison, not a claim that
all savings come from GPU arithmetic. The six candidate elapsed times span
73.17-80.00 seconds; no slow successful run was removed. Matched-preparation
CPU/hybrid controls are separately extracted from the long-sequence experiment.

## Completed formal qualification

The final `p63q` series passes all 94 measured models. Six balanced fresh pairs
reduce elapsed median from **82.19 to 73.94 seconds** (10.04%), with all six
pairs faster in both elapsed and charged clocks. Ordinary CPU48 median is
**202.82 seconds**, a 2.743x complete-system ratio. The fresh targets below
70 and 65 seconds remain unmet. These are empirical repetitions on one machine,
not a theorem of performance superiority on other hardware or all inputs.

Eight fresh default-A processes per engine within the scenario experiment give
**193.18 seconds CPU versus 76.42 hybrid (2.528x)** with matched plans/files/RSS
preparation, 48-thread capacity, passive waiting, private inputs, input hashing
and reset telemetry. The worker clock differs from the fresh-pair clock; do
not cross-divide the two. The hybrid also contains earlier CPU/data-layout/I/O
improvements, so even the matched comparison is not pure GPU-hardware isolation.

Two reversed-order trials of each ten-scenario strategy give these median totals:

| Engine | Fresh processes, seconds | Persistent worker, seconds | Less elapsed |
|---|---:|---:|---:|
| CPU with equivalent preparation | 1753.83 | 1466.35 | 16.39% |
| Hybrid with equivalent preparation | 714.47 | 590.70 | 17.32% |

Persistent CPU / hybrid is 2.482x. Each total includes eight 50,000-household and
two 10,000-household scenarios, setup and input checks, excluding external audits.
Every scenario passes decisions, logical matrices, summary values and all 34
component checks. Existing departure-key spelling equivalence and diagnostic
numerical bounds remain; this is not byte identity of every output file.

| Persistent trial | Fifth-to-tenth USS growth, bytes | Active GPU growth, bytes |
|---|---:|---:|
| Hybrid 1 | 189591552 | 0 |
| Hybrid 2 | 189046784 | 0 |
| CPU 1 | 357072896 | 0 |
| CPU 2 | 128790528 | 0 |

All pass the predeclared 512 MiB USS / 128 MiB active-device limits. Every hybrid
scenario starts and final reset finishes with zero active default CuPy pool
bytes. CPU telemetry does not initialize CUDA. Nonzero host growth is disclosed;
finite boundary checks do not prove leak freedom over an unlimited lifetime.
RSS samples are observed peaks, and boundary GPU pool counts are not continuous
physical GPU peak measurements. Parsed-file cache capacity is 256 entries;
JSON plan entries have a 16 MiB limit but no global disk-cache eviction policy.

Final full tests: 619 passed in the main workspace; 618 passed and one expected
historical-checkpoint skip in the clean replica. Both report 91 existing/library
warnings. The rebuilt directory verifies all 416 measured production files,
98 package pins, public archive and selected inputs. Earlier pending statements
describe the chronological development record, not the final status.

The generated [34-component table](phase63-component-comparison.md),
`phase63-complete-comparison.json`, `phase63-sequence-qualification.json`,
JUnit reports and reconstruction/delivery receipts provide the audit trail.
The next meaningful work is a separately profiled end-to-end reduction in
remaining preparation/scheduling/destination costs and independent-machine
replication, not rebranding batch averages as meeting the fresh-run target.
