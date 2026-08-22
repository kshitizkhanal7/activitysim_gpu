# ActivitySim integration

## Phase 6 destination-logsum backend

ActivitySim 1.4's trip-destination component normally runs the same trip-mode
choice logsum model twice for every sampled stop: origin-to-stop and
stop-to-primary-destination. The Phase 6 patch adds the explicit setting
`DESTINATION_LOGSUM_BACKEND`, whose default is `activitysim`. Set it to
`choiceforge_combined` to preserve the two keyed random-draw blocks while
stacking deterministic preprocessing, utility evaluation, and nested-logsum
work. Three-zone models use the original path.

The ready-to-run overlay is
`benchmark-data/configs_phase6_choiceforge/trip_destination.yaml`. In the
interleaved prototype experiment it reduced the complete component median from
16.307 to 14.238 seconds and left all substantive final outputs byte-identical.
The separate segmented CUDA destination kernel is not yet called live: the
current ActivitySim loop exposes one small purpose at a time, while the measured
transfer-inclusive crossover requires about 35,000 packed interaction rows.

## Validated seam

ActivitySim 1.4's `logit.make_choices` obtains stable random draws from the
workflow state's random-number manager and calls a Numba `choice_maker` over a
chooser-by-alternative probability matrix. `choiceforge.activitysim_adapter`
mirrors that signature and preserves:

- the workflow state's ownership of random draws;
- chooser indexes in returned pandas Series;
- float64 probability and random-draw precision;
- alternative traversal and subtraction order;
- the exact-zero boundary rule;
- the first-maximum fallback for incomplete probability rows.

The integration test invokes ActivitySim's actual `choice_maker` as its oracle
and requires exact alternative equality from both ChoiceForge backends.

This seam proves compatibility, but replacing only final sampling is not the
main performance objective. A useful GPU implementation must fuse expression
evaluation, availability, logsum-exp, and choice so it never transfers or
materializes the probability table.

## Reproducible Windows environment

ActivitySim 1.4 cannot run in the project's Python 3.12 microbenchmark
environment because it pins NumPy below 1.26. The validated integration uses a
separate Python 3.11 environment:

```powershell
uv python install 3.11
uv venv .venv-asim --python 3.11
uv pip install --python .venv-asim\Scripts\python.exe `
  -r requirements-activitysim.txt -e .
```

ChoiceForge registers DLL directories for NVIDIA's pip-installed CUDA
components. This is necessary for CuPy 13 on modern Windows; CuPy 14 performs
more of this discovery automatically but requires NumPy 2 and therefore cannot
share ActivitySim 1.4's environment.

Run all framework and CUDA compatibility tests:

```powershell
.\.venv-asim\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## Public prototype MTC workflow

Create ActivitySim's packaged 25-zone benchmark:

```powershell
.\.venv-asim\Scripts\activitysim.exe create `
  -e prototype_mtc -d benchmark-data\prototype_mtc
```

Run from the generated `prototype_mtc` directory:

```powershell
activitysim run -c configs -d data -o output
```

The local validation completed all 34 model steps for 5,000 households and
8,212 persons in 70.568 seconds, with a 580.8 MB USS high-water mark. This is a
framework baseline only: ChoiceForge is not injected into the full pipeline yet.

## Sharrow status

A `sharrow: require` compile run eventually completed in 27.3 minutes and
reached approximately 15 GB USS. Three subsequent cached, single-process
production runs completed in 61.650, 59.119, and 61.790 seconds, for a median of
61.650 seconds. Final household, person, tour, and trip CSV hashes were
identical across the three runs. See the
[Phase 1 results](phase1-results-2026-08-10.md) for component medians and the
full protocol. Compilation time is reported separately and is not compared
with warm CPU or GPU execution.

## Phase 2 deterministic scheduling replay

Phase 2 completed the first three integration milestones at the more relevant
`interaction_sample_simulate` boundary. The opt-in capture harness runs the
canonical `prototype_mtc` configuration and records evaluated scheduling
terms, ragged alternative groups, probabilities, ActivitySim-owned draws, and
selected positions for all six mandatory scheduling batches. It does not edit
installed ActivitySim files or replace its random-number manager.

```powershell
.venv-asim\Scripts\python.exe scripts\capture_phase2_activitysim.py `
  --project benchmark-data\prototype_mtc\prototype_mtc `
  --output benchmark-data\prototype_mtc\prototype_mtc\output_phase2_capture `
  --capture benchmark-results\phase2-replay
```

All 4,477 choices match ActivitySim. The resident ragged kernel is faster than
the strong lowered CPU boundary, but transferring the expanded term matrix is
slower. See the [Phase 2 results](phase2-results-2026-08-10.md).

## Phase 3 compact compiler

Phase 3 replaces the expanded matrix with a compact chooser/alternative/row
ABI and compiles the required arithmetic, comparisons, and Boolean expressions
to both CUDA and a strong 48-thread Numba baseline. The native largest batch
shrinks from 151.6 MB to 22.0 MB and the GPU becomes 3.83x faster including
transfers, with zero mismatches across all six batches. Full details and raw
sample locations are in the [Phase 3 results](phase3-results-2026-08-11.md).

## Phase 4 configured backend

Phase 4 completes the mandatory-scheduling integration. A small tracked
ActivitySim 1.4 patch adds `CHOICE_BACKEND`, defaulting to `activitysim`, and
dispatches to `choiceforge` only when explicitly configured. The backend uses
ActivitySim's own feasible alternatives, mode-choice logsums, random-number
manager, choice labels, previous-tour state, and timetable updates. Unsupported
tracing and estimation paths fall back to ActivitySim.

The complete component is faster than cached Sharrow in two three-trial
protocols, and every final tour schedule and logsum field matches exactly. See
the [Phase 4 results](phase4-results-2026-08-11.md).

## Phase 5 scheduling-suite backend

Phase 5 reuses the same configured dispatch for joint, non-mandatory, and
at-work scheduling. The lowerer now handles categorical temporary assignments,
ordinary dataframe arithmetic, different subsets of timetable primitives, and
tour-owned as well as person-owned timetables.

Across three full-model trials, the sum of ActivitySim's four scheduling
workflow timers falls from 8.045 to 6.325 seconds median, a 1.272x speedup.
All seven substantive final CSVs have the same SHA-256 hashes as the cached-
Sharrow reference in every trial. See the
[Phase 5 results](phase5-results-2026-08-11.md).

## Next integration milestone

1. Add tiled online logsum-exp for large destination alternative sets.
2. Keep compatible skim and encoded chooser inputs GPU-resident.
3. Preserve ActivitySim tracing and estimation contracts on the GPU path.
4. Reproduce on larger public models and additional GPU hardware.

## Phase 7 destination backend

Apply the tracked patch to ActivitySim 1.4, then place the Phase 7 overlay before
the base configuration. The patch defaults both selectors to `activitysim`, so
installing it alone changes no model behavior.

```yaml
DESTINATION_LOGSUM_BACKEND: choiceforge_tripnum_batched
DESTINATION_NESTED_LOGIT_BACKEND: choiceforge_cuda_mtc21
```

The first selector batches all purpose segments for one trip number through a
single OD+DP preprocessor pass. Unsupported estimation, three-zone, multiple-
preprocessor, or purpose-dependent-preprocessor cases fall back before random
sampling. The second selector replaces only the canonical MTC 21-mode nest
reduction; a CUDA error falls back on the already-evaluated utilities.

The integration patch is mechanically checked against the pristine ActivitySim
1.4 wheel:

```powershell
git apply --check --directory=tmp/phase6-wheel/extracted `
  integration/activitysim-1.4-choiceforge.patch
```

## Phase 8 pinned current ActivitySim integration

Phase 8 validates a separate patch against ActivitySim commit
`16ab11180a26912987eb902daf945e268f3efc11` (reported package version
`1000.dev1+g16ab11180`). The current API adds an `alts_context` argument at the
scheduling boundary and no longer exports the historical `THREE_ZONE` LOS
constant. ChoiceForge forwards a non-null alternatives context to the original
ActivitySim implementation and treats numeric zone-system value 3 as the
legacy three-zone case. Both changes are conservative: installing the patch
still leaves every backend selector at `activitysim` by default.

Create the pinned environment and verify the patch:

```powershell
git clone https://github.com/ActivitySim/activitysim.git tmp\activitysim-phase8-source
git -C tmp\activitysim-phase8-source checkout 16ab11180a26912987eb902daf945e268f3efc11
git -C tmp\activitysim-phase8-source apply `
  ..\..\integration\activitysim-current-choiceforge.patch
uv venv --python 3.11 .venv-phase8
uv pip install --python .venv-phase8\Scripts\python.exe `
  -e tmp\activitysim-phase8-source -e ".[gpu,test]"
.venv-phase8\Scripts\python.exe -m pytest -q
```

The expected local result is 33 passing tests. The Phase 8 overlay lives in
`benchmark-data/configs_phase8_choiceforge`; it explicitly enables the four
scheduling components and the trip-number-batched destination backend. Run the
reproducible 50,000-household interleaved experiment with:

```powershell
pwsh scripts\run_phase8_interleaved.ps1 -Households 50000 -Repetitions 3
.venv-phase8\Scripts\python.exe benchmarks\benchmark_phase8_activitysim.py
```

The public workflow setup and compile cache come from ActivitySim's official
`sharrow-contrast/mtc_mini` workflow. The 500-household `sharrow: test` compile
pass is a correctness/compilation stage and is not included in warmed
production comparisons. All A/B trials use `sharrow: require`, 24 OpenBLAS
threads, identical data, and fresh processes.
