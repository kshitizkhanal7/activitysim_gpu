# Phase 60: whole-step runtime optimization

## Result

Six balanced complete-model pairs reduce median elapsed time from **98.42 to
90.16 seconds**, an **8.40% reduction** (1.092x). Every candidate improves both
elapsed and charged time. All six candidates finish in **89.91-90.69 seconds**,
meeting both the under-100-second target and the under-95-second stretch target.
Charged medians fall from **91.92 to 83.46 seconds**.

The stronger fresh regular CPU control, with 48 Numba threads, has a median
elapsed time of **201.61 seconds** across two runs: the complete Phase 60 hybrid
is **2.236x faster**, or **55.28% less elapsed time**, against that configuration.
Fresh one-thread CPU controls have medians of **296.18 seconds elapsed** and
**290.45 seconds charged** (3.285x elapsed speedup). Both CPU configurations,
including every step, are reported separately in the
[complete 34-step comparison](phase60-component-comparison.md). No model steps
or output products were dropped. All decisions, all 115 logical matrices and
values throughout all 24 summary reports pass their declared contracts.
Three final-source changed scenarios pass as well.

| Pair | Order | Phase 59 elapsed (s) | Phase 60 elapsed (s) |
|---|---|---:|---:|
| 1 | 59 then 60 | 98.61 | 89.93 |
| 2 | 60 then 59 | 97.91 | 90.49 |
| 3 | 59 then 60 | 98.15 | 90.07 |
| 4 | 60 then 59 | 98.30 | 90.69 |
| 5 | 59 then 60 | 98.80 | 90.25 |
| 6 | 60 then 59 | 98.55 | 89.91 |

The current Phase 59 control is itself faster than its historical 109.84-second
median. The phase's causal comparison uses the newly matched 98.42-second
control; subtracting Phase 60 from the historical number would overstate the
demonstrated incremental gain. Cache/system/session effects are not separately
identified here. The six pairs are retained in full, without outlier removal.

This is a successful whole-runtime phase, **not a new GPU kernel advantage**.
Its new savings come from CPU preparation and an existing parallel CPU
calculation around the established GPU path. The stronger CPU comparison is
intentionally prominent, even though it produces a smaller headline ratio.

### Where the incremental time changed

| Step | Phase 59 (s) | Phase 60 (s) | Observed reduction (s) |
|---|---:|---:|---:|
| Non-mandatory tour frequency | 6.70 | 2.90 | 3.80 |
| Mandatory tour scheduling | 10.30 | 8.80 | 1.50 |
| Household activity coordination (CDAP) | 6.30 | 4.90 | 1.40 |
| Non-mandatory tour scheduling | 6.30 | 4.90 | 1.40 |
| Summary reports | 4.20 | 3.85 | 0.35 |

These are descriptive complete-run component medians, not isolated timings or
additive attribution. Not every component improves: joint-tour destination
changes from 1.45 to 1.70 seconds and trip destination from 8.60 to 8.75 seconds.
The experiment does not establish whether each small movement is noise or a
cross-step effect. The full-model paired result, not selected component rows,
determines success. Against regular CPU48, frequency is only 3.30 versus 2.90
seconds; its large incremental saving over Phase 59 mainly comes from using
existing CPU parallelism, as the real-input sweep demonstrates.

The fail-closed qualifier reports `replicated_improvement_wall_target_met` in
`benchmark-results/phase60-formal-qualification.json`. The complete comparison
is `benchmark-results/phase60-complete-comparison.json`. Development failures,
instrumented profiles and the thread sweep are retained separately and excluded
from the six-pair performance result.

## Approved objective

Reduce launch-to-exit time below 100 seconds in every candidate of six balanced
Phase 59/60 pairs, with a 95-second stretch target. At planning time Phase 59's
historical measured median was 109.84 seconds; newly matched Phase 59 controls,
not that historical number, determine the result above.

Preserve all modeled decisions, exact random ledgers, all 115 matrices, values
in all 24 summary reports under the declared formatting contract, and existing diagnostic
bounds. Qualify changed scenarios on the final implementation. Retain failed
experiments and processor-attribution controls. Update the beginner explainer
and PDF after qualification and publish the tested changes.

## Initial measurement plan

Profile complete mandatory scheduling, trip destination, non-mandatory tour
frequency, CDAP, non-mandatory scheduling, departures, mode choice and summary
output. Separate preparation and arithmetic before selecting implementation
work. Profiling is explicitly instrumented and excluded from timing evidence.
No source edits or concurrent heavy work during benchmark children.

Priorities from Phase 59 component medians: mandatory scheduling 10.80 seconds,
trip destination 8.90, non-mandatory frequency 7.10 and CDAP 6.90. These are
component ceilings, not additive promised savings. The compact retry kernel
is too small for processor switching alone to deliver the whole-phase target.

## Profile and first implementation

`p60profile1` completed and passed its inherited decision/matrix gates, but is
instrumented and not timing evidence (128.59 s elapsed). The expanded cProfile
shows substantial avoidable CPU preparation around the existing GPU runtime:

- CDAP: 3.29 s constructing specifications and 3.26 s constructing Sharrow
  flows in the instrumented run. First change constructs the same ordered slug
  matrix in arrays from current coefficients, retaining original numeric mapping.
- Mandatory scheduling: 0.94 s in Python slot mapping, 0.53 s in semantic slot
  preparation, plus two-column sorting. Replace these with collision-free int16
  pair keys and vectorized, fully checked slot mappings; do not remove live CPU
  boundary adjudication.
- Non-mandatory scheduling: 3.79 s in row-wise iterator lookups. Use compact
  period lookup and direct two-index timetable gathers, avoiding a repeated
  row-by-day intermediate matrix.
- Non-mandatory frequency: 4.33 s in Sharrow's compiled `idotter`. Its generated
  implementation is already `parallel=True, fastmath=False`; test CPU thread
  masks before attempting an arithmetic rewrite. The reduction order within
  each chooser/alternative remains the upstream implementation.
- Summaries: 2.39 s constructing repeated bin labels. Format each distinct
  bounds/rank tuple once, retaining observed-rank calculation, numeric conversion
  and original handling of empty/missing inputs.

These instrumented nested costs overlap and must not be added into a savings
claim. None is presented as new GPU hardware superiority. Optimized CPU input
preparation still feeds the existing generated GPU arithmetic.

The first 34 tests exposed an empty-bin edge case; fixed by retaining the
upstream empty/missing path. All 34 then passed. The first integrated run
`p60dev1` failed because ActivitySim's default CLI reset NUMBA_NUM_THREADS after
the pool had initialized. The replacement uses upstream `--fast` only to
preserve pool capacity, explicitly keeps other numerical libraries at one
thread, and masks Numba to one outside the selected step. Both Phase 59 and
Phase 60 comparison processes use the same initial pool/mask arrangement.
The failed run is not performance evidence.

## First complete result and real-input CPU sweep

`p60dev2` (24 frequency threads) passes every modeled decision, all matrices
and values in all 24 summary reports. Its 87.82 seconds charged / 94.52 seconds elapsed
is one development observation, not yet replicated qualification.

The instrumented `p60threadcontrol` repeats the actual Sharrow interaction
dispatcher with the live input arrays and coefficients for all eight frequency
segments. Every output byte agrees at every tested thread mask. The same
upstream `parallel=True, fastmath=False` program runs on each mask; no GPU
advantage is claimed for this improvement. One warm call per mask precedes
three repetitions with reversed thread order on alternating repetitions.

| Numba threads | Sum of eight segment compute medians |
|---|---:|
| 1 | 4.2245 s |
| 4 | 1.9337 s |
| 12 | 0.6762 s |
| 24 | 0.3735 s |
| 48 | 0.2679 s |

The selected default is 48. This clock includes dispatcher allocation and
arithmetic, but excludes preparing the input arrays, compilation and model
publication. Inputs are live and hashed, not saved expected answers. The
instrumented full run (125.00 seconds elapsed) is explicitly excluded from
performance qualification. Its modeled outputs, matrices and summaries pass.
The separate regular-CPU control must also be tested with multithreaded Numba;
retaining only the older single-thread comparison would overstate attribution.

## Final-source scenario qualification

All 467 repository tests pass before final timing (39.37 s, 70 upstream/test
warnings). Three changed candidates then pass the actual external decision,
matrix and report-value oracles using the frozen source/configuration checks:

| Scenario | Households | Charged (s) | Elapsed (s) |
|---|---:|---:|---:|
| Seed 991 | 10,000 | 49.86 | 56.42 |
| Seed 17 | 50,000 | 84.92 | 91.62 |
| Seven departure attempts | 50,000 | 83.82 | 90.43 |

These use the independently generated regular CPU scenario outputs from Phase
59, not the seed-0 reference for all inputs. They qualify changed inputs; they
are not an interleaved scenario-speed experiment. Existing diagnostic bounds,
all 115 matrices and the declared 24-report exact-value contract remain intact.

## Replication commands and boundaries

Use the pinned public MTC environment documented in Phase 59, including the
verified native skim artifact and ActivitySim source patch. No clean-machine
one-command setup is claimed. Use fresh tags and retained output directories.
The formal series is serial; do not run other tests or benchmarks concurrently:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRO60CPU48 --modes regular --repetitions 2 --phase60 --regular-numba-threads 48
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRO60 --modes gpu,candidate --repetitions 6 --phase60
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag REPRO60CPU1 --modes regular --repetitions 2 --phase60 --regular-numba-threads 1
```

With `--phase60`, `gpu` means the complete Phase 59 live/sparse-writer control,
not Phase 58; `candidate` means Phase 60. The harness enables all required
Phase 59 flags in both versions. Both have a 48-thread Numba pool initially
masked to one. Phase 60 activates 48 only during non-mandatory tour frequency,
then restores the previous mask. Other numerical libraries remain one-thread;
both matrix writers retain their four CPU compression workers. `--fast` is
used to prevent the upstream CLI from resetting the pool capacity, not to
enable fast-math or change model settings.

Changed-scenario commands follow the Phase 59 reproduction examples, replacing
`--phase59 --live-mandatory --sparse-matrices` with `--phase60`. To independently
regenerate CPU references, use `--modes regular,candidate`; to reuse a verified
reference use `--modes candidate --scenario-baseline <output-directory>`.
Supply the seed-991 overlay and `--households 10000`, the seed-17 overlay, or
the retry-seven overlay respectively. Do not prune outputs before qualification.

```powershell
.venv-phase8/Scripts/python.exe scripts/qualify_phase60.py --comparison benchmark-results/phase58-p60formal-summary.json --scenarios benchmark-results/phase58-p60scenario10k-summary.json benchmark-results/phase58-p60scenarioseed17-summary.json benchmark-results/phase58-p60scenarioretry7-summary.json --output benchmark-results/phase60-formal-qualification.json
.venv-phase8/Scripts/python.exe scripts/report_phase60_comparison.py --qualification benchmark-results/phase60-formal-qualification.json --regular1 benchmark-results/phase58-p60cpu1-summary.json --regular48 benchmark-results/phase58-p60cpu48-summary.json --output benchmark-results/phase60-complete-comparison.json --markdown docs/phase60-component-comparison.md
```

For fresh replication substitute the new tags in those paths. The qualifier
re-reads the published summaries, verifies each live implementation report,
requires all six pairs in balanced order and compares per-attempt retry
counts. It hashes implementation reports, summary inputs and qualification
code. A target miss or a slower pair remains visible in the status; it is
not discarded. All process launches and operational overhead remain in wall
time. Independent output verification is outside the model clocks.

No new consumer of the entity store is claimed in this phase. Profiling
selected explicit slot/timetable preparation and existing CPU parallelism
first; trip destination's normalized packet/transfer path remains a further
opportunity. Likewise, live CPU boundary rechecks remain intact rather than
being replaced with an unqualified approximate or saved-answer shortcut.

## Interpretation and replication limits

The workstation has a 24-core / 48-thread AMD Threadripper PRO 5965WX and an
NVIDIA RTX A4000 with 16 GB of device memory. The pinned environment uses
Python 3.11.14, NumPy 2.4.6, Numba 0.66, CuPy 14.1.1 and Sharrow 2.16.2.
ActivitySim's source revision is
`16ab11180a26912987eb902daf945e268f3efc11`, with the retained integration patch
in `integration/activitysim-current-choiceforge.patch`. The harness fingerprints
the actual patched source, not just the upstream Git revision.

These are fresh model processes with previously populated compilation and
filesystem caches, using the same local SSD and a 50,000-household sample in
the public 1,454-zone geography. They are not cold-install timings, the entire
regional population, or predictions for a different GPU. The two CPU controls
at each thread setting are measured separately from the six balanced hybrid
pairs; they are useful complete-model comparisons, not a randomized causal
experiment isolating GPU hardware. Neither CPU setting is claimed to be the
strongest possible CPU implementation.

The supported arithmetic contract is layered: exact modeled decisions and
logical matrix values; the declared exact-value summary contract; bounded
diagnostic logsum differences. It is not byte identity of all files or a
mathematical proof for every future configuration. Tests of multiple seeds,
household counts and departure limits provide substantially more evidence
than one successful run, but cannot establish universal equivalence.

Source fingerprints and evidence hashes detect later changes. They do not
install dependencies, download public inputs, or recreate omitted proprietary
data; this experiment uses public inputs but still needs its documented
environment and native skim preparation. The retained run commands, reports,
oracle summaries and per-step timings make the measured claim auditable.

## Final release checks (September 13, 2026)

- The actual qualification CLI completed with
  `replicated_improvement_wall_target_met`; the complete-comparison CLI also
  reverified both CPU configurations' summary outputs.
- After all timed processes exited, the full repository suite passed again:
  **467 passed, 70 warnings in 39.22 seconds**. No tests or PDF rendering ran
  concurrently with the final model timing series.
- The current production source fingerprint exactly matches the fingerprint
  shared by the six pairs, four fresh CPU controls and three changed scenarios.
- The explainer Markdown and 154-page PDF include the final results, both CPU
  controls, assumptions, negative development evidence and remaining work.
  The cover, one-minute summary and new concluding sections were rendered and
  visually inspected; text extraction found no replacement characters.
- Only this phase's implementation, tests, documentation and compact evidence
  are included in publication. Older unrelated untracked experiments and local
  benchmark output directories are preserved.
