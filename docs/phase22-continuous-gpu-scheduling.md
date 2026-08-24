# Phase 22: continuous raw-skim-to-schedule execution

Phase 22 joins the two halves proved separately in Phase 21. Generated CUDA
now evaluates the public MTC mode-choice expressions from raw network skims,
reduces the 21-mode nested logit, scatters the resulting logsums into a GPU
cache, prepares feasible timetable alternatives, makes scheduling choices, and
mutates the timetable for the next sequential tour batch.

The live public-model result is exact and faster in every one of three paired
runs. The median paired speedup is **1.257x**; the ratio of median CPU and GPU
times is **1.261x**. All 81,983 mandatory-tour TDD, start, and end outputs match
the frozen ActivitySim checkpoint in every run.

This is deliberately described as a predominantly GPU path with an exact
boundary adjudicator, not as an absolutely CPU-free path. ActivitySim/Sharrow
and CUDA do not have identical float32 dot, exponential, and reduction
semantics. The GPU detects numerically ambiguous draws without looking at the
expected answer. Sharrow re-evaluates only those cases. There were 57 such
tours, or 0.0695% of the 81,983 tours, and the adjudicator downloaded 11,400
bytes of logsum data per run.

## What changed

The implementation adds four connected boundaries:

1. `mtc21_nested_logsums_cuda(..., return_device=True)` returns the final
   float64 logsum vector without downloading it.
2. `_simple_simulate_mtc21_logsums_cuda` accepts a device sink and sends stable
   chooser identity plus ActivitySim's authoritative `out_period` and
   `in_period` labels with the device values.
3. `IntegratedGpuMandatoryScheduler` scatters those values on CUDA, regenerates
   feasible alternatives from its device timetable, consumes ActivitySim's
   exact random draws, and mutates the timetable after every batch.
4. The scheduling kernel reports each draw's distance from its nearest
   cumulative-probability boundary. A 2e-6 empirically qualified guard routes
   only ambiguous rows through the exact Sharrow resolver.

The period labels matter. School and university representative logsum rows use
an 18:00 representative time for the `EV` period even though the ordinary MTC
time-label rule maps the literal hour 18 to `PM`. Inferring the period solely
from the representative hour created an apparent duplicate cache slot. Phase
22 now carries the categorical labels supplied by ActivitySim and fails on an
unknown label.

## Arithmetic finding and resolution

The first continuous run changed tour `13282973` from TDD 169 to 168. Its
ActivitySim random draw was `0.9779642212405256`, only about 8.5e-10 beyond the
CPU cumulative boundary after TDD 168. The discrepancy was not an identity,
random-stream, or timetable error.

The forensic capture established that Sharrow:

- materializes 65 float32 scheduling features, including two zero-coefficient
  temporary expressions;
- evaluates a float32 `np.dot`;
- disables max-shift overflow protection when ActivitySim's
  `skip_failed_choices` setting is enabled; and
- uses the CPU NumPy float32 exponential and sum before the Numba choice loop.

CUDA `expf`, GPU reduction order, and the compact 63-term expression compiler
are all numerically valid, but they are not a bit-identical implementation of
that CPU recipe. Hard-coding the saved TDD would have made the test meaningless.
Instead, the CUDA kernel produces a boundary-margin measurement. The exact resolver
receives only the ambiguous chooser rows, the already-consumed random draws,
and their 25-slot float64 logsum cache. It runs the real ActivitySim/Sharrow
choice contract and returns the resolved TDD before the GPU timetable mutates.

This design gives an exact result now and isolates the remaining upstream work:
a future Sharrow CUDA backend or a dedicated expression compiler can replace
the resolver only after it defines identical dot, exponential, sum, and choice
semantics. The 2e-6 guard is qualified for this frozen public benchmark; a new
model, specification, or architecture must requalify it rather than assuming it
is a universal numerical-error bound.

## Public benchmark results

The paired benchmark resumes the same 50,000-household public Prototype MTC
checkpoint immediately before mandatory scheduling. Each CPU and GPU run reads
the same raw skim collection and writes a fresh pipeline.

| Pair | CPU control (s) | Integrated GPU (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 42.358 | 36.599 | 1.157x |
| 2 | 40.389 | 31.963 | 1.264x |
| 3 | 40.250 | 32.030 | 1.257x |
| Median | 40.389 | 32.030 | 1.257x median paired |

The ratio of medians is 1.261x. GPU won every pair. These are full resumed
mandatory-scheduling component times, including raw-skim setup and ActivitySim
orchestration—not isolated kernel-only timings.

Every GPU run also recorded:

- 6 of 6 integrated sequential batches;
- 1,210,124 generated-CUDA raw-skim rows;
- zero CUDA candidate fallbacks;
- zero bulk modeled-logsum downloads;
- exactly 57 guarded boundary rows;
- exactly 11,400 boundary-logsum download bytes;
- exact ActivitySim random draws;
- zero TDD, start, or end mismatches; and
- a restart/audit checkpoint with selected-TDD and timetable hashes.

The canonical evidence is
[`phase22-live-paired-summary.json`](../benchmark-results/phase22-live-paired-summary.json).
Its six input report hashes make the summary restartable and tamper-evident.
The primary live run is
[`phase22-integrated-live.json`](../benchmark-results/phase22-integrated-live.json),
and its device checkpoint is
[`phase22-integrated-checkpoint.json`](../benchmark-results/phase22-integrated-checkpoint.json).

## Reproduction

Start from a fresh copy of the frozen pipeline because ActivitySim opens a
resumed pipeline for update. Then run the integrated gate:

```powershell
./.venv-phase8/Scripts/python.exe scripts/run_phase22_integrated_scheduling.py `
  --project benchmark-data/phase9-mtc-full/prototype_mtc_extended `
  --config-overlay benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs_sh `
  --data benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full `
  --output <fresh-output-with-copied-pipeline> `
  --reference-pipeline benchmark-data/phase9-mtc-full/prototype_mtc_extended/o-p17modeproof16-baseline-50000-1/pipeline.parquetpipeline `
  --report <live-report.json> `
  --checkpoint <checkpoint.json> `
  --kernel-reports <empty-short-report-directory>
```

Run the CPU control with `scripts/run_phase21_activitysim_logsum.py --engine
cpu`. Build the paired proof from at least three CPU and three GPU reports:

```powershell
python scripts/summarize_phase22_live_pairs.py `
  --output benchmark-results/phase22-live-paired-summary.json `
  --cpu <cpu-1.json> --cpu <cpu-2.json> --cpu <cpu-3.json> `
  --gpu <gpu-1.json> --gpu <gpu-2.json> --gpu <gpu-3.json>
```

The summarizer fails unless every CPU control is exact, every GPU proof gate
passes, GPU wins every pair, and the boundary population is deterministic.

## What this proves—and what it does not

Phase 22 proves that this machine can accelerate the complete resumed public
mandatory-scheduling component, from raw skims through final sequential
schedules, while preserving all published outputs exactly. It also proves
that the rare arithmetic boundary can be detected independently of an answer
key and resolved with a tiny, measured exception.

It does not prove an absolutely CPU-free ActivitySim model. Pandas and
ActivitySim still orchestrate batches, own the random stream, write the
pipeline, and adjudicate 0.0695% of near-boundary choices. It does not cover
non-mandatory, joint, at-work, trip scheduling, destination choice, or the
complete model runtime. It is one GPU architecture and one public model.

The next major work is to port the same certified interface to the other
scheduling components, then move orchestration and pipeline state behind a
device-native runtime. Separately, the Sharrow/CUDA arithmetic contract should
define a portable, correctly reproducible dot/exp/sum policy; only then can the
57-row resolver be removed honestly.
