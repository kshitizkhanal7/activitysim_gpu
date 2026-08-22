# Benchmark and evidence protocol

## Phase 6 destination evidence

Phase 6 uses two complementary boundaries. The replay benchmark consumes all
30 captured real trip-destination segments and compares a fused batched Numba
CPU implementation with one segmented CUDA launch. It reports both
transfer-inclusive and resident timing, exact choice mismatches, numerical
logsum error, and a prefix-based crossover curve. The ActivitySim experiment
runs fresh processes in interleaved order A1/B1/A2/B2/A3/B3 and hashes seven
substantive final CSVs from all six trials. Do not combine the kernel's 1.464x
speedup with the component's 1.145x or whole-model 1.052x speedup: they describe
different measured boundaries.

## Claims this repository may support

A valid result has the form:

> On the documented hardware, software versions, input model, and sample size,
> ChoiceForge reduced median component runtime from X to Y while producing zero
> choice mismatches and a maximum logsum error of Z.

It must not be generalized to all ActivitySim workloads or all GPUs.

## Required baselines

The synthetic microbenchmark currently compares a materializing NumPy reference
with CUDA. Before an ActivitySim performance claim, add:

1. ActivitySim legacy evaluation;
2. precompiled Sharrow, single process;
3. Sharrow with the recommended CPU configuration;
4. ActivitySim multiprocessing where applicable;
5. ChoiceForge including host/device transfers;
6. ChoiceForge with model arrays already GPU-resident, labeled separately.

Phase 1 adds fused serial and 24-core Numba baselines to the synthetic linear
choice benchmark. These are the primary CPU comparison for the primitive
because they avoid the NumPy oracle's global utility matrix. See
[Phase 1 results](phase1-results-2026-08-10.md).

Phase 2 adds a deterministic six-batch replay of ActivitySim's real
`mandatory_tour_scheduling` interaction boundary. It compares a NumPy
BLAS/Numba CPU implementation with the ragged fused CUDA kernel and labels
resident and transfer-inclusive results separately. See
[Phase 2 results](phase2-results-2026-08-10.md).

Phase 3 compiles the required expressions from a 22.0 MB compact ABI instead
of transferring the 151.6 MB evaluated-term matrix. The CPU comparator is a
generated 48-thread Numba expression kernel, not the NumPy oracle. On the
native largest batch, the GPU is 3.83x faster including transfers with zero
choice mismatches. See [Phase 3 results](phase3-results-2026-08-11.md).

Phase 4 times the complete ActivitySim mandatory scheduling workflow component,
including upstream mode-choice logsums, timetable work, compact packing,
random draws, transfers, result mapping, and timetable mutation. Three normal
runs improve from a 5.515-second cached-Sharrow median to 4.925 seconds, with
zero final-tour mismatches. A matched-checkpoint protocol independently shows
8.599 to 8.014 seconds. See
[Phase 4 results](phase4-results-2026-08-11.md).

Phase 5 applies the same explicit backend to mandatory, joint, non-mandatory,
and at-work tour scheduling. The four complete workflow timers sum to an 8.045-
second cached-Sharrow median and a 6.325-second ChoiceForge median: 1.272x
faster. SHA-256 verification finds all seven substantive final CSVs byte-
identical in all three ChoiceForge trials. See
[Phase 5 results](phase5-results-2026-08-11.md).

Phase 7 captures the exact 21-mode utility matrices before ActivitySim's nested
reduction: 30 batches, 107,854 rows, and 18.119 MB in FP64. Thirty-one warmed,
interleaved repetitions compare ActivitySim's pandas nest reducer with the
fused CUDA kernel, including both transfers. The live proof then alternates
fresh Phase 6 and Phase 7 processes A1/B1/A2/B2/A3/B3 and hashes all substantive
outputs. See [Phase 7 results](phase7-results-2026-08-11.md).

Phase 8 pins current ActivitySim commit
`16ab11180a26912987eb902daf945e268f3efc11` and uses the official public
`sharrow-contrast/mtc_mini` workflow at 50,000 households and 190 zones. The
production experiment alternates fresh A/B processes three times, uses
`sharrow: require`, fixes OpenBLAS at 24 threads, retains ActivitySim's complete
component timers, and hashes seven final CSVs. It separately captures 30 real
nested-logit batches with 3,186,130 rows and replays them 11 times in alternating
CPU/GPU order. The GPU measurement includes both transfers. See
[Phase 8 results](phase8-results-2026-08-11.md).

Phase 9A downloads the pinned public `prototype_mtc_extended` full-data asset:
2,875,192 available households, 7,566,527 persons, and 1,454 zones. On the
64 GB workstation, the validated 50,000-household full-geography A/B pair keeps
seven substantive files byte-identical with ChoiceForge enabled only for trip
destination. Its observed trip-destination ratio is 1.400x (39.2 to 28.0
seconds); the one-pair whole-model ratio is 1.063x (198.794 to 187.010 seconds).
These are not median claims. The Phase 9 report records the failed experimental
scheduling exactness gate, the 100,000-household memory limit, and the
high-memory interleaved protocol. See [Phase 9 full-MTC results](phase9-full-mtc.md).

## Timing rules

- Perform enough full-workload warm-ups to exclude JIT compilation and settle
  allocation pools. Phase 1 uses ten and retains every measured sample.
- Synchronize the CUDA stream before reading elapsed time.
- Report every sample, the median, and preferably a bootstrap 95% interval.
- Report kernel-resident and transfer-inclusive measurements separately.
- Pin dependency versions and record CPU, GPU, driver, OS, thread counts, and
  power settings.
- Run the benchmark on an otherwise idle machine.

## Correctness rules

- Pass identical random draws to every backend.
- Preserve alternative ordering and availability.
- Require exact selected-alternative equality.
- Report maximum and percentile logsum error.
- Compare row counts, schemas, invalid rows, and output labels.
- Run full end-to-end ActivitySim invariants for persons, tours, and trips.

## Running the current benchmark

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe benchmarks\benchmark_linear_choice.py `
  --choosers 100000 --alternatives 32 --features 16 `
  --repetitions 7 --output benchmark-results\a4000-100k-32.json
```

Run multiple sizes. Small cases establish the crossover point where transfer
and launch overhead exceed parallel-compute gains; large cases establish
throughput and memory scaling.

Run the real-data Phase 2 replay after capturing the MTC component:

```powershell
$env:PYTHONPATH = "src"
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase2_activitysim_replay.py `
  --repeats 9 --scales 1 2 4
```

Run the compact Phase 3 compiler benchmark:

```powershell
$env:PYTHONPATH = "src"
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase3_compact_scheduling.py `
  --repeats 15 --scales 1 2 4 8
```

Summarize the matched Phase 4 component trials:

```powershell
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase4_activitysim_component.py
```

Summarize the Phase 5 full-model scheduling-suite trials and verify hashes:

```powershell
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase5_scheduling_suite.py
```

Capture, benchmark, and summarize Phase 7:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.venv-asim\Scripts\python.exe scripts\capture_phase7_nested_logsums.py `
  --project benchmark-data\prototype_mtc\prototype_mtc `
  --output benchmark-data\prototype_mtc\prototype_mtc\output_phase7_nested_capture `
  --capture benchmark-results\phase7-nested-logsum-capture `
  --config benchmark-data\prototype_mtc\prototype_mtc\configs_sharrow `
  --config benchmark-data\configs_phase5_choiceforge `
  --config benchmark-data\configs_phase7_choiceforge
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase7_nested_logit.py
.venv-asim\Scripts\python.exe benchmarks\benchmark_phase7_activitysim_component.py
```

Run and summarize Phase 8's current-version, public-scale experiment:

```powershell
pwsh scripts\run_phase8_interleaved.ps1 -Households 50000 -Repetitions 3
.venv-phase8\Scripts\python.exe benchmarks\benchmark_phase8_activitysim.py

.venv-phase8\Scripts\python.exe scripts\capture_phase7_nested_logsums.py `
  --project benchmark-data\phase8-mtc-mini\prototype_mtc_sf `
  --output benchmark-data\phase8-mtc-mini\prototype_mtc_sf\output-phase8-nested-capture `
  --capture benchmark-results\phase8-nested-logsum-capture `
  --config benchmark-data\configs_phase8_choiceforge `
  --config benchmark-data\phase8-mtc-mini\prototype_mtc_sf\configs_sh
.venv-phase8\Scripts\python.exe benchmarks\benchmark_phase7_nested_logit.py `
  --capture benchmark-results\phase8-nested-logsum-capture `
  --output benchmark-results\phase8-nested-logsum-summary.json `
  --repetitions 11 --phase 8B
```
