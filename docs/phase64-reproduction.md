# Phase 64 reproduction: preparation, measurements and limits

This extends the Phase 63 pinned Windows/RTX A4000 workspace. It is not a claim
that Phase 64 has already been rebuilt on another machine. Preserve the Phase 63
bootstrap, source patch, 98-package lock and public input provenance. Do not
compare a new run with an old headline while calling it a contemporaneous pair.

## Raw input preparation

From the repository root, with the Phase 63 environment and public data ready:

```powershell
.venv-phase8/Scripts/python.exe scripts/build_phase64_inputs.py --tag myinputpreparation
```

This explicitly prepares raw numeric household, person and land-use values;
the preparation receipt records the cost, bytes and bit-exact round trip. It
refuses to replace existing artifacts. Existing valid artifacts are checked
on every model read, including the entire original CSV and artifact SHA-256.
Changed CSVs cannot hit their old artifact, even if size and timestamp are
unchanged. Unsupported dtypes use the original parser. No predicted choices,
probabilities or sampled household subset are stored in this artifact.

## Larger complete models

Run one command at a time. Choose unused tags; never overwrite earlier evidence.

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase64_scale.py --tag my100k --households 100000 --features expressions,inputs,location_boundary
.venv-phase8/Scripts/python.exe scripts/run_phase64_scale.py --tag my250k --households 250000 --features expressions,inputs,location_boundary
```

Each command generates an ordinary CPU reference first, then runs the hybrid
and audits its complete outputs against that reference. One-second resource
monitoring makes these diagnostic feasibility runs, not a repeated speed
qualification. The harness stops only its own process tree if host available
RAM stays below 2 GiB for three samples or disk free space falls below 8 GiB.
No automatic output deletion occurs. Review the failure receipt and logs before
retrying with a new tag. GPU out-of-memory failures remain failures, never an
excuse to silently change precision, model size or output checks.

## Repeated qualification

After selecting the candidate, inspect the predeclared campaign before running:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase64_campaign.py --tag myformal --features expressions,inputs,location_boundary --plan
.venv-phase8/Scripts/python.exe scripts/run_phase64_campaign.py --tag myformal --features expressions,inputs,location_boundary
```

The campaign has 42 sequential models, described in the Phase 64 phase notes.
It expects four independent CPU scenario references from Phase 63. Its resumed
stages must match current source, configuration and available data digests;
partial failed output is preserved and requires investigation and a new tag.
The launcher explicitly sets `OMP_WAIT_POLICY=PASSIVE` for ordinary CPU as well
as hybrid children. If a retained campaign has invalid CPU controls, use a new
`--cpu-control-tag` when resuming; the qualifier checks the replacement's source,
inputs, thread policy and outputs, and the old controls remain on disk. This is
how the `p64f1` campaign replaces its two originally unspecified-policy CPU runs
with `p64cpu2`, without relabeling their metadata or rerunning valid stages.
Do not edit production code, the integrated runner, comparison harness or batch
worker while a measured run is active. Do not run tests, rendering, another
benchmark or another model concurrently. The final qualifier distinguishes
correct outputs, acceptable finite memory growth, pairwise speed improvement
and the under-70-second target. They are separate tests, not interchangeable.

## What the result can prove

To reproduce GPU attribution, first capture one diagnostic model, then run the
standalone calculation and full-model replacement control sequentially:

```powershell
.venv-phase8/Scripts/python.exe scripts/run_phase58_comparison.py --tag mycapture --phase64 --phase63-features plans,files,rss --modes candidate --repetitions 1 --capture-mode-inputs benchmark-data/my-trip-inputs --phase61-capture-inputs benchmark-data/my-live-inputs --output-root phase63-runs
$env:NUMBA_NUM_THREADS='48'
$env:OMP_WAIT_POLICY='PASSIVE'
.venv-phase8/Scripts/python.exe scripts/benchmark_phase64_modes.py --inputs benchmark-data/my-trip-inputs benchmark-data/my-live-inputs/tour_modes --output benchmark-results/my-mode-controls.json
.venv-phase8/Scripts/python.exe scripts/run_phase64_ablation.py --tag myablation
```

The capture run is explicitly excluded from speed evidence. The standalone
benchmark checks upstream nested probabilities, sweeps five CPU thread settings
and runs nine alternating repetitions. The ablation uses two reversed full-model
pairs. It changes only tour/trip reduction and reports its actual transfer bytes.

Complete-model wall clocks include startup, live numerical checks and output
work. The separate worker clock also includes input hashing and reset checks;
do not mix the two clocks. The hybrid still uses CPU code, including numerical
boundary adjudication. Raw input preparation and expression-code reuse are
CPU-side improvements available to both engines in the matched comparison.

Controlled mode reduction benchmarks compare the same host utility arrays and
random draws, sweep CPU threads and separately time GPU transfers and resident
execution. They exclude utility generation, RNG and boundary adjudication
equally. Their ratios are not ratios for all ActivitySim or all GPU work.
One workstation and a supported public model are not a universal performance
or mathematical floating-point-equivalence proof. Report actual sample counts,
target misses, preparation costs and failed attempts alongside successes.

## Final evidence audit

After all measurements, rerun the complete tests into a new JUnit XML and log,
then consolidate the campaign, ablation, primitive controls and both final-code
scale receipts with `scripts/report_phase64_results.py`. Its `--help` lists the
required paths and tags. It rejects changed source, incomplete output checks,
unexercised destination safeguards, a memory qualification failure or test
regressions. Passing consolidation does not imply passing the speed target:
the result records those booleans separately.

The beginner Markdown and PDF must describe the actual clocks and limits.
After selectively staging publication files, `verify_phase64_git_checkout.py`
checks their bytes with both `core.autocrlf=true` and `false` in new retained
temporary directories. `verify_phase64_delivery.py` checks the consolidated
evidence, current source, staged byte audit and a visual-review receipt for the
actual PDF. The receipt is created only after rendering and inspecting the
cover, opening pages, appendix transition and every new appendix page. A
successful text extraction is not a visual review.

Prepared binary inputs and full model-output directories are not Git artifacts.
Recreate them from the pinned public inputs using the commands above. Historical
development captures explain diagnosis; the final-code measurements and staged
source are the reproduction contract. Pinning and exact output tests do not
guarantee the same elapsed seconds on another machine or under another load.
