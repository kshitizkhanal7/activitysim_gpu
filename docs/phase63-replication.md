# Phase 63 replication procedure (executed qualification)

This package makes the exact local experiment reconstructible. It does not
turn one workstation's measurements into independent replication, and the
Phase 63 performance/retention qualification passes all 94 measured models.
The fresh median improves to 73.94 seconds, but the under-70-second target fails.

## Tested boundary

- Windows x64, Python 3.11.14; the measured machine has 64 GiB host RAM,
  a Threadripper PRO 5965WX and an RTX A4000 with 16 GiB GPU memory.
- The selected compiled destination path is specialized for the public MTC
  specification and NVIDIA sm86. Another GPU architecture is not silently
  certified by this recipe. Install a suitable NVIDIA driver separately.
- Git and `uv` must already be installed. The bootstrap does not install system
  drivers or change system-wide Python. It uses the repository's `.venv-phase8`.
- Reserve substantial free disk space for the public data, raw skim image,
  generated programs, independent CPU references and all retained model outputs.
  The campaign deliberately does not delete old evidence to make room.

## Prepare the pinned inputs and environment

From a checkout containing the Phase 63 files, run in PowerShell:

```powershell
./scripts/bootstrap_phase63.ps1
```

The script creates Python 3.11.14's local environment if needed, installs the
98 exact library pins in `requirements-phase63-lock.txt`, checks out ActivitySim
commit `16ab11180a26912987eb902daf945e268f3efc11`, and applies the exact integration
patch. It refuses an unexpected upstream revision or unrecognized dirty source.
It then installs both local packages with dependency resolution disabled, because
the experiment's explicit, tested pins—not newly resolved moving constraints—are
the environment contract.

`reproducibility/phase63-inputs.json` contains the exact public configuration
snapshot and license, five input-file digests, and the authenticated public-data
release URL. Configuration comes from this sealed snapshot, not today's moving
`extended` branch. The materializer downloads public **inputs**, never model
answers. It verifies the archive and each selected member and refuses to
overwrite a different existing file. Failed partial downloads remain visible
for investigation; they are not silently treated as successful downloads.

To verify an already prepared workspace without installing anything:

```powershell
./scripts/bootstrap_phase63.ps1 -CheckOnly
```

The local check and a separately rebuilt same-machine environment have passed.
The bootstrap pins ActivitySim's distribution label as well as its commit:
`1000.dev1+g16ab11180`. Git tag history would otherwise change the generated
version label. Reviewed snapshots also restore exact patched-file newline bytes;
unexpected model-code differences are rejected rather than overwritten.
The repository's `.gitattributes` also pins checkout line endings for measured
Python/PowerShell/CUDA sources and kernel metadata. The deliberately edited
coefficient CSV keeps its exact mixed upstream newlines. This prevents a user's
`core.autocrlf` setting from silently changing cache/evidence identities.
Package versions are not cryptographic attestations
of every installed binary, nor do they promise identical behavior on a different
OS or CPU. All model/output gates still have to run on the target machine.

## One-command prepared-workspace qualification

Use a new descriptive tag so the campaign cannot overwrite existing outputs:

```powershell
$env:CUPY_CACHE_DIR = Join-Path (Get-Location).Path 'isolated-cache/cupy'
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location).Path 'isolated-cache/numba'
./.venv-phase8/Scripts/python.exe scripts/run_phase63_campaign.py --tag replication-01
```

For a non-running outline, add `--plan`. The real command performs these stages
sequentially, with no overlapping model benchmarks:

1. Verify the pinned environment, exact configurations and public inputs.
2. Generate four independent regular-CPU references: default 50,000 households,
   seed 17, 10,000 households/seed 991, and a changed CDAP coefficient. Confirm
   that the coefficient change actually changes the intended activity decisions.
3. Build the raw network skim image if absent, or verify its entire content
   digest if already present. A new build is an explicitly charged preparation
   experiment, not hidden inside a warm performance claim.
4. Run six reversed-order fresh-process Phase 62/63 pairs and two regular CPU48
   fresh-process controls.
5. Run two reversed-order trials of four ten-scenario strategies: fresh hybrid,
   persistent hybrid, fresh CPU, persistent CPU. Both engines get equivalent
   preparation/diagnostic options and private raw-input copies. Every scenario
   hashes its public inputs and starts with fresh application state.
6. Check every complete output, live candidate proof, all 34 model components,
   exact logical matrices/report values, timing policy and finite memory limits.

This is 94 measured models, plus reference generation and any image build.
It is intentionally a long unattended experiment, not a quick installation
smoke test. Successful individual runs do not mean all stages have qualified.

Completed stages can be resumed with the same tag only while recorded source,
configuration and input digests still match. Failed partial outputs are kept.
After changing implementation code, start a new formal campaign: mixing old and
new measured sources would invalidate the comparison. An optional
`--reference-map` accepts already independently generated local CPU references;
the default recomputes them so a fresh checkout does not depend on historical
output directories or files from the author's machine.

## What remains manual and what must be reported

The campaign produces evidence, not a publication claim by itself. Review the
fresh-process paired results and per-component table, memory qualification,
full test suite, beginner explainer and visually rendered PDF before delivery.
Do not report a mixed 10,000/50,000-household batch average as the time for a
50,000-household fresh run. Do not describe reduced USS diagnostic sampling as
a GPU kernel speedup. Do not describe zero active CuPy array-pool bytes after
reset as zero physical GPU memory use.

## Executed reconstruction, September 20, 2026

`tmp/phase63-replica2` was staged without the original environment, public data,
runtime caches or model answers. Its bootstrap installed all 98 pinned libraries,
cloned the pinned upstream source, downloaded the 742,553,665-byte public archive,
and verified all five extracted inputs. All 416 fingerprinted production files
match the working implementation, including the shipped source-keyed CUDA binary
and the legacy environment-settings script. The latter two were added to the
measurement fingerprint before the formal campaign started.

The first clean test run exposed missing benchmark **source scripts** in the
staging recipe. Those scripts were added to the recipe and copied into the
replica; no benchmark outputs were copied. Consequently the original staging
receipt describes the initial copy, not the later source-only repair. The final
test run from the replica's own working directory passed **601 tests**, with
**one expected skip** and 91 dependency/deprecation warnings (43.63 seconds).
The skip requires a historical CPU checkpoint intentionally not staged. An
earlier 602-pass run from the original working directory could see that old
checkpoint and is not claimed as fully isolated clean-checkout evidence.

CuPy and Numba cache directories are isolated beneath the replica. Tests and
preparation warm those directories: this is a reconstructed environment, not
a claim that every measured process starts with an empty compiler cache.
The complete `p63formal` one-command campaign has started in this replica;
its performance and durability qualification is still pending. This is still
one machine, not independent second-machine replication.

The first formal comparison attempt exposed another real Windows constraint:
its candidate checkpoint path was exactly 260 characters, while this host has
long-path support disabled. The parent directory existed, but Python could not
create the checkpoint file. No model arithmetic change was needed. Campaign
outputs now use the shorter repository-local `phase63-runs` directory, with a
preflight path-length guard and regression tests; no system registry setting
was changed. Failed outputs and the incomplete `p63formal` comparison remain.

The restarted complete series is `p63q`. It explicitly supplies the four newly
generated CPU references through `--reference-map`; their 416 production-source
digests still match exactly. Only output-path orchestration changed. The raw
image is verified in full before reuse. The original image-building model took
272.80 seconds and passed all decision, matrix and report checks. Neither that
preparation run nor the abandoned comparison is part of the formal 94-model
performance series. This recovery is documented, not presented as an error-free
first execution of the command.

## Completed campaign and publication checks

The resumed `p63q` campaign finished all 94 measured models: 12 balanced fresh
comparisons, two ordinary CPU48 controls and 80 changed-scenario models. Its
sequence qualifier reports `outputs_qualified`; its comparison reports
`replicated_improvement_target_not_met`. All six pairs improve both clocks.
The default-A worker-only matched-preparation comparison is 193.18 seconds CPU
versus 76.42 hybrid; the separate fresh-pair comparison is 202.82 ordinary CPU
versus 73.94 latest hybrid. Do not mix those clock boundaries.

After measurements, the full main suite passes 619 tests, and the clean suite
passes 618 with the one expected missing historical-checkpoint skip. Both have
91 dependency/deprecation warnings. JUnit receipts record the final counts;
earlier 601/602 counts above describe earlier source revisions of the tests.

After the campaign, generate the report explicitly (the campaign does not
automatically author documentation or render a PDF):

```powershell
./.venv-phase8/Scripts/python.exe scripts/report_phase63.py --tag replication-01
./.venv-phase8/Scripts/python.exe -m pytest tests -q --junitxml=benchmark-results/local-tests.xml
```

The report refuses to overwrite an existing final comparison. Keep a new
checkout/evidence destination when repeating an already delivered experiment.
The source-only reconstruction verifier compares production bytes, archive,
inputs, package pins and the two actual JUnit runs. The publication verifier
recomputes the long-sequence qualification and matched preparation table,
checks unchanged evidence, and requires visual review of the updated PDF.
Original evidence contains absolute author-workstation paths for local auditing;
replication regenerates new evidence and independent references at its own paths.

Git publication additionally round-trips selected staged files into two new
directories using `core.autocrlf=true` and `false`, checking actual file digests.
This tests checkout byte stability, not an additional model run. Neither Git
byte identity nor a same-machine rebuild replaces a second machine's numerical
and performance qualification. The preparation costs, failed first attempt and
source-only staging repair above remain part of the delivered record.
