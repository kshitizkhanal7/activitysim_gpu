# Phase 9A: full-geography public Prototype MTC benchmark

## Purpose

Phase 8A established an end-to-end improvement on ActivitySim's 50,000-household, 190-zone `mtc_mini` workflow. Phase 9A moves to the public full-geography Prototype MTC extended data: 2,875,192 households, 7,566,527 persons, and 1,454 zones. This is 57.5 times as many households and 7.7 times as many zones as Phase 8A.

The public asset is `data_full.tar.zst` from the `activitysim-prototype-mtc` v1.3.4 release. Its pinned SHA-256 is `b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3`.

## Local safety boundary

The upstream `configs_mp/settings.yaml` documents the complete 2,875,192-household run as requiring 64 processes and 432 GiB RAM. The Phase 8 workstation has 63.9 GB RAM. It must not be used for an unchunked all-household run.

The official single-process full-geography workflow uses a 100,000-household study size. On the Phase 8 workstation, an initial 100,000-household baseline completed, but subsequent 100,000-household attempts were not stable within the 63.9 GB memory envelope. Phase 9A therefore establishes its local gate at 50,000 households with full 1,454-zone skims and Sharrow required. This is an external-scale integration and correctness gate, not a final full-population superiority claim.

## Validated local result

One fresh-process baseline/ChoiceForge pair ran on August 17, 2026 at 50,000 households and 1,454 zones. The ChoiceForge condition enables only trip-destination acceleration; all scheduling decisions stay on ActivitySim's path. All seven substantive final CSVs were byte-identical: 50,000 households, 132,536 persons, 175,579 tours, and 442,682 trips.

| Boundary | Baseline | ChoiceForge | Observed ratio |
| --- | ---: | ---: | ---: |
| All 34 model steps | 198.794 s | 187.010 s | 1.063x |
| Trip destination | 39.2 s | 28.0 s | 1.400x |

This pair is a successful full-geography correctness and scale result, not a statistically sufficient performance claim. A high-memory host must run at least three interleaved fresh A/B pairs before the observed ratios are promoted to medians or a superiority statement.

## Reproduction

The public release and extracted data are located at:

`benchmark-data/phase9-mtc-full/prototype_mtc_extended/data_full`

Run the validated local-scale pair:

```powershell
.\scripts\run_phase9_mtc_full_ab.ps1 -Households 50000 -Repetitions 1 -RunTag f50
.\.venv-phase8\Scripts\python.exe .\benchmarks\benchmark_phase9_mtc_full.py `
  --manifest .\benchmark-results\phase9-mtc-full-f50-runs.json `
  --output .\benchmark-results\phase9-mtc-full-f50-summary.json
```

The runner uses `configs_sh` plus `configs`, with the full data directory. The ChoiceForge condition adds `benchmark-data/configs_phase9_choiceforge`, which enables the trip-destination backends and deliberately leaves scheduling on ActivitySim. It records wall time, ActivitySim's high-water memory markers, and refuses to overwrite an output directory.

On Windows, use a short `-RunTag` such as `d2` if a deeply nested checkpoint path fails to be created. The runner treats `Time to execute all models until this error` as a failure, not completion.

### Scheduling correctness gate

The experimental scheduling overlay is isolated in `benchmark-data/configs_phase9_experimental_scheduling` and is not part of a Phase 9 performance claim. A 100,000-household full-geography A/B gate found nine changed schedule choices, caused by floating-point decision boundaries, plus one propagated non-mandatory-tour frequency change. The run is retained as a regression fixture. Scheduling stays on the ActivitySim path until ChoiceForge has a high-precision boundary safeguard that reproduces every affected choice and preserves downstream random-number behavior.

Do not escalate beyond 50,000 households on the 63.9 GB workstation. On a high-memory host, start with 100,000 households and increase only after the prior scale remains comfortably within memory. For a defensible performance claim, use at least three fresh, interleaved A/B trials at a fixed scale.

## Full-population high-memory runbook

Use a Windows or Linux host with at least 512 GiB RAM, a CUDA-capable GPU with at least 24 GB VRAM, local SSD scratch space, and a pinned ActivitySim/ChoiceForge environment. Start from the upstream `configs_mp` configuration and enable the documented 400 GB chunk budget and 60 processes only after a smaller chunk-training pass succeeds.

Run baseline and ChoiceForge in alternating fresh processes, preserve timing and chunk logs, and compare hashes for every substantive `final_*.csv` file. Report end-to-end time separately from scheduling, trip-destination, and nested-logit boundaries. Do not extrapolate a partial-scale speedup to the 2.875-million-household population.
