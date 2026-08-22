# Phase 10: scheduling precision guard

## Purpose

Phase 9 found nine changed scheduling choices in a 100,000-household, 1,454-zone
Prototype MTC run when the experimental 32-bit ChoiceForge scheduler was enabled.
This phase makes the safety rule executable: a GPU scheduling result may never
change the model merely because a floating-point cumulative probability lands on
the other side of an ActivitySim random draw.

## Implementation

`interaction_sample_simulate_choiceforge` now accepts
`precision_guard="shadow_fallback"`.

For each supported scheduling batch, it:

1. snapshots the ActivitySim random-channel offsets;
2. obtains the normal GPU result using ActivitySim's controlled draw;
3. restores the offsets and runs ActivitySim's authoritative implementation on
   the same batch and draw;
4. compares every chosen alternative; and
5. returns ActivitySim's result whenever any choice differs.

Restoring offsets is essential. Calling the fallback after the GPU without that
restore would consume the *next* draw and could create a different downstream
random-number sequence. The new unit test checks both the returned fallback
choice and the one-draw-only offset advance.

The new settings overlay is
`benchmark-data/configs_phase10_shadow_scheduling`. It is explicit and opt-in:
the default remains `CHOICEFORGE_PRECISION_GUARD: off`. The production Phase 9
destination-only overlay is unchanged.

## What the guard proves and what it costs

This is a correctness guard, not a scheduling-speed configuration. It performs
ActivitySim's scheduling calculation in every guarded batch, so its scheduling
time includes a full shadow calculation. Its purpose is to make a true 64-bit
or near-boundary selective fallback safe to develop and measure later.

## Full-geography smoke result

On August 17, 2026, a fresh baseline/guarded pair used 1,001 households and all
1,454 public MTC zones. The guarded run logged 14 scheduling-batch comparisons,
each with zero mismatches. The seven substantive final CSV files were
byte-identical.

| Boundary | Baseline | Guarded run | Note |
| --- | ---: | ---: | --- |
| Four scheduling components | 6.9 s | 7.6 s | Shadow verification is expected to add work. |
| Trip destination | 17.8 s | 10.7 s | The existing destination backend remains enabled. |
| All 34 model steps | 89.290 s | 82.417 s | One pair; not a performance claim. |

The machine-readable result is `benchmark-results/phase10-smoke-summary.json`.

## 50,000-household guard finding and resource result

On August 17, 2026, the guard was exercised over all four tour-scheduling
components of a 50,000-household run across all 1,454 zones. It compared 30
batches and detected
three one-choice GPU/CPU disagreements: two mandatory-work batches and one
mandatory-school batch. In every case the guard restored the ActivitySim
random-number offsets and returned ActivitySim's authoritative choice for the
batch. This is direct evidence that the experimental scheduler cannot yet be
advertised as unconditionally exact.

The same run used the conservative 8 GB ActivitySim chunk target. It stayed at
about 20--25 GB private process memory through its most demanding phases, but
the general chunk profile made it operationally unsuitable: trip destination
alone took 12:03, and the normal trip-scheduling retry loop expanded into many
slow, per-chunk iterations. The run was intentionally stopped after 40 minutes
once this resource/throughput trade-off had been established. It is not a
completed final-file comparison and is not used for a timing claim.

The executable capped configuration remains useful as a diagnostic safety
profile. The runner now applies its cap to *both* A/B conditions through
ActivitySim CLI settings, rather than applying it only to the ChoiceForge
condition. A fresh capped smoke launch confirmed `chunk_size: 8000000000` and
`chunk_training_mode: training` in the baseline log before it was intentionally
stopped as redundant. Do not use this profile to report GPU speedups.
The machine-readable partial-run record is
`benchmark-results/phase10-50k-capped-guard-partial.json`.

## 100,000-household gate and resource finding

The first Phase 10 baseline at 100,000 households completed in 310.932 seconds
with a 10.2 GB RSS high-water marker. Its ChoiceForge partner failed before
scheduling because Windows rejected a deeply nested checkpoint path. The runner
now uses deliberately short output names (`o-<tag>-<condition>-...`), which
removes that path-length failure mode.

The compact-path rerun then reached CDAP with about 38.5 GB private memory and
was terminated without a Python traceback. This reproduces the Phase 9 finding:
100,000 households is not a stable repeated A/B size on this 63.9 GB workstation.
No 100k correctness or speed claim is made from these incomplete attempts.

## Reproduction

Run the smoke gate or, on a sufficiently large host, the 100k gate:

```powershell
.\scripts\run_phase10_mtc_shadow_guard.ps1 -Households 1001 -RunTag phase10-smoke
.\scripts\run_phase10_mtc_shadow_guard.ps1 -Households 100000 -RunTag p10
```

Then summarize the produced manifest:

```powershell
.\.venv-phase8\Scripts\python.exe .\benchmarks\benchmark_phase9_mtc_full.py `
  --manifest .\benchmark-results\phase9-mtc-full-phase10-smoke-runs.json `
  --output .\benchmark-results\phase10-smoke-summary.json
```

For the real 100k gate, use a host with at least 512 GB RAM and run several
alternating fresh-process baseline/guarded pairs. Hash every substantive
`final_*.csv` file before reporting a result. Only after the guard is exact at
scale should a selective near-boundary fallback replace the full shadow pass.
