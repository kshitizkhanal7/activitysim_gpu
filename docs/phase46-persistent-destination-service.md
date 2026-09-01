# Phase 46: persistent exact destination service

## Result

Phase 46 turns Phase 45's 19 public destination-sampling calls into one
prewarmed, persistent CUDA service. It owns reusable device workspaces,
generates ActivitySim's keyed NumPy MT19937 random draws bit-for-bit on CUDA,
evaluates each probability exponential once, accelerates the sparse exact
boundary with ActivitySim's authoritative Numba routine, and packs sampled
rows without two global pandas sorts.

Three fresh matched Phase 45/46 pairs used the public Prototype MTC extended
model with 50,000 households, 1,454 zones, and all 34 model steps. All three
candidate runs produced seven byte-identical published CSV files and zero
changed decision cells.

| Qualified measurement | Phase 45 | Phase 46 | Result |
|---|---:|---:|---:|
| five target components, median aggregate | 33.0 s | 29.4 s | 1.122x; 10.9% lower |
| complete model lifecycle, median | 148.3 s | 146.935 s | 1.009x; 0.92% lower |
| complete lifecycle pairs won | - | 3 of 3 | repeatable direction |
| changed decision cells | - | 0, 0, 0 | exact |

The lifecycle candidate includes 1.69-1.73 seconds of cold Phase 46 prewarm in
every run. The gain is therefore not created by hiding compile time outside the
measurement.

## Architecture

`Phase46DestinationService` is created once for a model run. Its grow-only
workspace covers the largest public call and is then reused:

- float32 utility and probability-weight surfaces;
- compact choices, selected probabilities, duplicate flags, and pick counts;
- guard, validation, and row-maximum vectors;
- MT19937 state, seed, offset, and output buffers.

The measured maximum allocation is 391,398,912 bytes. That covers 38,525,184
dense utility cells (26,496 choosers by 1,454 zones) and remains below the
declared 1 GiB limit.

Four reviewed expression shapes are precompiled: the 9-term school program,
11-term workplace program, and two 7-term tour-destination forms. Unknown
expressions still fail closed. Prewarm also compiles ActivitySim's exact Numba
sampling helper before model timers start; the benchmark adds the full prewarm
duration back into lifecycle time.

## Exact keyed random numbers on CUDA

ActivitySim owns one seed and consumed-draw offset per chooser. Phase 46 reads
that ledger, reconstructs NumPy `RandomState`'s scalar-seeded MT19937 state on
CUDA, consumes the recorded offset, produces the requested float64 values, and
then advances ActivitySim's authoritative offset by exactly the same amount.

Each qualified run covers:

- 38 GPU RNG calls: 19 sampling calls and 19 final-choice calls;
- 402,780 chooser rows;
- 6,243,090 float64 random values.

Focused tests include different seeds and offsets through 650 consumed draws.
A live 23,509-row shadow compared 705,270 gradeschool draws bit-for-bit and
found zero mismatches.

## Exact probability arithmetic

Phase 46's probability kernel computes NumPy-compatible float32 exponential
weights once per cell. A generated pairwise reduction tree reproduces NumPy's
float32 sum association for all 1,454 alternatives. CUDA then performs the
inverse CDF and duplicate accounting.

The first full-model attempt exposed one changed gradeschool decision. GPU RNG
was exact, but Phase 46 had subtracted the row maximum in place to reuse the
utility buffer. Phase 45 retained unshifted utilities for its sparse exact
adjudicator. The transformed values were mathematically equivalent but changed
a boundary sample from zone 660 to 659, and a later final choice from 585 to
583.

The fix preserves the unshifted utility surface. A reusable row vector stores
the maximum, and the weight kernel performs explicit float32 subtraction while
the exact adjudicator receives the original values. Fresh end-to-end output
verification then returned to zero differences. This failed run is important
evidence for the rule that arithmetic layout, not only algebra, belongs in the
reproducibility contract.

## Sparse exact boundary and compact packing

The existing conservative CDF guard identifies 7,313 of 201,390 chooser rows
(3.63%). Those rows need NumPy's authoritative probability normalization.
Phase 45 selected their 30 alternatives in a Python loop. Phase 46 calls
ActivitySim's own Numba `sample_choices_maker_preserve_ordering`; a focused
benchmark measured identical choices and probability bits at about 160x the
speed of the Python helper.

After sampling, Phase 45 builds a pandas frame and performs two global sorts.
Phase 46 sorts each chooser's at-most-30 compact values directly with NumPy and
constructs the already ordered frame. A 26,496-row focused benchmark was exact
and 4.53x faster for this packing operation.

## Component results and tradeoff

| Target component | Phase 45 median | Phase 46 median | Result |
|---|---:|---:|---:|
| school location | 7.9 s | 5.9 s | 1.339x |
| workplace location | 10.1 s | 8.7 s | 1.161x |
| joint-tour destination | 3.3 s | 3.7 s | 0.892x; 0.4 s slower |
| non-mandatory-tour destination | 9.5 s | 8.7 s | 1.092x |
| at-work subtour destination | 2.5 s | 2.3 s | 1.087x |

The small joint-tour family regressed in the median. Its segments cannot fully
amortize the persistent GPU RNG and service boundaries. This is recorded rather
than averaged away. The five-family aggregate improved in every pair by 3.1,
4.0, and 4.3 seconds, and the complete lifecycle improved in every pair by
1.09-3.47 seconds.

## Reproduction

```powershell
.\scripts\run_phase46_persistent_destination_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p46final
.\.venv-phase8\Scripts\python.exe `
  scripts\summarize_phase46_qualification.py
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

Primary evidence:

- `benchmark-results/phase46-p46final-summary.json`
- `benchmark-results/phase46-p46final-qualification.json`
- `benchmark-results/phase46-p46final-exact-{1,2,3}.json`
- `benchmark-results/phase46-p46final-gpu-{1,2,3}.json`

## Claim boundary and next work

Phase 46 is not a wholly GPU-only ActivitySim model. Dense preliminary utility,
probability weights, inverse-CDF sampling, duplicate accounting, and keyed RNG
generation run on CUDA. ActivitySim still owns orchestration and the random
ledger. NumPy/Numba handles measured sparse probability-boundary rows. The
richer final sampled-choice utility remains the authoritative Sharrow CPU
evaluator.

The next large opportunity is a strict compiled final-choice service that keeps
the 4,696,676 sampled alternative rows in compact device form, translates the
reviewed Sharrow programs from the shared expression IR, and proves identical
utility arithmetic before becoming authoritative. Small-segment batching or a
device-side persistent RNG-state registry should be tested specifically against
the joint-tour regression. Whole-model impact will remain bounded until more of
the currently non-targeted 34-step pipeline enters the same persistent runtime.
