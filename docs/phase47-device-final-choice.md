# Phase 47: strict CUDA sampled final choice

## Result

Phase 47 moves the reviewed final sampled-choice utility programs for school,
workplace, joint-tour, non-mandatory-tour, and at-work destinations from
Sharrow CPU evaluation to one prewarmed CUDA compiler. It covers all 19 public
calls, 201,390 choosers, and 4,696,676 sampled alternatives.

An exhaustive live shadow found zero utility-bit mismatches across all
4,696,676 alternatives. Three fresh 50,000-household Phase 46/47 pairs each
produced seven byte-identical published CSV files and zero changed decision
cells. Every pair favored Phase 47 after charging both cold prewarms.

| Qualified measurement | Phase 46 | Phase 47 | Result |
|---|---:|---:|---:|
| strict final-choice boundary, median | 1.786 s | 0.362 s | 4.94x |
| five target components, median aggregate | 29.8 s | 27.7 s | 1.076x |
| complete 34-step lifecycle, median | 144.600 s | 143.815 s | 1.0055x |
| lifecycle pairs won | - | 3 of 3 | replicated |
| changed decision cells | - | 0, 0, 0 | exact |

The lifecycle gains were 1.933, 1.785, and 0.485 seconds. The median gain is
0.785 seconds, or 0.54%. This is a real but deliberately narrow improvement
over an already GPU-accelerated Phase 46 control.

## Compiler and runtime

The compiler accepts exactly four reviewed expression programs: school,
workplace, tour destination, and at-work destination. It also accepts only the
observed compact widths 21, 25, 29, and 30. Unknown formulas or widths stop;
they never silently fall back to a different calculation.

For every compact sampled row, CUDA reads the resident distance skim and
computes distance splines, income interactions, size terms, shadow-price
adjustments, mode-choice logsums, and sampling corrections. The generated
kernel uses the same grouped-left float32 reduction ABI as Sharrow/OpenBLAS.
It writes directly into the reusable padded chooser-major workspace, avoiding
the generic pandas interaction frame and CPU utility compiler in production.

The persistent Phase 46 service supplies the workspace and exact GPU MT19937
draws. Phase 47 reuses those allocations; it does not introduce a second dense
surface.

## Exactness design

Phase 47 has three independent layers of evidence:

1. The live exhaustive shadow evaluates all 19 final programs through both the
   generated CUDA kernel and authoritative Sharrow CPU flow. Every float32
   utility bit matched across 4,696,676 alternatives.
2. The CUDA final selector uses a conservative boundary flag. Only 7 of
   201,390 final-choice rows entered the exact NumPy adjudicator in each
   production run, and none disagreed before correction. The prior 1.7-second
   cold Numba compilation was removed by an exact NumPy float64-CDF/search
   contract, separately tested against ActivitySim's compiled helper.
3. An independent table verifier compared all three fresh pairs. Accessibility,
   households, joint-tour participants, land use, persons, tours, and trips
   were byte-identical in every pair.

The exhaustive shadow stays opt-in because it deliberately reruns Sharrow and
would erase the production speedup. Production adjudicates only flagged rows;
the three independent end-to-end comparisons prove the released path.

## Performance detail

The final-choice runtime median fell from 1.786 to 0.362 seconds. Phase 47's
CUDA utility kernels themselves consumed roughly 0.064 seconds in the shadow
run; most remaining boundary time is compact preparation, one utility download
for ActivitySim's exact normalization/logsum contract, and controlled choice.

| Target component | Phase 46 median | Phase 47 median | Result |
|---|---:|---:|---:|
| school location | 6.1 s | 5.2 s | 1.173x |
| workplace location | 9.2 s | 8.7 s | 1.057x |
| joint-tour destination | 3.4 s | 3.5 s | 0.971x |
| non-mandatory-tour destination | 8.4 s | 8.4 s | 1.000x |
| at-work subtour destination | 2.5 s | 2.2 s | 1.136x |
| five-family aggregate | 29.8 s | 27.7 s | 1.076x |

The five-family aggregate improved by 1.8-2.3 seconds in every pair. Joint
tours remain 0.1 seconds slower in the median, so batching small segments is
still justified. Whole-model speedup is smaller than boundary speedup because
the replaced final-choice work was only about 1.2% of Phase 46's lifecycle.

## Reproduction

```powershell
.\scripts\run_phase47_device_final_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p47final
.\.venv-phase8\Scripts\python.exe `
  scripts\summarize_phase47_qualification.py
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

For the expensive exhaustive arithmetic shadow, set
`CHOICEFORGE_PHASE47_SHADOW=1` and run the Phase 47 integrated runner. The
production three-pair script intentionally leaves it off.

Primary evidence:

- `benchmark-results/phase47-p47final-summary.json`
- `benchmark-results/phase47-p47final-qualification.json`
- `benchmark-results/phase47-p47final-exact-{1,2,3}.json`
- `benchmark-results/phase47-p47final-gpu-{1,2,3}.json`
- `benchmark-results/phase47-p47shadow2-gpu.json`

## Claim boundary and next major phase

Phase 47 is not a GPU-only ActivitySim model. CUDA is authoritative for the
reviewed final utility and normally for selection. ActivitySim/NumPy remains
authoritative for compact normalization and logsums after one device-to-host
utility transfer, plus seven guarded rows. ActivitySim still owns model
orchestration, tables, random-ledger semantics, and non-targeted components.

The next major opportunity is therefore not another isolated arithmetic
kernel. It is a resident destination graph: retain compact samples, mode
logsums, final utilities, normalization, and selected results across adjacent
stages; batch small segments; implement a strict GPU logsum/probability ABI;
and expose it behind an upstream Sharrow/ActivitySim backend contract. That
removes the remaining transfer and Python call boundaries while preserving the
same fail-closed compiler, exhaustive shadows, matched pairs, and independent
replication gates.
