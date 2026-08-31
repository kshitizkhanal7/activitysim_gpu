# Phase 44: compact exact final-choice runtime

## Outcome

Phase 44 replaces ActivitySim's generic table mechanics around final sampled
trip-destination choice with a reviewed compact ragged runtime. It covers all
30 purpose/direction programs in the public 50,000-household Prototype MTC
extended model: 91,524 trip choosers and 2,094,156 sampled alternative rows.

Three fresh matched Phase 43/44 pairs all favored Phase 44. Median
`trip_destination` time fell from 10.1 to 9.5 seconds (0.6 seconds saved,
5.94% lower, 1.063x). Median time across all 34 model steps fell from 158.1
to 156.4 seconds (1.7 seconds saved, 1.08% lower, 1.011x). The compact final
boundary itself fell from 1.092 to 0.822 seconds (0.270 seconds saved, 1.328x).

Every pair produced seven byte-identical published CSV files. Independent
verification found zero changed decision cells, zero destination-logsum
difference, and zero mode-logsum difference.

## What was compiled and what remained authoritative

The public final-choice specification has 16 expression slots: two temporary
distance values and 14 effective utility terms. Those terms cover destination
size, zero-size availability, directional distance, proximity to the primary
origin/destination, sampling correction, and two trip-mode logsums plus their
availability masks.

`trip_destination_final.py` validates the exact ordered expression tuple and
the one-column finite coefficient contract. An unknown, reordered, added, or
removed expression fails closed. The runtime then supplies a narrow interaction
frame to Sharrow's authoritative compiled CPU evaluator. This choice preserves
Sharrow's float32 intermediate values and OpenBLAS dot-product association.

Phase 44 does **not** claim that this final 16-slot evaluator is a new GPU
kernel. The larger trip pipeline still uses the previously qualified GPU
sampling and logsum kernels. This phase removes Python/pandas overhead at the
last CPU boundary while deliberately retaining the evaluator whose arithmetic
already matches ActivitySim exactly.

## Compact ragged representation

The generic ActivitySim path joins a wide chooser table onto every sampled
alternative, counts options with a group-by, pads utilities with `np.insert`,
and maps rectangular choice positions back through a generic index operation.
Phase 44 instead validates that alternative rows are contiguous by chooser and
constructs one offset array:

```text
trip offsets:       [0, 23, 51, 79, ...]
alternative rows:  [trip 0 options][trip 1 options][trip 2 options]...
```

For chooser `i`, rows `offsets[i]:offsets[i+1]` are its legal alternatives.
A Numba loop pads the compiled utility vector directly into ActivitySim's
required float32 rectangle with the same `-999` dummy utility. Only columns
referenced by the reviewed program or required by Sharrow relationships are
materialized. Categorical purpose codes are repeated without converting their
meaning or ordering.

ActivitySim remains authoritative for `utils_to_probs`, zero-probability
handling, its keyed random ledger, `make_choices`, and result indexing. Phase
43's compact controlled random state is inherited unchanged.

## Sharrow relationship compatibility

The final expression program reads only `od_skims` and `dp_skims`, but
ActivitySim's Sharrow trip-destination template declares a broader relationship
graph. The compact runtime therefore retains `trip_period`,
`purpose_index_num`, and the metadata of `dnt_skims`. The last item is used only
to name the primary-origin relationship; its values are not evaluated by the
reviewed expression program. This is explicit in code because silently pruning
these fields makes Sharrow's graph construction fail before compilation.

## Measured evidence

| Pair | Phase 43 final boundary | Phase 44 final boundary | Phase 43 trip destination | Phase 44 trip destination | All-model Phase 43 -> 44 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.089 s | 0.821 s | 10.0 s | 9.5 s | 160.9 -> 153.7 s |
| 2 | 1.092 s | 0.825 s | 10.1 s | 9.8 s | 157.6 -> 156.4 s |
| 3 | 1.108 s | 0.822 s | 10.2 s | 9.5 s | 158.1 -> 156.8 s |
| median | 1.092 s | 0.822 s | 10.1 s | 9.5 s | 158.1 -> 156.4 s |

Median Phase 44 boundary decomposition across all 30 calls:

| Stage | Time |
|---|---:|
| build narrow frames | 0.093 s |
| Sharrow utility evaluation | 0.582 s |
| ragged-to-rectangular padding | 0.028 s |
| ActivitySim probability math | 0.045 s |
| ActivitySim controlled choice | 0.057 s |
| complete boundary | 0.822 s |

The large 7.2-second whole-model improvement in pair 1 includes unrelated
component variance and must not be attributed solely to Phase 44. The stronger
causal evidence is the stable 0.27-second final-boundary saving, the
`trip_destination` win in all three pairs, and the exact-output proofs. The
median whole-model result is reported because all three complete runs also
moved in the favorable direction, but it remains a small 1.1% effect.

## Reproduction and proof gates

Run the matched experiment and consolidated qualification with:

```powershell
.\scripts\run_phase44_compact_final_ab.ps1 -Repetitions 3 -Households 50000 -RunTag p44final
.\.venv-phase8\Scripts\python.exe scripts\summarize_phase44_qualification.py
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

The authoritative qualification requires all three candidate reports to pass
every runtime gate, exact 30/91,524/2,094,156 workload coverage, the reviewed
16-slot/14-term ABI on every call, three byte-exact output verifiers, a faster
compact boundary in every pair, a faster target component in every pair, and a
faster complete model in every pair. The full suite passes 183 tests.

Evidence:

- `benchmark-results/phase44-p44final-summary.json`
- `benchmark-results/phase44-p44final-qualification.json`
- `benchmark-results/phase44-p44final-exact-{1,2,3}.json`
- `benchmark-results/phase44-p44final-gpu-{1,2,3}.json`

## Next major opportunity

The remaining median 0.582-second utility evaluation is the clearest direct
target, but a dedicated CUDA evaluator must reproduce Sharrow's exact
float32/OpenBLAS arithmetic semantics. The safe path is to lower the already
reviewed expression tuple into the shared strict IR, generate CPU and CUDA
implementations from one numeric policy, shadow every utility and choice, and
promote only after exhaustive bitwise agreement. A larger whole-model dent will
also require reusing this compact compiler boundary in other expensive choice
components rather than optimizing trip destination alone.
