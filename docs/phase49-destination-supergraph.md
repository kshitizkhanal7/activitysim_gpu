# Phase 49: inter-stage destination supergraph

## Result

Phase 49 removes the host round trip between the CUDA mode-choice logsum
reducer and the CUDA sampled-destination final utility. The public ActivitySim
workflow still controls model order and tables, but its intermediate pandas
`mode_choice_logsum` column is now a compatibility placeholder for the three
tour-destination families. School and workplace return only the selected
person-level mode logsums required in published outputs.

The complete 50,000-household Prototype MTC extended workload has 19 calls,
4,696,676 sampled alternatives, and 201,390 final choosers. Every call passed
an exact repeated-index identity check before the consumer could read the
resident vector.

| Qualified measurement | Phase 48 host bridge | Phase 49 resident bridge | Result |
|---|---:|---:|---:|
| exact 19-call handoff, median of 31 | 13.265 ms | 1.226 ms | 10.822x; 90.759% lower |
| full sampled logsum D2H | 37,573,408 B | 0 B | eliminated |
| compact selected-output D2H | not applicable | 862,352 B | 107,794 required values |
| redundant final-logsum H2D | 18,786,704 B | 0 B | eliminated |
| net transfer avoided | - | 55,497,760 B | per complete run |
| float32 handoff bit mismatches | - | 0 | exact |
| whole-model median | 144.100 s | 143.517 s | 1.004x observed |
| whole-model pairs won | - | 2 of 3 | not replicated |
| changed published decision cells | - | 0, 0, 0 | exact |

The promoted performance claim is the directly measured inter-stage handoff.
The 0.405% lifecycle median improvement is reported, but it is not promoted as
replicated whole-model superiority because the candidate lost one pair by
0.016 seconds.

## What changed

Phase 48 performed this sequence for each sampled alternative vector:

1. CUDA produced a float64 mode-choice logsum.
2. CuPy downloaded every value to host memory.
3. ActivitySim stored the values in a pandas alternatives column.
4. Phase 47 narrowed the column to float32.
5. CuPy uploaded it for the final destination utility.

Phase 49 instead publishes a device packet containing the original float64
vector, its exact float32 device conversion, the complete repeated chooser-ID
sequence, and the producer trace identity. The final utility consumes that
packet only if its alternatives index is byte-for-byte equal in value and
order. A missing producer, reordered duplicate, changed row count, partial
scheduling contract, stale packet, or leftover selected-output packet stops
the run.

For joint, non-mandatory, and at-work destinations, no mode-choice logsum is
downloaded. For school and workplace, final choice first determines one
alternative row per person on CUDA. The bridge gathers the original float64
logsum for those 107,794 selected rows, downloads 862,352 bytes, and restores
the published column by stable person ID. It never downloads the other
2,436,123 location alternatives.

## Why the arithmetic remains exact

The old final utility used `np.asarray(mode_choice_logsum, dtype=float32)` and
then uploaded those bits. Phase 49 uses a CUDA float64-to-float32 conversion
and compares its entire result with the old path in the 31-repetition handoff
qualifier. The mismatch count is zero across all 4,696,676 values.

Published school/workplace mode logsums are not reconstructed from the
float32 final-utility values. The bridge retains the original float64 device
vector and gathers selected rows from it, preserving the former output dtype
and bits. The three fresh full-model pairs independently compare seven final
CSV files and report zero changed decision cells.

Phase 48's exact exponential table, probability reduction, guarded choice,
compact logsum calculation, and resumed keyed MT19937 state remain unchanged.
Phase 49 changes transport and ownership, not the travel model or its random
ledger.

## Upstream compatibility reasoning

The design follows current upstream contracts rather than pretending that
ActivitySim is already GPU-native:

- ActivitySim's Sharrow `require` mode fails when compiled evaluation cannot
  run, while `test` mode evaluates both paths. Phase 49 likewise fails closed
  on a changed inter-stage ABI.
- Sharrow's `DataTree` and reusable `Flow` show how linked datasets and a
  compiled expression can form a reusable backend boundary. Phase 49 extends
  that idea to device-owned intermediate results.
- Current ActivitySim documentation explicitly says Sharrow accelerates
  utility specifications but not preprocessors or postprocessors. The larger
  remaining destination cost is therefore pandas preprocessing and Python
  orchestration, not this now-optimized transfer.

Primary upstream references:

- [ActivitySim: Using Sharrow](https://activitysim.github.io/activitysim/develop/dev-guide/using-sharrow.html)
- [Sharrow DataTree and Flow API](https://activitysim.github.io/sharrow/api.html)
- [ActivitySim runtime performance guidance](https://activitysim.github.io/activitysim/develop/users-guide/performance/index.html)

## Proof stack

1. **Contract tests:** duplicated IDs pass; missing, reordered, malformed, or
   unconsumed packets fail; selected float64 outputs restore in output order.
2. **Exact 19-call boundary:** 31 paired repetitions use every public call
   shape; float32 bit mismatches are zero.
3. **Complete workload:** all 19 calls cover 4,696,676 alternatives and finish
   with zero pending packets and zero fallbacks.
4. **Compact output:** only 107,794 selected school/workplace float64 values
   cross back to the host.
5. **Fresh-process replication:** three Phase 48/49 pairs run the complete
   34-step model.
6. **Independent output verification:** all seven published files are exact in
   every pair.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.\.venv-phase8\Scripts\python.exe scripts\benchmark_phase49_handoff.py `
  --repetitions 31 `
  --output benchmark-results\phase49-handoff-qualification.json

.\scripts\run_phase32_full_model_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p49final `
  -Baseline phase48 -CandidatePhase 49

.\.venv-phase8\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Primary evidence:

- `benchmark-results/phase49-handoff-qualification.json`
- `benchmark-results/phase49-p49final-summary.json`
- `benchmark-results/phase49-p49final-{base,gpu,exact}-{1,2,3}.json`

## Claim boundary and next opportunity

Phase 49 proves a much faster and exact device handoff, plus an exact compact
published-output strategy. It does not prove a large whole-model gain. The
optimized handoff saves about 12 milliseconds inside a model that takes about
144 seconds; even infinite acceleration of this boundary cannot remove a
second.

The audit measured roughly 1.14 seconds of binding resolution, 0.77 seconds of
host packing, 0.31 seconds of input upload, and 2.65 seconds of mode-utility
kernel time across these same 19 calls, with 1.95 GB of dense row inputs. The
next ambitious phase must replace the dense ActivitySim logsum preprocessor
with a compact owner/sample/land-use ABI and generate its 41-float/31-integer
row state on the device. That is the remaining destination boundary large
enough to seek a repeated component and lifecycle gain.
