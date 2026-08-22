# Phase 13: strict CPU reference and real Sharrow comparison gate

## Outcome

Phase 13 is complete. ChoiceForge now has a normative CPU evaluator generated
from the strict IR, an exact comparison gate against Sharrow feature and utility
matrices, first-divergence diagnostics, deterministic JSON evidence, and a
reproducible public ActivitySim run.

This phase does not replace Sharrow and does not make a new speed claim. Its
purpose is to freeze the arithmetic contract that the Phase 14 CUDA generator
must match. During every Phase 13 comparison, ActivitySim/Sharrow remained
authoritative and the strict evaluator could not change a utility, probability,
choice, random draw, or final output.

## Strict numeric contract

Phase 13 originally recorded this policy as IR version 2. Phase 14's real CUDA
qualification exposed one previously implicit hardware behavior, so the current
canonical document is IR version 3 and adds explicit float32 subnormal
flush-to-zero on both CPU and GPU. All other rules remain unchanged:

| Boundary | Required behavior |
|---|---|
| Expression arithmetic | float64, source-tree order |
| Completed feature storage | one cast to float32 |
| Coefficients | one cast to float32 |
| Utility accumulation | source-ordered float32 |
| Multiply and add | separate operations; no FMA contraction |
| Rounding | IEEE-754 round-to-nearest, ties-to-even |
| Fast math | disabled |
| Float32 subnormals | flush to signed zero (version 3) |
| NaN and infinity | preserved |

The evaluator rejects an unknown IR version, an incomplete or altered numeric
policy, a mismatched document hash, unresolved coefficient symbols, malformed
shapes, and non-finite coefficients. It does not silently select a nearby
implementation.

The canonical public MTC document contains 379 terms and 21 alternatives. Its
current version-3 IR hash is
`ebaba5a32bfffa90c28f7ff6245c0f7f47dbf714d69bf1a4736942c8143963d5`.
The generated file is
[`mtc21-strict-ir.json`](../benchmark-results/mtc21-strict-ir.json).

The Phase 13 reports remain immutable historical version-2 evidence; their
canonical IR hash was
`772586d120e317e0e8a8a55d8b1235d60f3d65b5a9b2745a5ac8df402150b4b7`.

## Implemented reference and diagnostic API

[`choiceforge.sharrow_ir`](../src/choiceforge/sharrow_ir.py) now provides:

- `evaluate_strict_cpu`, the normative term evaluator and utility oracle;
- `ordered_float32_utilities`, a separate multiply/add reference loop;
- `compare_strict_to_sharrow`, an exact term and alternative gate;
- first divergent row, term, expression, alternative, values, and stage;
- a diagnostic split between expression-policy differences and ordered-
  accumulation differences; and
- `write_comparison_report`, which emits deterministic, fail-closed JSON.

The ActivitySim destination adapter activates this comparison only when
`CHOICEFORGE_STRICT_CPU_BATCHES` is set. Observation mode writes evidence while
retaining Sharrow. `CHOICEFORGE_STRICT_CPU_REQUIRE_EXACT=1` converts any mismatch
into a hard gate failure for future backend qualification.

## Verification

The full repository suite passes:

```text
69 passed, 1 warning in 3.90 seconds
```

The warning is the existing Windows permission warning for pytest's optional
cache. It is not a test or implementation failure.

Tests now cover:

- canonical generation of all 379 public MTC terms and 21 alternatives;
- strict execution of every term and coefficient-resolved alternative;
- declared float64 expression, float32 feature, coefficient, and utility types;
- source-ordered utility accumulation against an independent scalar loop;
- exact-pass and deliberate-failure comparison reports;
- separation of expression and accumulation discrepancies;
- coefficient resolution and fail-closed missing-symbol behavior;
- tampered policy and document-hash rejection; and
- the existing CPU, CUDA, scheduling, destination, skim, and nested-logit suite.

## Real public ActivitySim comparison

Command:

```powershell
.\scripts\run_phase13_strict_cpu.ps1 -Households 1001 -RunTag real-gate
```

The run used public Prototype MTC Extended full geography and completed all 34
ActivitySim models in 95.511 seconds. The strict gate observed 30 real trip-mode
batches containing 85,126 rows. Every batch evaluated all 379 terms and all 21
alternatives.

| Measurement | Result |
|---|---:|
| Real batches | 30 |
| Rows | 85,126 |
| Feature cells compared | 32,262,754 |
| Exact feature cells | 32,234,288 (99.9118%) |
| Utility cells compared | 1,787,646 |
| Exact utility cells | 1,194,300 (66.8085%) |
| Maximum feature difference | 0.000030517578125 |
| Maximum utility difference | 0.25 |
| ActivitySim/Sharrow authority | preserved |

No real batch exactly matched the new strict policy. This is an expected and
useful result: current Sharrow is a production implementation being observed,
not the newly declared cross-device oracle. The gate classified all 28,466
different feature cells as expression-policy differences. Of 593,346 different
utility cells, 183 were fully explained by changed expression inputs and
593,163 also differed from the strict source-ordered multiply/add reduction.

The first feature divergence occurred in
`util_TNC_Single_Bridge_toll`: strict produced 97.6323013305664 and Sharrow
produced 97.63229370117188, a difference of 0.00000762939453125 after float32
feature storage. The first utility divergence was one float32 step for
`SHARED2FREE`: strict produced -1.1387739181518555 and Sharrow produced
-1.138773798942566.

These reports turn the earlier vague statement "CPU and GPU round differently"
into an executable contract and localized evidence. Phase 14 subsequently
matched the strict CPU arrays exactly without imitating whichever Sharrow
discrepancy happened to occur on this compiler and workstation.

The aggregate evidence is
[`phase13-strict-cpu-summary.json`](../benchmark-results/phase13-strict-cpu-summary.json).
The 30 individual reports are under
[`phase13-strict-cpu-real-gate`](../benchmark-results/phase13-strict-cpu-real-gate).
Their deterministic tree hash is
`108007aca296c5d65efe9bb7c1c0f6e7b84529d06ff34403852c61ae0ef11bb0`.

## Phase 13 success gate

Phase 13 succeeds because:

1. one explicit numeric policy now controls the CPU oracle;
2. every canonical MTC term and alternative executes under that policy;
3. an independent scalar accumulation test agrees exactly;
4. malformed or changed contracts fail closed;
5. real Sharrow batches are compared term by term and alternative by
   alternative;
6. every observed mismatch belongs to a reported semantic stage; and
7. ActivitySim completed normally with its original results authoritative.

Exact agreement with current Sharrow is not redefined as the Phase 13 gate.
Doing so would make Sharrow's current compiler behavior, rather than the
published numeric policy, the oracle. Exact cross-device agreement becomes the
Phase 14 gate when CUDA is generated from this same IR.

## Reproduction artifacts

- Runner: [`run_phase13_strict_cpu.ps1`](../scripts/run_phase13_strict_cpu.ps1)
- Summarizer: [`summarize_phase13_strict_cpu.py`](../scripts/summarize_phase13_strict_cpu.py)
- Canonical generator: [`generate_mtc21_strict_ir.py`](../scripts/generate_mtc21_strict_ir.py)
- Tests: [`test_sharrow_ir.py`](../tests/test_sharrow_ir.py)
- Canonical IR file SHA-256:
  `6d96df17d31a4547a22afc009744f1ea87367cfa6d4c87a05b25145a30f40d9c`
- Aggregate summary file SHA-256:
  `12135b240079cef8c55d7679f815e6d228e1630f78b1dc0f5aaa484ac7e03b66`

## Next gate

Phase 14 has now generated CUDA from the revised IR version 3 and achieved exact
array equality with `evaluate_strict_cpu` for synthetic edge cases, the
canonical 379-term model, and all 30 real public batches. See
[`phase14-strict-cuda-generator.md`](phase14-strict-cuda-generator.md). Current
Sharrow remains the ActivitySim fallback until the generated device path passes
the later repeated, byte-identical full-model performance gate.
