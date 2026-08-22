# Phase 14: strict CUDA generation and exact cross-device gate

## Outcome

Phase 14 is complete. ChoiceForge now generates a CUDA C++ utility kernel
directly from the same strict intermediate representation (IR) used by the CPU
answer key. The generated GPU evaluator and strict CPU evaluator agree exactly
on synthetic edge cases, the canonical public MTC utility specification, and
all 30 trip-mode batches captured during a real public ActivitySim run.

This is a correctness and integration result, not a new production speed
claim. ActivitySim/Sharrow remained authoritative throughout the public run;
generated GPU values were observed in shadow mode and could not change choices,
random streams, logsums, or final model outputs.

## One arithmetic contract

CUDA runtime compilation exposed one portability issue that Phase 13's CPU-only
work could not reveal: the deployed NVIDIA cubin toolchain flushes float32
subnormal values to signed zero. Rather than ignore that edge case, Phase 14
makes it part of the public contract. IR version 3 records:

| Boundary | Required behavior |
|---|---|
| Expression arithmetic | float64, source-tree order |
| Completed feature storage | one cast to float32 |
| Coefficients | one cast to float32 |
| Utility accumulation | source-ordered float32 |
| Multiply and add | separate operations; no FMA contraction |
| Rounding | round-to-nearest, ties-to-even |
| Fast math | disabled |
| Float32 subnormals | flush to signed zero |
| NaN and infinity | preserved |

The CPU evaluator explicitly applies the same float32 flush-to-zero rule after
feature and coefficient casts and after every ordered multiply and add. CUDA is
compiled with `--ftz=true`, `--fmad=false`, and `--prec-div=true`. Tests include
subnormal values plus NaN and infinity, so the rule is exercised rather than
merely documented.

The current canonical MTC IR contains 379 terms and 21 alternatives. Its
version-3 document hash is
`ebaba5a32bfffa90c28f7ff6245c0f7f47dbf714d69bf1a4736942c8143963d5`.

## Generated kernel design

[`choiceforge.sharrow_cuda`](../src/choiceforge/sharrow_cuda.py) emits CUDA C++
from the IR operation trees instead of maintaining a second handwritten
expression implementation. The generated kernel uses:

- typed packs for float64, int64, and boolean inputs;
- one CUDA block per model row;
- thread 0 to evaluate all 379 terms once into block shared memory;
- one thread per alternative to accumulate its 379 coefficients in original
  source order using explicit round-to-nearest multiply and add operations; and
- a device-return option that hands the 21 generated utilities directly to the
  existing MTC nested-logsum reducer.

The device-resident pipeline test confirms zero utility device-to-host copies
and zero reducer host-to-device copies at that handoff. Feature capture and
downloads remain enabled in the real Phase 14 qualification run because exact
term-by-term evidence is more important than performance during this gate.

Compilation is cached by the strict IR hash, typed input schema, coefficient
payload, generated source, and relevant kernel options. A changed expression,
numeric policy, type, or coefficient set therefore cannot accidentally reuse an
incompatible kernel.

## Verification

The complete repository suite passes:

```text
69 passed, 1 warning in 3.90 seconds
```

The warning is Windows denying pytest's optional cache directory; it is not a
test or implementation failure. The five new CUDA tests cover every supported
IR operation, NaN/infinity/subnormal behavior, cache reuse, all 379 canonical
terms and 21 alternatives, and the device-resident nested-logsum handoff.

## Real public ActivitySim gate

Command:

```powershell
.\scripts\run_phase14_strict_cuda.ps1 -Households 1001 -RunTag real-gate
```

The public Prototype MTC Extended full-geography model completed all 34 model
steps in 101.78 seconds. The gate observed 30 real trip-mode batches containing
85,126 rows:

| Measurement | Result |
|---|---:|
| Exact batches | 30 / 30 |
| Rows | 85,126 |
| Exact feature cells | 32,262,754 / 32,262,754 |
| Exact utility cells | 1,787,646 / 1,787,646 |
| Maximum feature absolute difference | 0.0 |
| Maximum utility absolute difference | 0.0 |
| Terms per batch | 379 |
| Alternatives per batch | 21 |
| ActivitySim/Sharrow authority | preserved |

Every generated feature and utility bit matched the strict CPU answer key.
This closes Phase 14's exact cross-device correctness gate.

The qualification reports also record median diagnostic times of 116.157 ms
for host input packing/transfer, 5.868 ms for the generated kernel, and 1.593 ms
for result download. These are deliberately **not** a speed claim: observation
mode materializes and downloads the entire 379-column feature matrix, and the
comparison timing boundary differs from production Sharrow. A fair performance
claim requires removing qualification downloads, integrating the device path,
and repeating full-model interleaved A/B trials.

## Replication artifacts and hashes

- CUDA generator: [`sharrow_cuda.py`](../src/choiceforge/sharrow_cuda.py)
- ActivitySim shadow gate: [`activitysim_destination.py`](../src/choiceforge/activitysim_destination.py)
- Runner: [`run_phase14_strict_cuda.ps1`](../scripts/run_phase14_strict_cuda.ps1)
- Summarizer: [`summarize_phase14_strict_cuda.py`](../scripts/summarize_phase14_strict_cuda.py)
- Tests: [`test_sharrow_cuda.py`](../tests/test_sharrow_cuda.py)
- Canonical IR: [`mtc21-strict-ir.json`](../benchmark-results/mtc21-strict-ir.json)
- Aggregate evidence: [`phase14-strict-cuda-summary.json`](../benchmark-results/phase14-strict-cuda-summary.json)
- Individual reports: [`phase14-strict-cuda-real-gate`](../benchmark-results/phase14-strict-cuda-real-gate)
- Canonical IR file SHA-256:
  `6d96df17d31a4547a22afc009744f1ea87367cfa6d4c87a05b25145a30f40d9c`
- Generated CUDA source SHA-256:
  `63d16a939dd62c9c8da2ae0d8cbd0ee9bfd5bfaa828932ee32820460c65b2fa4`
- Report-tree SHA-256:
  `6b84ae838c67b2d4077321124a6a1ff3e6d7ee5fa6375908f5a32a18b68f80e2`
- Aggregate summary file SHA-256:
  `8335f880ff0fdfbbbd81debd00995c340fa52d0ed0c4005067bfa983d33b8152`

## Phase 14 success gate

Phase 14 succeeds because one hashed IR now generates both comparison targets,
all supported operations and numeric edge cases pass exactly, the full
379-by-21 public utility passes exactly, all 30 real public batches pass exactly,
and the experimental output remains safely non-authoritative.

The next major gate is production integration: eliminate qualification-only
feature downloads and repeated host packing, use generated device utilities in
the real nested-logsum/choice path, require zero shadow mismatches, then run the
same repeated byte-identical full-model A/B protocol used in Phase 11.
