# RFC: strict typed IR and GPU backend for Sharrow utility flows

## Decision

Propose a shared, typed expression IR as the only semantic input to future
strict CPU and CUDA utility evaluators.  This prevents CPU and GPU backends
from independently interpreting ActivitySim expressions.

## Motivation

Sharrow currently compiles expressions with Numba and normally enables
`fastmath`. That is excellent for speed, but permits floating-point
transformations that are not a portable numerical contract. The current Phase
12 shadow harness demonstrated the consequence: a semantically equivalent GPU
evaluator can differ in the last float32 bits before a utility sum.

## IR contract

`choiceforge.sharrow_ir` is a prototype data-only IR. Every term records:

- its source-order position and label;
- a typed operation tree (`skim`, `column`, arithmetic, compare, `clip`, or
  `maximum`);
- the three skim directions explicitly; and
- coefficients by alternative.

The canonical JSON hash identifies both the specification and numeric policy.
The implemented IR version 3 policy is: float64 expression arithmetic, cast
each completed feature and coefficient to float32, then source-ordered float32
utility accumulation using separate multiply and add operations, IEEE
round-to-nearest-even, no FMA contraction, fastmath disabled, and explicit
float32 subnormal flush-to-zero on both targets. NaN and infinity are preserved.

## Proposed upstream work

1. Move the prototype IR into Sharrow and generate it from the same expression
   rewrite pass used by `DataTree.setup_flow`.
2. Add a strict CPU evaluator from that IR. It is the cross-device reference;
   existing Numba fastmath flows remain the performance mode.
3. Add a CUDA target that emits one fused kernel for directional skim gathers,
   term evaluation, source-ordered utility accumulation, and optional nested
   logsum handoff.
4. Add golden tests with term values, utility values, numeric-policy hash, and
   final logsums. GPU enablement requires exact strict-CPU equality, not merely
   close probabilities.
5. Keep ActivitySim's current Sharrow path as fallback until the public
   Prototype MTC suite and repeated full-model byte-identical gate pass.

## Why CUDA C++/NVRTC rather than a separate Numba GPU rewrite

The backend needs explicit arithmetic controls, inspectable generated source,
and a stable ABI for compiled kernels. CUDA C++/NVRTC provides these directly.
Numba's built-in CUDA target is deprecated in favor of a separate
`numba-cuda` package, creating an additional moving dependency for Sharrow.

## Phase 13 delivered

The strict CPU reference, ordered accumulator, exact comparison gate, and
first-divergence report are implemented. All 379 public MTC terms and 21
alternatives execute under the declared policy. A public 1,001-household run
compared 30 real Sharrow batches and 85,126 rows while leaving Sharrow
authoritative. The reports classify current Sharrow differences by expression
and accumulation stage instead of treating current compiler behavior as the
cross-device oracle. See [`phase13-strict-cpu-reference.md`](phase13-strict-cpu-reference.md).

## Phase 14 delivered

The CUDA C++ target is now generated from IR version 3 and cached by the full
semantic inputs. It passed every supported operation and numeric-edge test, the
canonical 379-term by 21-alternative utility, and all 30 real public batches:
32,262,754 exact feature cells and 1,787,646 exact utility cells across 85,126
rows. The generated device utility can feed the MTC nested-logsum reducer with
no utility round trip through host memory.

The generated output remains a shadow while Sharrow is authoritative. Upstream
qualification next requires a production call-site, zero shadow mismatches, and
repeated byte-identical full-model A/B performance trials. See
[`phase14-strict-cuda-generator.md`](phase14-strict-cuda-generator.md).
