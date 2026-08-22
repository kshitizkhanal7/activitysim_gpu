# ChoiceForge design and correctness contract

## Objective

ChoiceForge is an experimental GPU execution backend for the repeated discrete-
choice operations in activity-based travel demand models. The first milestone
targeted fixed alternative sets. Phase 2 added ragged real ActivitySim
scheduling replay, Phase 3 added a validated compact-expression compiler,
Phase 4 integrated that compiler into mandatory scheduling, and Phase 5
generalized it across four tour-scheduling components.

Phase 6 adds a segmented ragged destination kernel that selects coefficients
by purpose segment and executes all captured segments in one CUDA launch. It
also adds a separate combined-direction ActivitySim logsum backend. The kernel
belongs at a sufficiently large packed boundary; the live prototype uses the
logsum optimization because purpose-by-purpose control flow exposes GPU jobs
below the measured crossover.

The fused-linear operation is

```text
chooser features × alternative coefficients + constants
    -> availability mask
    -> stable log-sum-exp
    -> inverse-CDF Monte Carlo choice
```

The CUDA kernel does not materialize the chooser-by-alternative utility matrix
in global device memory.

## Reproducibility boundary

ChoiceForge never generates random numbers inside a kernel. The caller supplies
one uniform draw per chooser in `[0, 1)`. An ActivitySim integration must obtain
these draws from ActivitySim's random-number manager. This preserves stream
ownership, household stability, and the ability to compare CPU and GPU choices
using identical draws.

Alternatives are evaluated in input column order. Choice is the first cumulative
weight greater than or equal to `uniform * total_weight`, matching ActivitySim's
subtract-until-nonpositive boundary rule. A missed probability row falls back to
the first maximum-probability alternative, also matching ActivitySim.

## Numerical behavior

All milestone-one inputs and calculations use IEEE float32. Logsum-exp is
stabilized by subtracting the largest available utility. Unavailable or non-
finite utilities receive zero probability. A row with no finite, available
alternative produces choice `-1` and logsum `-inf`.

GPU reductions and CPU BLAS may accumulate values in different orders. The
validation policy is therefore:

- choices must match exactly on benchmark data;
- logsums use explicit absolute and relative tolerances;
- every mismatch is counted rather than silently discarded;
- random draws close to a probability boundary are retained as adversarial
  correctness cases, not removed from reported benchmarks.

## Current kernel organization

One CUDA block processes one chooser. Threads represent alternatives, padded to
the next power of two. Shared memory contains one utility/weight row plus one
reduction buffer. The implementation supports 1--1,024 alternatives.

This organization is appropriate for fixed alternatives and medium-sized
sampled choice sets. Very large destination sets require a tiled streaming
kernel with online logsum accumulation; that is milestone two.

The Phase 3 scheduling kernel also uses one block per chooser, but CSR-style
offsets define ragged feasible-alternative sets. Generated scalar expressions
load chooser, alternative, and row-varying values from a compact ABI. The
kernel embeds coefficients and fuses expression evaluation, stable logsum-exp,
and choice.

The accepted grammar supports numeric constants, named columns, arithmetic,
comparisons, Boolean conjunction/disjunction, and unary negation. Calls,
attributes, unknown names, and unsupported syntax are rejected at compile time.

A block-wide synchronization barrier separates maximum reduction from scratch
buffer reuse. This is required when an alternative set spans multiple warps;
regression tests cover 33 and 190 alternatives because a 32-alternative test
cannot expose cross-warp races.

Phase 4 adds a pandas-facing packer and explicit ActivitySim dispatch. Its
timetable optimizer exploits a structural fact: the public MTC model has only
21 daily periods per chooser. It calculates timetable state once on the small
chooser-by-period grid and gathers seven required row values, instead of
calling generic timetable functions for every interaction row.

Phase 5 makes the lowerer component-neutral. It extracts simple string-category
assignments as compact chooser columns, supports equality and inequality, and
compiles ordinary dataframe arithmetic after removing ActivitySim's `df.`
namespace. Timetable optimization accepts any supported subset of primitive
calls and infers whether a timetable row is owned by `person_id`, `tour_id`, or
another explicitly named chooser column. This supports mandatory, joint, and
non-mandatory scheduling without component-specific branches. At-work capped
time expressions use the safe generic primitive evaluator around the same
compiled kernel contract.

## Honest limitations

- This is a configured backend for four tour-scheduling components, not a replacement for
  ActivitySim's complete `interaction_simulate` implementation.
- Phase 3 supports the expressions required by MTC mandatory scheduling, not
  arbitrary ActivitySim expressions.
- Input conversion and transfer can dominate small workloads.
- Stateful timetable primitives are vectorized on the CPU, and mode-choice
  logsums are still calculated upstream with Sharrow.
- The Phase 7 nested-logit kernel is FP64 and specific to the canonical MTC
  21-mode topology; arbitrary nest trees are not yet compiled.
- GPU tracing and estimation mode are not implemented. Destination batching
  rejects estimation and three-zone models before consuming random draws.

## Next architecture milestone

1. Add tiled online logsum-exp for large destination-choice alternative sets.
2. Keep skims and encoded chooser columns resident on the GPU across model
   components.
3. Add exact sampled-alternative correction terms and ActivitySim labels.
4. Expand tracing, estimation, expression, and nested-logit support.
5. Reproduce on a different public model, operating system, and GPU generation.

## Phase 7 destination-logsum architecture

The original destination loop repeated the same 70-expression preprocessor for
each of ten trip purposes and each of three trip numbers. Phase 7 preserves
purpose-specific sampling and simulation but stacks all OD and DP chooser rows
for one trip number. The preprocessor therefore runs three times instead of 30.
ActivitySim-owned OD and DP random draws retain their original order.

After Sharrow evaluates the 404-row, 21-alternative mode-choice specification,
a fused FP64 CUDA kernel reduces the fixed MTC nest in one launch. The boundary
includes host-to-device transfer and result transfer. It validates alternative
order, topology, and coefficients; any CUDA failure reduces the already-
evaluated utility matrix on ActivitySim's CPU path, without re-evaluating
expressions or consuming random numbers again.

Purpose batching is deliberately conservative. Before sampling, it rejects
estimation, three-zone path building, multiple preprocessors, and any
preprocessor expression that references a coefficient whose value varies by
purpose. Only this preflight exception triggers the original per-purpose path.

## Phase 8 current-version and scale architecture

Phase 8 preserves the same narrow dispatch while moving from ActivitySim 1.4
to pinned current commit `16ab11180a26912987eb902daf945e268f3efc11`. A new
ActivitySim alternatives context is explicitly forwarded to the fallback path;
ChoiceForge does not attempt to emulate explicit-error-term behavior. LOS
compatibility accepts both the older exported `THREE_ZONE` constant and its
historical numeric value without weakening the preflight rejection.

The 50,000-household public workflow exposes the next bottleneck clearly. Each
optimized run schedules 21,045,645 feasible alternative rows in 16 GPU calls.
The largest call has 9,561,750 rows, 50,325 choosers, and a 327.920 MB compact
input. Its measured CUDA/backend portion is about 0.4 seconds, while complete
lowering, stateful timetable evaluation, packing, ActivitySim random-number
retrieval, GPU work, and result mapping take about 3.3 seconds. CPU-side
stateful preparation is now more important than kernel arithmetic.

Trip destination processes three real trip-number batches containing
2,381,612, 605,782, and 198,736 OD+DP logsum rows. The design keeps sampling
and final simulation purpose-specific, so ActivitySim owns choice-set
construction, random streams, and selected labels. The acceleration comes from
sharing invariant preprocessor work and replacing the validated fixed nest
reduction, not from changing behavioral rules.

The most valuable next engineering targets are therefore persistent encoded
timetable inputs, faster keyed-random retrieval, and broader compiled
preprocessors. A more elaborate CUDA choice kernel alone cannot remove those
host costs. General claims still require a model with different specifications,
independent reproduction, and additional GPU hardware.
