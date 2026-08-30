# Phase 42: general numeric compiler and compact cached trip runtime

## Outcome

Phase 42 turns Phase 41's successful one-shape arithmetic recipe into a
validated numeric-policy compiler. A declarative policy describes the number
of utility terms, their exact float32 addition order, the exponential
implementation, and the probability-sum tree. The same policy generates the
CPU reference evaluator and CUDA source. Its SHA-256 identity changes if any
arithmetic choice changes, so incompatible compiled code cannot be silently
reused.

The production trip-destination runtime also removes repeated host work. It
passes one compact directional view instead of materializing a complete
outbound/inbound pandas table, compiles the ten purpose contracts and native
ABIs once, resolves ten simulation specifications once, then reuses each for
trip numbers two and three. All validation and ActivitySim's keyed random
ledger remain in force.

Three fresh matched Phase 41/42 pairs on the public 50,000-household model all
favored Phase 42 and all passed independent exact-output verification. Median
`trip_destination` time fell from 14.9 to 10.7 seconds, a 4.2-second or 28.19%
reduction and **1.393x speedup**. Median time for all 34 model steps fell from
156.7 to 151.8 seconds, a 4.9-second or 3.13% reduction and **1.032x speedup**.

## What is general now

`Float32ReductionPolicy` validates that every source-order term appears exactly
once. `grouped_left_reduction` and `ordered_left_reduction` create explicit
association schedules for arbitrary positive term counts. `reduce_float32`
is the reference evaluator; `float32_reduction_cuda` emits the corresponding
CUDA statements.

`Float32ProbabilityPolicy` generates NumPy 2.4.6-compatible float32
exponentiation and recursive pairwise sums for arbitrary positive alternative
counts. `NumericPolicyCompiler` combines the utility and probability policies
under `choiceforge-numeric-policy-compiler-v1`. The qualified production ABI
is:

```text
a995581add26a19967491f1be988ec5ef824df9a1a2f1084fe9106c07a47713d
```

The compiler still deliberately fails closed. It accepts only a fully covered,
source-ordered float32 reduction, disables contracted FMA for utility
arithmetic, and supports only the reviewed NumPy AVX2/FMA3 exponential and
128-value pairwise policy. It does not infer unknown CPU-library behavior.

## Independent compiler probe

[`phase42-numeric-compiler-probe.json`](../benchmark-results/phase42-numeric-compiler-probe.json)
compiled and executed twelve shapes on the RTX A4000. GPU outputs were compared
by their float32 bit patterns with the generated CPU/NumPy references.

| Probe family | Shapes | Rows per shape | Bit mismatches |
|---|---|---:|---:|
| utility reduction | 1, 3, 5, 15, 17, 31 terms | 257 | 0 |
| probability total | 1, 7, 8, 129, 257, 1,454 alternatives | 257 | 0 |

Changing the 15-term grouped reduction to ordered-left produced a different
ABI hash. Changing 1,454 alternatives to 1,453 also produced a different hash.
This proves both execution across varying shapes and cache-key sensitivity to
two semantic mutations; it is not a claim that every possible model shape has
already been qualified.

## Compact and cached runtime boundaries

The old direction-expansion boundary duplicated a very large pandas table.
`Phase42DirectionalFrame` instead carries the base sampled table once plus two
coordinate arrays. The native logsum boundary verifies that rows are stable
within each trip, that each direction supplies exactly three controlled draws,
and that those draws agree where the directional state requires them to agree.
It then constructs only the small representative-state packet needed by CUDA.

Each candidate report proves the same complete call pattern:

| Runtime contract | First-use work | Reuses | Total uses |
|---|---:|---:|---:|
| purpose/logsum contract | 10 | 20 | 30 |
| strict expression IR | 10 | 20 | 30 |
| hash-addressed native ABI codegen | 10 | 20 | 30 |
| final simulation specification | 10 | 20 | 30 |
| compact directional bundle | - | - | 30 |

The full workload contains three trip-destination calls, 30 purpose programs,
91,524 trip rows, and 2,094,156 retained sample rows. The caches are scoped to
one model process; no result value or random draw is cached.

## Final qualification evidence

The primary source is
[`phase42-p42final-summary.json`](../benchmark-results/phase42-p42final-summary.json).
[`phase42-p42final-qualification.json`](../benchmark-results/phase42-p42final-qualification.json)
rechecks the compiler probe, all three candidate proof-gate sets, ABI identity,
cache counts, complete workload shape, three output verifiers, and three
pairwise timing wins.

| Trial | Phase 41 all model | Phase 42 all model | Saved | Exact decisions |
|---|---:|---:|---:|---:|
| 1 | 155.7 s | 153.7 s | 2.0 s | yes |
| 2 | 157.2 s | 151.5 s | 5.7 s | yes |
| 3 | 156.7 s | 151.8 s | 4.9 s | yes |
| median | 156.7 s | 151.8 s | 4.9 s | 3 of 3 |

The median Phase 42 directly instrumented trip boundary was 9.312 seconds:

| Stage | Median time |
|---|---:|
| full-zone GPU sampling | 1.674 s |
| compact preparation | 1.540 s |
| preprocessor | 1.045 s |
| native trip logsums | 3.165 s |
| final simulation | 1.812 s |
| complete instrumented boundary | 9.312 s |

The ActivitySim component timer is 10.7 seconds because it includes surrounding
framework work outside this direct boundary.

## Replication result

All three independent verifiers reported zero changed decision cells and zero
changed decision rows. In this final qualification, maximum
`destination_logsum` and `mode_choice_logsum` differences were also zero in all
three pairs. Every Phase 41 proof gate remained active: all 133,075,896
full-zone sampling utilities stayed in the exact resident CUDA path, all
2,745,720 keyed draws remained owned by ActivitySim, and fallback stayed zero.

## Targets, assumptions, and claim boundary

The phase began with two stretch goals: `trip_destination` below 8 seconds and
the whole model below 150 seconds. The qualified medians are 10.7 and 151.8
seconds, so neither stretch goal is reported as met. They were optimization
targets, not proof gates, and the evidence file preserves both `false` values.
The phase nevertheless earns promotion because it wins all three pairs,
remains exact, materially improves its target component, and makes the
arithmetic machinery reusable.

The qualification applies to this NVIDIA RTX A4000, CuPy/NumPy/SciPy/Numba
environment and the public Prototype MTC extended 1,454-zone,
50,000-household benchmark. The generalized compiler can emit other term and
alternative counts, but a new CPU architecture, NumPy exponential path,
OpenBLAS schedule, expression kind, dtype, or model layout must select and
qualify the corresponding policy. Phase 42 is not yet an upstream Sharrow
backend and does not eliminate CPU orchestration, pandas tables, file I/O, or
the many ActivitySim components not migrated to CUDA.

## Larger significance and next major phase

Phase 41 proved that identical arithmetic can remove a CPU safety pass. Phase
42 proves that the mechanism can be data-driven, hash-addressed, tested across
multiple shapes, and embedded in a faster end-to-end runtime. This is the
credible bridge from a special kernel to a dedicated expression compiler.

The next major phase should compile the remaining sparse final-simulation and
preprocessor work from the same IR, retain chosen probabilities/logsums on the
device, and replace the remaining pandas group/merge boundary with compact
indexed arrays. That targets the measured 1.045-second preprocessor,
1.812-second simulation, and parts of the 1.540-second preparation stage. A
successful promotion must again use fresh matched pairs, exact complete-output
verification, versioned contracts, and fail-closed unsupported cases.
