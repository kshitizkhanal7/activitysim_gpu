# Phase 43: compact controlled-random trip state

## Outcome

Phase 43 removes the largest remaining avoidable CPU cost inside final trip
destination simulation without changing a model equation or random number.
ActivitySim previously asked its keyed random-number manager for the same
trip-level state through 30 purpose calls and expanded six normal draws across
2,094,156 sampled-destination rows. Phase 43 advances the same per-trip random
channels in three trip-number batches, stores only unique-trip values, and
feeds the final uniform choice draw directly into ActivitySim's unchanged
`choice_maker` arithmetic.

Three fresh matched Phase 42/43 pairs ran the complete public 50,000-household,
1,454-zone, 34-step model. The directly instrumented trip boundary improved in
all three pairs. Its median fell from 9.261 to 8.526 seconds (**1.086x**), while
final simulation fell from 1.817 to 1.152 seconds (**1.578x**). ActivitySim's
published `trip_destination` timer fell from 10.6 to 9.9 seconds, a 6.60%
reduction and **1.071x speedup**. All seven final output tables were
byte-identical in all three pairs.

The complete-model median moved only from 152.4 to 152.1 seconds (**1.002x**).
Two individual whole-model candidate runs were slower because unrelated model
steps varied by more than the 0.7-second component gain. Phase 43 therefore
claims a replicated component and boundary improvement, not a statistically
resolved whole-model speedup.

## What the profile found

A dedicated Phase 43 profile decomposed the 1.831-second final-simulation
boundary:

| Work | Time |
|---|---:|
| interaction simulation | 1.751 s |
| Sharrow utility evaluation inside it | 0.584 s |
| probability conversion | 0.022 s |
| keyed random lookup and choice | 0.722 s |
| surrounding final-simulation work | 0.080 s |

The keyed random lookup was the largest removable sub-boundary. The accepted
implementation reduces its measured choice section to a median 0.046 seconds
across all 30 calls. Utility calculation and probability arithmetic remain
unchanged and authoritative.

## Compact random ledger

The full workload contains 91,524 intermediate trips and 2,094,156 retained
sampled destinations. Each trip belongs to one purpose within a trip-number
batch. Phase 43 verifies that sampled alternatives are contiguous and ordered
exactly like their chooser trips and that purpose trip IDs do not overlap.
Only then does it concatenate the ten disjoint purpose indexes.

For each of the three trip-number batches it makes exactly three ActivitySim
RNG calls:

1. three OD normal draws for every unique trip;
2. three DP normal draws for every unique trip; and
3. one final uniform choice draw for every unique trip.

The complete run therefore makes nine batched calls. It retains 183,048
directional trip rows, avoids 4,005,264 repeated directional sample rows, and
stores 91,524 final-choice draws. All 91,524 choice draws are consumed exactly
once. The random manager is still ActivitySim's; ChoiceForge does not invent a
seed, generator, or alternate sequence.

## Exact semantic adapter

Phase 43 does not replace logit arithmetic. Its adapter preserves ActivitySim's
probability-sum validation, bad-choice reporting, `choice_maker`, pandas index
and result types, and `-99` failure marker. It replaces only
`random_for_df(probs)` with the already-generated uniform values for the exact
active trip index. If the index does not match, the original ActivitySim
function runs.

The native logsum ABI gained a separate compact-draw shape check. Phase 42
packets must still contain expanded directional draws; Phase 43 packets must
contain exactly `2 * unique_trip_rows` by three values. Unsupported ordering,
duplicate trip IDs, inconsistent channel names, unexpected draw shapes, or a
different preprocessor draw count fail before modeled results are accepted.

## Qualification evidence

[`phase43-p43final-qualification.json`](../benchmark-results/phase43-p43final-qualification.json)
recomputes the promotion gates from the three baseline reports, three candidate
reports, three independent exact verifiers, and the matched-pair summary.

| Pair | Phase 42 direct boundary | Phase 43 direct boundary | Final simulation Phase 42 → 43 | Exact outputs |
|---:|---:|---:|---:|---:|
| 1 | 9.481 s | 8.642 s | 1.809 → 1.158 s | 7 of 7 byte-identical |
| 2 | 9.261 s | 8.526 s | 1.817 → 1.152 s | 7 of 7 byte-identical |
| 3 | 9.200 s | 8.378 s | 1.826 → 1.149 s | 7 of 7 byte-identical |
| median | 9.261 s | 8.526 s | 1.817 → 1.152 s | 3 of 3 verifiers pass |

Every candidate report also proves 30 compact-choice calls, zero fallback,
the complete 91,524-trip/2,094,156-sample workload, nine RNG calls, and all
earlier Phase 42 compiler and resident-CUDA gates. Maximum destination and mode
logsum differences were zero. The full Python suite passes 179 tests.

## Assumptions and claim boundary

The equivalence relies on ActivitySim's keyed random channels: advancing a
unique trip ID once in a batched frame is equivalent to advancing that same ID
inside its only purpose frame. Phase 43 validates the disjoint-ID and ordering
preconditions rather than assuming them. A model where the same trip appears
in multiple purpose bundles must use a different ledger contract.

The measured result applies to the RTX A4000, current ActivitySim/Sharrow and
NumPy stack, and this public benchmark. The optimization is mostly CPU
orchestration around GPU-resident trip logsums; it is not a claim that a new
GPU kernel alone produced the 1.578x final-simulation result. It strengthens
the GPU runtime by removing redundant host work at its random boundary.

## Next major opportunity

The remaining final simulation spends about 0.58 seconds evaluating sparse
utilities and roughly 0.5 seconds constructing, padding, grouping, and mapping
interaction tables. The next ambitious phase should compile the 14 final
destination utility terms from the shared IR, operate on compact ragged
offsets, and fuse exact probability/choice on the device or in one normalized
array service. It must shadow ActivitySim first, fingerprint the arithmetic
policy, preserve keyed draws, fail closed for unsupported specifications, and
repeat complete-output matched qualification before promotion.
