# Phase 41: exact shared arithmetic and guard-free resident sampling

## Outcome

Phase 41 removes Phase 40's last CPU arithmetic adjudication from the public
trip-destination sampler. A versioned compiler contract now emits the exact
float32 reduction schedule used by Sharrow's Numba/OpenBLAS evaluator and a
second contract reproduces NumPy 2.4.6's AVX2 float32 exponential and recursive
pairwise sum on CUDA. Utilities, weights, normalization, inverse-CDF choice,
and duplicate counting remain on the GPU. ActivitySim still owns and advances
its keyed random-number ledger.

The final three-pair 50,000-household qualification passed every proof gate.
All 30 programs covered 91,524 trip choosers, 1,454 alternatives, 133,075,896
utility cells, and 2,745,720 controlled draws per run. Every candidate used
zero CPU guard rows and zero fallback calls. All three independent output
verifiers found zero changed modeled decision cells.

## The arithmetic finding

Sharrow materializes 15 float32 features and calls
`np.dot(features, coefficients.reshape(15, 1), out=result)`. Numba lowers that
shape to SGEMV, not to a source-order scalar loop. On the qualified SciPy
OpenBLAS 0.3.30 Haswell path, the one-row SGEMV fallback evaluates three
left-associated groups of four products, then three scalar tail products:

```text
sum = (((p0 + p1) + p2) + p3)
sum += (((p4 + p5) + p6) + p7)
sum += (((p8 + p9) + p10) + p11)
sum += p12
sum += p13
sum += p14
```

A deterministic 200,000-row probe found zero bit mismatches for that schedule.
The previously tested sequential, tree, and lane schedules produced tens of
thousands of mismatches. The contract is named
`sharrow15-openblas-sgemv-group4-left-v1` and is hashed in every runtime event.

The full utility shadow then compared every one of the public workload's
133,075,896 CUDA utilities with live Sharrow and found:

| Utility qualification | Result |
|---|---:|
| cells compared | 133,075,896 |
| bit mismatches | 0 |
| maximum absolute difference | 0 |
| envelope violations | 0 |

This proof is what permits Phase 41 to set the utility error envelope to zero.

## Exact probability semantics

Exact utilities alone were insufficient. The first guard-free experiment used
CUDA `expf` and a sequential float32 total. It completed the model but changed
15 trip destinations and 31 final modeled cells. ActivitySim actually uses
NumPy's float32 exponential followed by NumPy's recursive pairwise sum with an
eight-lane base block and a 128-element recursion threshold.

Phase 41 ports NumPy 2.4.6's AVX2/FMA3 exponential polynomial and generates the
fixed 1,454-alternative pairwise tree at compile time. A synthetic qualification
over 5,000 complete rows (7,270,000 weights) produced zero bit mismatches in
the final exponential sum. The probability ABI is named
`numpy246-avx2-exp-pairwise128-v1` and also carries a stable hash.

The production CUDA chooser uses this exact total, float32 division, the
original ActivitySim float64 keyed draws, and the same preserved-order CDF
comparison. Phase 40's interval proof is bypassed only in Phase 41 because it
proves an error envelope that is no longer present. The actual choice
calculation and the independent final-output verifier remain authoritative.

## Final three-pair evidence

The source artifact is
[`phase41-p41final-summary.json`](../benchmark-results/phase41-p41final-summary.json).
The consolidated
[`phase41-p41final-qualification.json`](../benchmark-results/phase41-p41final-qualification.json)
checks the arithmetic probe, complete utility shadow, probability probe, three
candidate reports, and three exact verifiers. Each underlying artifact is
retained beside it.

| Complete public benchmark | Phase 40 median | Phase 41 median | Change |
|---|---:|---:|---:|
| all 34 model steps | 165.8 s | 158.3 s | 7.5 s saved; 1.047x |
| `trip_destination` | 23.1 s | 15.1 s | 8.0 s saved; 1.530x |
| pairs won | - | 3 of 3 | promotion gate passed |

The whole-model reduction is 4.52%; the trip-destination reduction is 34.63%.
The component savings are larger than the whole-model savings because unrelated
steps have normal run-to-run noise.

Across the three Phase 41 candidates, the direct resident boundary was stable:

| Direct boundary metric | Trial 1 | Trial 2 | Trial 3 |
|---|---:|---:|---:|
| utility kernel | 0.292 s | 0.283 s | 0.278 s |
| exact probability/choice kernel | 0.630 s | 0.627 s | 0.618 s |
| duplicate kernel | 0.048 s | 0.046 s | 0.034 s |
| former CPU guard section | 0.000119 s | 0.000119 s | 0.000127 s |
| complete sampling boundary | 1.723 s | 1.713 s | 1.653 s |

Phase 40's final clean boundary took 10.480 seconds, including 7.329 seconds in
the CPU guard. The Phase 41 median direct boundary is 1.713 seconds: about
6.12x faster, with the CPU arithmetic work eliminated rather than hidden.

## Replication result

All three output verifiers reported:

- zero changed modeled decision cells and zero changed decision rows;
- six non-trip output CSVs byte-for-byte identical;
- maximum `destination_logsum` difference `0.000012`, inside the `0.0001` gate;
- zero `mode_choice_logsum` difference; and
- overall success.

The diagnostic logsum is bounded rather than byte-identical because later GPU
trip-logsum phases retain their already qualified tolerance. Phase 41 itself
restores exact sampling choices and selected probabilities for the reviewed
contract.

## Assumptions and claim boundary

The qualified environment is an NVIDIA RTX A4000 with 16 GB, CuPy 14.1.1,
NumPy 2.4.6, SciPy 1.17.1, Numba 0.66.0, and the public Prototype MTC extended
1,454-zone/50,000-household workload. The utility ABI is tied to the reviewed
15-expression specification and the SciPy OpenBLAS 0.3.30 Haswell SGEMV
schedule. The probability ABI is tied to NumPy 2.4.6's AVX2/FMA3 float32 path.

Unsupported specifications, alternative counts, estimation mode, sample
sizes, or layouts fail closed. There is no general claim that every OpenBLAS,
NumPy, CPU architecture, or travel model uses these same schedules. Portability
requires generating and qualifying a matching ABI for that environment.

## Larger significance and next work

Phase 41 validates the central design proposed after Phase 40: identical
results do not require moving ambiguous rows back to the CPU if both backends
share explicit arithmetic semantics. The compiler contract turns implicit
library behavior—dot association, exponential implementation, and sum tree—
into versioned, reviewable source.

The next major opportunity is to generalize this mechanism beyond one 15-term
sampling specification and one fixed alternative count. A credible upstream
Sharrow backend should extract expression IR, discover or select a declared
numeric policy, generate CPU and CUDA code from it, cache the ABI by hash, and
run mutation tests for coefficients, skims, land use, chooser rows, and random
draws. Phase 41 is the working end-to-end reference implementation and proof
that this approach can produce both replication and a material whole-model win.
