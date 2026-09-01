# Phase 48: resident destination probability graph

## Result

Phase 48 replaces the last compact CPU normalization boundary inside the five
sampled-destination families with a fail-closed resident CUDA graph. The graph
computes the reviewed utility, NumPy-compatible float32 exponential weights,
pairwise row totals, probabilities, one-draw choice, and compact logsum inputs.
It also continues the keyed MT19937 state left on the device by destination
sampling instead of rebuilding the final random stream from seeds.

The public 50,000-household Prototype MTC extended workload contains 19 calls,
201,390 choosers, and 4,696,676 sampled alternatives. Three fresh Phase 47/48
matched pairs produced seven byte-identical published CSV files and zero
changed decision cells in every pair.

| Qualified measurement | Phase 47 | Phase 48 | Result |
|---|---:|---:|---:|
| resident final boundary, median | 0.4002 s | 0.3182 s | 1.258x; 20.5% lower |
| direct boundary pairs won | - | 3 of 3 | replicated |
| five target components, median aggregate | 28.1 s | 27.9 s | timing resolution/noise |
| complete 34-step lifecycle, median | 147.2 s | 144.717 s | 1.017x observed |
| lifecycle pairs won | - | 2 of 3 | not a replicated claim |
| changed decision cells | - | 0, 0, 0 | exact |

The boundary savings were 0.0363, 0.0934, and 0.0568 seconds. The lifecycle
changes were -0.617, +2.982, and +2.483 seconds. A localized 0.082-second
median boundary saving is smaller than whole-model timing noise, so Phase 48
does **not** claim replicated lifecycle superiority. Its promoted performance
claim is the directly instrumented 19-call boundary.

## What became device resident

For every final sampled-choice row, the runtime now performs this chain:

1. evaluate the reviewed destination expression into resident float32 utility;
2. subtract the row maximum and calculate NumPy-compatible exponential weight;
3. add weights with NumPy's measured pairwise float32 reduction order;
4. divide each weight by its total and walk the cumulative probability line;
5. compare the line with the exact ActivitySim float64 random ticket;
6. return the selected compact position, selected probability, row total, and
   only the sparse guard information needed by orchestration; and
7. calculate the official logsum with the GPU total plus NumPy's scalar log.

The scalar `log(total)` remains on the CPU because CUDA's logarithm is not
bit-identical to this NumPy build. This transfers one float32 total per chooser,
not the dense utility matrix. Production avoids 23,759,764 bytes of dense
utility download and transfers 2,618,966 bytes of compact results across all
19 calls.

ActivitySim still owns orchestration, tables, model sequencing, and the random
ledger. Phase 48 is a backend boundary, not a separate travel model.

## Exact random-state continuation

ActivitySim assigns random-number streams by stable chooser identity. Phase 46
generated the 30 sampling draws on GPU and retained each MT19937 state. Final
choice can reorder the chooser rows, so Phase 48 builds an identity permutation
from final rows back to the stored state rows.

ActivitySim's intervening logsum work advances each relevant ledger offset by
six draws. The resume kernel therefore advances the retained state by exactly
six values and emits the next value. Resume is permitted only when:

- the final chooser set is exactly the stored chooser set;
- every identity maps to one stored state;
- every observed offset delta is the same non-negative number; and
- that delta is at most the reviewed safety limit of 4,096 draws.

Otherwise the state-resident path is rejected. All three production runs and
the exhaustive shadow recorded 19 resume hits and zero misses. Focused tests
compare the resumed float64 bits with independently reseeded NumPy
`RandomState`, including a permuted chooser order and the six skipped draws.

## A versioned fail-closed backend contract

The backend identifies itself as `choiceforge.cuda.destination_graph`, version
1. It publishes `require` and `test` modes, four reviewed compact widths (21,
25, 29, and 30), float32 utility semantics, float64 keyed MT19937 semantics,
the supported exponential domain, and hashes for both the expression ABI and
exponential correction table.

Unknown programs, widths, arithmetic domains, random-ledger relationships, or
bad row totals stop the fast path. They do not silently substitute a similar
calculation. This follows the useful upstream pattern in ActivitySim's Sharrow
integration: required mode treats unsupported compilation as an error, while
test mode can run both implementations for comparison. Sharrow also exposes a
compiled reusable Flow over a DataTree, which is the architectural precedent
for a reusable device backend rather than a one-off kernel.

Upstream references:

- [ActivitySim: using Sharrow](https://activitysim.github.io/activitysim/develop/dev-guide/using-sharrow.html)
- [Sharrow Flow and DataTree introduction](https://activitysim.github.io/sharrow/intro.html)
- [ActivitySim performance guidance](https://activitysim.github.io/activitysim/develop/users-guide/performance/index.html)

## Exact exponential semantics

`exp` is deceptively difficult. CUDA's native exponential and NumPy's
vectorized float32 exponential can differ by one or more final bits, and a tiny
weight difference can move a cumulative-choice boundary.

Phase 48 ports NumPy 2.4.6's x86-v3 float32 approximation to CUDA. A live shadow
still found one one-bit difference, so the project did not weaken the test. It
added an exhaustive scanner that interprets every one of the 2^32 possible
float32 bit patterns and compares the CUDA result with this machine's NumPy.

| Exhaustive scan fact | Result |
|---|---:|
| bit patterns visited | 4,294,967,296 |
| finite inputs compared | 4,278,190,080 |
| finite mismatches before correction, all magnitudes | 2,180,536 |
| finite patterns in declared [-80, 80] domain | 2,235,564,034 |
| mismatches inside declared domain | 73 |
| checked-in correction entries | 73 |
| table equality | exact |

The 73 input/output bit pairs are sorted, embedded in the backend, and hashed
as `7d381f55dfc0a244bda39418af15e79d6f81980c194138b43d24dce9e1affe69`.
The GPU uses a small binary-search correction after the base exponential.
Ordinary inputs outside [-80, 80] fail closed; the reviewed -999 padding
sentinel is allowed. The complete live workload actually ranged from -54.528
to 22.047 before row shifting, safely inside the contract.

The scanner is tied to the recorded NumPy 2.4.6, CuPy 14.1.1, Windows x86-v3,
and RTX A4000 environment. A materially different arithmetic target must rerun
qualification rather than assuming the table transfers unchanged.

## Proof stack

Phase 48 requires all of these independent checks:

1. **Exhaustive arithmetic domain:** every float32 bit pattern is visited; the
   73-entry operational-domain table must be complete and hash-identical.
2. **Complete live shadow:** all 19 calls run both the new graph and the CPU
   answer path. Across 4,696,676 alternatives there are zero bit mismatches in
   utility, weight, row total, probability, choice, or logsum.
3. **Sparse choice guard:** seven of 201,390 rows near a cumulative boundary
   are independently adjudicated; zero choices differ before adjudication.
4. **Random-state proof:** every final draw resumes keyed device state with the
   exact chooser permutation and six-draw ledger advance.
5. **Fresh replication:** three fresh-process Phase 47/48 pairs pass every
   runtime gate and independent output verifier.
6. **Final tables:** accessibility, households, joint-tour participants, land
   use, persons, tours, and trips are byte-identical in every production pair
   and in the exhaustive shadow run.

## Reproduction

```powershell
.\.venv-phase8\Scripts\python.exe scripts\scan_phase48_exp_domain.py
.\scripts\run_phase48_resident_destination_ab.ps1 `
  -Repetitions 3 -Households 50000 -RunTag p48final
.\.venv-phase8\Scripts\python.exe `
  scripts\summarize_phase48_qualification.py
.\.venv-phase8\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The expensive complete live shadow uses
`CHOICEFORGE_PHASE48_SHADOW=1`. It is a qualification mode, not a production
timing mode.

Primary evidence:

- `benchmark-results/phase48-p48final-summary.json`
- `benchmark-results/phase48-p48final-qualification.json`
- `benchmark-results/phase48-p48final-{base,gpu,exact}-{1,2,3}.json`
- `benchmark-results/phase48-p48shadow6-gpu.json`
- `benchmark-results/phase48-p48shadow6-output-verification.json`
- `benchmark-results/phase48-exp-domain-scan.json`

## Claim boundary and next major phase

Phase 48 proves a faster, exact, resident final probability boundary on this
public workload and machine. It does not prove a GPU-only ActivitySim model,
portable bit identity on unqualified hardware/software, or a replicated
whole-model speedup. The five component timers are rounded to tenths of a
second and the lifecycle contains roughly 145 seconds of unrelated work.

The next large opportunity is an inter-stage destination supergraph, not a
smaller final-choice kernel. It should retain the sample itself and upstream
mode-choice logsums across model boundaries, batch the small joint and at-work
segments, return only final zone choices and required diagnostics, and expose
the versioned compiler through an upstream Sharrow/ActivitySim backend API.
That wider graph can remove pandas construction and Python dispatch measured in
seconds, large enough to seek replicated component and lifecycle gains while
retaining the same fail-closed and exact-replication guarantees.
