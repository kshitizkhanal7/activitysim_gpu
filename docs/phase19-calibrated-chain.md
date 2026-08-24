# Phase 19: calibrated public MTC household-to-person chain

## Outcome

Phase 19 replaces Phase 18's synthetic choice equations with two real,
calibrated ActivitySim components from the public Prototype MTC Extended model:

1. household auto ownership; then
2. mandatory tour frequency for persons whose CDAP result is mandatory.

The first GPU result is not decorative. It is joined by `household_id` into the
second GPU model and replaces the auto-ownership value in its person features.
On the public 50,000-household checkpoint, the chain covers 132,536 persons and
78,900 mandatory-person choosers. It evaluates 127 published expressions and
two five-alternative MNLs.

The GPU reproduces every saved ActivitySim choice: **0 of 50,000** auto choices
and **0 of 78,900** mandatory-tour-frequency choices differ. All expression
features and ActivitySim random draws are bit-exact. The largest utility error
is `1.776e-15`, and the largest probability error is `4.441e-16`.

Nine measured repetitions on the NVIDIA RTX A4000 give:

| Boundary | Median | Speedup over independent CPU replay |
|---|---:|---:|
| Independent ActivitySim-semantics CPU replay | 0.458724 s | 1.000x |
| GPU modeled execution | 0.025713 s | **17.840x** |
| GPU including one ingress and final egress | 0.037997 s | **12.073x** |

The machine-readable qualification is
[`phase19-calibrated-chain.json`](../benchmark-results/phase19-calibrated-chain.json).

## Claim boundary

This result is a **calibrated checkpoint replay**, not a complete GPU-native
ActivitySim run. The inputs are saved public tables immediately before the two
components. Earlier school location, workplace location, accessibility, free
parking, and CDAP work was produced by the reference ActivitySim run and is
treated as immutable input state.

Allowed CPU control-plane work is limited to reading public Parquet/CSV,
decoding the CDAP `M` input flag, resolving coefficient names, one input
upload, kernel launch and scalar validation, final output download, and report
writing. After ingress is sealed, the GPU must perform:

- zone lookups for land-use and accessibility values;
- all 29 auto-ownership expressions;
- auto utilities, probabilities, random draws, and choices;
- the household-to-person keyed join;
- all 98 active mandatory-tour-frequency expressions; and
- mandatory-tour utilities, probabilities, random draws, and choices.

Telemetry reports zero modeled CPU fallbacks, zero modeled host-to-device bytes
after sealing, and zero modeled device-to-host bytes before final egress.

## Why these two components were chosen

They are the shortest public chain that tests the missing properties of Phase
18 without requiring the entire model at once:

- The equations and estimated coefficients are published, so results are
  behaviorally meaningful rather than synthetic.
- Auto ownership is a household decision used by the later person decision,
  so the chain has a real cross-table dependency.
- The specs exercise comparisons, Boolean algebra, two-sided clipping,
  piecewise variables, conditional division, constants, and unavailable
  alternatives.
- Both components use ActivitySim's stable entity random channels, making exact
  random compatibility testable.
- Their saved input and output checkpoints permit independent reconstruction
  without modifying ActivitySim or trusting only the new implementation.

This gives much more evidence per implementation hour than beginning with a
large destination or scheduling component whose sampling, skims, and
variable-row outputs would introduce several new uncertainties simultaneously.

## What was built

### Coefficient-resolved calibrated compiler boundary

[`calibrated_chain.py`](../src/choiceforge/calibrated_chain.py) reads a public
ActivitySim specification and its coefficient table, resolves symbolic names,
and deliberately quantizes coefficients through float32 before storing their
exact values in float64. That unusual-looking step matches ActivitySim's legacy
path: specification columns are cast to float32, while the expression matrix
and dense utility product are float64.

Only reviewed syntax is accepted. Phase 19 extends the strict expression
interpreter with two-sided `clip` and three-argument `np.where`, which are
required by the published auto-ownership spec. Unsupported functions still
raise `ExpressionUnsupported`; they cannot silently run with Python `eval`.

### GPU indexed joins

`gather_by_key_gpu` stable-sorts source keys, performs `searchsorted` lookups,
validates that every target exists, and gathers columns on the device. It is
used for:

- household home zone to land-use fields;
- household home zone to accessibility fields;
- person household ID to household state; and
- person household ID to the newly computed GPU auto choice.

A missing key is a hard error. Validation reads only a Boolean control scalar;
it never downloads a modeled alternative result.

### Exact ActivitySim random stream on CUDA

ActivitySim seeds a NumPy `RandomState` for each row with:

```text
(base seed + hash32(channel name) + hash32(step name) + entity ID) mod 2^32
```

It then uses the first MT19937 `random_sample` double for these MNL choices.
The new CUDA kernel reproduces that double bit for bit. It computes the first
two tempered MT19937 words needed by NumPy's 53-bit conversion. Because the
first twist outputs depend only on initial state words 0, 1, 2, 397, and 398,
the kernel does not allocate all 624 state words per GPU thread.

The current API deliberately supports only stream offset zero. That is exactly
what these two components need. Asking for a later offset raises
`GpuOnlyViolation` until its semantics are implemented and tested.

### ActivitySim-compatible probabilities and choice traversal

Utilities use float64. Each row is shifted by its largest utility before
exponentiation, following ActivitySim's overflow protection. Weights at or
below `1e-300` become zero, and the remainder is normalized.

The existing float64 CUDA choice kernel subtracts probabilities from the
random draw in published alternative order, exactly like ActivitySim's Numba
`choice_maker`. If rounding leaves a draw beyond the row sum, both paths fall
back to the largest probability.

## Independent oracle and checkpoint proof

The benchmark does not declare the GPU correct merely because it agrees with a
second function using the same GPU code. It uses three evidence layers:

1. An independent NumPy evaluator implements the declared expression,
   coefficient, probability, random, and ordered-choice semantics.
2. That CPU evaluator must reproduce both saved ActivitySim checkpoint output
   columns exactly.
3. GPU intermediates are compared against the independent CPU intermediates,
   and GPU final choices are separately compared with the saved checkpoints.

Observed results:

| Check | Observed result |
|---|---:|
| CPU auto choices vs ActivitySim checkpoint | 0 mismatches |
| GPU auto choices vs ActivitySim checkpoint | 0 mismatches |
| CPU mandatory-tour choices vs checkpoint | 0 mismatches |
| GPU mandatory-tour choices vs checkpoint | 0 mismatches |
| Auto expression features | bit-exact |
| Mandatory-tour expression features | bit-exact |
| Both ActivitySim random streams | bit-exact |
| Largest auto utility error | `1.7763568394e-15` |
| Largest mandatory-tour utility error | `0.0` |
| Largest probability error | `4.4408920985e-16` |
| Choices across nine repeated GPU runs | bit-exact |

These tiny utility/probability differences are ordinary library-level
floating-point differences, far below the published gates of `1e-10` and
`1e-12`. They did not change any choice.

## Performance method

Compilation and warm-up are excluded. Each of nine repetitions runs the
complete independent CPU replay and then the GPU chain. The GPU reports both:

- modeled execution after the input stream is synchronized; and
- transfer-inclusive time from runtime creation and input upload through final
  choice/logsum egress.

The CPU comparison is intentionally not a weak Python row loop for utilities.
It builds the same dense expression matrices, uses NumPy matrix products and
vectorized probability operations, and uses ActivitySim's required per-entity
`RandomState` semantics. Choice traversal is the required ordered algorithm.

The result establishes superiority for this exact two-component boundary on
this machine. It does not imply that file reading or the complete ActivitySim
workflow is 12 times faster.

## Assumptions and limitations

- The public checkpoint is the 50,000-household Prototype MTC Extended run
  already used by earlier project phases.
- Saved checkpoint indices are stable entity IDs and are used as random keys.
- `cdap_activity == "M"` is decoded before ingress because categorical file
  decoding is input handling, not modeled choice logic.
- Household, person, zone, and accessibility keys must be unique and complete;
  the GPU join fails on a missing target.
- Published coefficient resolution and alternative order are trusted input
  configuration, but their files are SHA-256 hashed in the result.
- The GPU-generated auto choice, not the saved auto output, feeds the GPU person
  model. The saved auto output is used only as an oracle after execution.
- Upstream location and CDAP tables are frozen. Their computations are not part
  of the speed measurement or GPU-only claim.
- Tour-row creation after mandatory-tour frequency is not yet ported. The
  qualified output is the person's frequency choice and logsum.
- Random offset zero is supported; general ActivitySim stream advancement is
  not.
- The benchmark has not yet been replicated on a second GPU or a second public
  model.

## Does Phase 19 make Phases 1-18 moot?

No, but it changes which result should lead the story.

Phase 19 supersedes Phase 18's **synthetic behavioral evidence** for this
household-person boundary. If explaining the strongest current result, start
with Phase 19: calibrated equations, exact public choices, and a 12.073x
transfer-inclusive speedup.

It depends on earlier work for the strict expression language, CUDA choice
kernel, float64 probability traversal, fail-closed state runtime, transfer
telemetry, checkpoint corpus, reproducible environment, and lessons about
arithmetic policies. Earlier destination, scheduling, and full-model phases
also cover components that Phase 19 does not. They remain provenance and
regression evidence; they should no longer be mistaken for the headline.

## Reproduce

From the repository root:

```powershell
$env:PYTHONPATH = "src"
./.venv-phase8/Scripts/python.exe -m pytest -q
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase19_calibrated_chain.py `
  --repetitions 9 `
  --output benchmark-results/phase19-calibrated-chain.json
```

The benchmark exits unsuccessfully unless every checkpoint, intermediate,
random, repeatability, boundary, and performance gate passes. The report hashes
all checkpoint/specification inputs and the three implementation files that
define the result.

## Next decisive phase

Phase 20 should extend the calibrated dependency graph in both directions:

1. Port the mandatory-tour postprocessor that creates variable-length tour
   rows, including deterministic IDs and category codes.
2. Add one downstream tour model that consumes those rows, preferably a model
   with skim access, so the chain tests dynamic table growth and the cache.
3. Implement general ActivitySim random offsets or fail-closed channel state
   advancement before any component requests more than one draw per step.
4. Add a device-checkpoint manifest containing input hashes, schema hashes,
   random channel offsets, component completion, and device-table hashes.
5. Repeat on a larger public population checkpoint and then another NVIDIA GPU.

That phase is harder than Phase 19 because it creates new rows and crosses from
person state into tour state. It is also the shortest path from a calibrated
choice replay to a genuinely expanding GPU-native ActivitySim state graph.
