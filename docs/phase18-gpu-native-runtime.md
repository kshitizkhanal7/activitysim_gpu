# Phase 18: GPU-native runtime foundation

## Outcome

Phase 18 proves a larger idea than the earlier single-component integration:
an entire **model-shaped chain of dependent calculations** can stay on the
GPU. On the complete public Prototype MTC household table (2,875,192 rows), the
chain runs 14.575x faster than its fused parallel Numba CPU comparison when
only modeled computation is timed, and 8.260x faster when one input upload and
final output download are included.

This is not yet a whole ActivitySim implementation. The implemented vertical
slice includes GPU feature construction, entity-stable random draws, two
dependent 21-alternative multinomial-logit choices, and deterministic zone
aggregation. It deliberately excludes tour/trip table construction, every
remaining ActivitySim component, file parsing, configuration parsing, and
checkpoint serialization.

The qualifying artifact is
[`phase18-gpu-native-full-households.json`](../benchmark-results/phase18-gpu-native-full-households.json).

## The exact meaning of “GPU-native”

An operating system and a Python process on the CPU must launch CUDA kernels.
CSV and YAML parsers also run on the CPU, and final results usually have to be
written to storage. Therefore “literally no CPU instructions” is neither
possible nor useful on this workstation.

Phase 18 uses this testable boundary:

| Allowed control-plane CPU work | Forbidden modeled CPU work |
|---|---|
| read public input and configuration | evaluate a utility expression |
| upload one input partition | generate a modeled random draw |
| launch and synchronize kernels | make a choice or compute a logsum |
| download final published outputs | aggregate modeled values |
| write checkpoints and reports | silently fall back from an unsupported GPU stage |

Once `GpuNativeRuntime.seal_ingress()` is called, a modeled host array, modeled
download, or CPU fallback is a hard `GpuOnlyViolation`. Telemetry records all
three counters. All are zero in the qualified run.

Scalar values read by the host to determine launch size or detect a kernel
error are control-plane synchronizations. They must never become a second CPU
implementation of the model calculation.

## Why this architecture

ActivitySim is a sequence of state-changing model steps, and Sharrow currently
compiles supported utility expressions through Numba. ActivitySim's own
documentation also explains that Sharrow may revert to legacy evaluation unless
configured to fail, and that preprocessors and postprocessors are outside the
Sharrow utility path. Those facts make a whole GPU model a runtime-and-state
problem, not merely a faster utility kernel problem:

- [ActivitySim workflow state](https://activitysim.github.io/activitysim/develop/dev-guide/core-workflow.html)
- [ActivitySim's Sharrow modes and limitations](https://activitysim.github.io/activitysim/develop/dev-guide/using-sharrow.html)
- [Sharrow overview](https://activitysim.github.io/sharrow/intro.html)

The Phase 18 design follows from five observations:

1. A strict GPU-only promise must fail closed. A hidden pandas or NumPy path
   would make both the performance and energy claim meaningless.
2. Randomness must be attached to a stable entity ID, seed, and stream—not row
   position—so a different partition size cannot change behavior.
3. Reductions must have a declared order. Atomic GPU additions can finish in
   different orders and change floating-point low bits.
4. State must remain in device arrays between stages. Repeated download and
   upload would erase much of the GPU advantage.
5. A 16 GB device cannot hold every possible full-model object at once, so
   deterministic partitions and an explicit memory budget are part of
   correctness, not afterthoughts.

## What was built

### Fail-closed runtime and device state

[`gpu_native.py`](../src/choiceforge/gpu_native.py) adds:

- `DeviceTable`, which rejects any column without a CUDA array interface and
  requires equal row counts;
- `GpuNativeRuntime`, which owns device tables, seals ingress, launches modeled
  stages, controls final egress, and records transfer/fallback telemetry;
- `GpuOnlyViolation`, the hard error for an invalid boundary crossing;
- `GpuMemoryBudget`, which makes reserve, hot skims, persistent state, and
  workspace allocations explicit; and
- `plan_household_partitions`, which creates complete deterministic half-open
  ranges.

### Partition-invariant GPU random numbers

The new CUDA kernel hashes `(entity_id, global_seed, stream_id)` with SplitMix64
and converts the upper 24 bits to a float32 value strictly inside `(0, 1)`.
Because it does not depend on batch position, one large partition and many
small partitions receive the same draw for every entity. An independent NumPy
oracle produces the exact same bits.

This is a foundation, not yet a drop-in reproduction of every historical
ActivitySim random channel. An upstream integration must assign stable stream
IDs and demonstrate compatibility or publish a deliberate new random policy.
ActivitySim's older random-number documentation likewise emphasizes stable
global, channel, row, and step seeding.

### Deterministic GPU aggregation

The segmented-sum kernel accepts sorted group IDs. One GPU thread owns each
adjacent group and adds rows in their original order. It uses no atomics. The
output remains device-sized: a start flag marks valid sums, avoiding a host
round trip just to discover the compact output length.

This policy sacrifices some possible parallel reduction speed for exact,
explainable ordering. Later work may add a fixed reduction tree, but it must
retain a hashed arithmetic policy and a CPU/device oracle.

### Full public-household vertical slice

[`benchmark_phase18_gpu_native.py`](../benchmarks/benchmark_phase18_gpu_native.py)
runs these modeled stages:

1. Build eight household features on the GPU from public MTC fields.
2. Generate a stable first random stream on the GPU.
3. Run a fused 21-alternative household choice without a global utility table.
4. Feed the first choice into a second feature vector.
5. Generate a second stable stream and run the dependent choice.
6. Stable-sort households by TAZ on the GPU and aggregate first choices by TAZ.
7. Download only the outputs used by the qualification report.

The synthetic coefficients are fixed and deterministic. The household rows are
real public benchmark data, but this is a systems workload, not a calibrated
behavioral model. It must not be used for policy forecasts.

## Data assumptions

All assumptions are executable and visible in the benchmark:

- `household_id` is the stable random key.
- MTC negative income codes mean missing for this systems proof and map to zero.
- Income is capped at $250,000 and linearly normalized. The cap prevents a
  small number of coding outliers from dominating utilities.
- Household size is capped at 8, workers and automobiles at 5, and household
  type at 7 before normalization.
- Coefficients are deterministic synthetic values across 21 alternatives and
  eight features. They test execution; they are not estimated preferences.
- Float32 is the published arithmetic policy for this vertical slice.
- GPU partition equality is bit-exact. CPU/GPU comparison has a numerical
  tolerance because CUDA `expf`/FMA and Numba math are not bit-identical.

An initial full-population run with `log1p(income)` found one CPU/GPU boundary
choice caused by a one-unit float32 difference between the two libraries. The
final benchmark uses a bit-identical linear input transform. Even then, the
different CPU/GPU exponential and fused-multiply-add implementations create one
first-stage boundary difference. Phase 18 reports that result instead of
pretending it does not exist.

## Full-public-data results

Environment: NVIDIA RTX A4000, driver 571.59, Python 3.11.14, public MTC full
geography, nine measured repetitions after JIT warm-up.

| Measure | Result |
|---|---:|
| Public households | 2,875,192 |
| GPU modeled-compute median | 0.031357 s |
| GPU with input/output transfer median | 0.055327 s |
| Parallel fused Numba CPU median | 0.457023 s |
| GPU compute speedup | **14.575x** |
| GPU transfer-inclusive speedup | **8.260x** |
| Sampled active GPU allocations | 474,411,520 bytes |
| Modeled CPU fallbacks | **0** |
| Modeled transfers after sealed ingress | **0 bytes** |

The first 50,000-row GPU sample after shape growth is visibly cold. Medians,
not minima, are the reported statistic. The active-allocation number is sampled
after stages and is a lower bound; short-lived temporary peaks can be higher.

### Replication and numerical gates

| Gate | Observed | Passed? |
|---|---:|---:|
| GPU final choice, full table vs 250k partitions | 0 differences | yes |
| GPU final logsum, full table vs 250k partitions | 0 bit differences | yes |
| First CPU/GPU choice mismatch | 1 / 2,875,192 (0.348 per million) | yes, limit 1 per million |
| Dependent CPU/GPU choice mismatch | 3 / 2,875,192 (1.043 per million) | yes, limit 2 per million |
| Largest dependent logsum difference | 0.007143 | yes, limit 0.01 |
| Largest zone-sum difference | 1.0 | yes, limit 1.0 |
| GPU-only boundary counters | all zero | yes |

The distinction matters. **Replication guarantee** means the GPU result is
unchanged by partitioning. **Cross-architecture numerical equivalence** means
the strong CPU implementation differs only within published bounds. Phase 18
does not claim bit-identical CPU/GPU MNL arithmetic.

## Capacity result on this RTX A4000

The live audit is
[`phase18-capacity-audit.json`](../benchmark-results/phase18-capacity-audit.json).
The device reports 16,376 MiB. NVIDIA specifies 16 GB GDDR6 ECC, 448 GB/s memory
bandwidth, 6,144 CUDA cores, and 19.2 TFLOPS FP32 for this model in the
[official RTX A4000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf).

The public OMX file contains 826 datasets totaling 13.389 GiB when counted as
raw uncompressed arrays. Earlier 50,000-household Phase 17 ActivitySim
integration runs peaked from 8,373 to 8,449 MiB of GPU memory. Therefore loading
every skim plus a safe reserve, full state, and working buffers is not credible
on this device.

The planning allocation is:

| Pool | Assumed budget |
|---|---:|
| Driver/CUDA/error reserve | 2 GiB |
| Hot skim cache | 4 GiB |
| Persistent model state | 2 GiB |
| Largest component workspace | 3 GiB |
| Unallocated safety/partition space | 4.99 GiB |

These are design assumptions, not measured full-model high-water marks. No
maximum whole-model household partition is claimed yet. The honest conclusion
is that a hot-skim cache and deterministic household partitions are mandatory.
The small Phase 18 state graph fits all 2.875 million households at once; the
eventual complete graph probably will not.

## How to reproduce

From the repository root with the Phase 8 CUDA environment:

```powershell
$env:PYTHONPATH = "src"
./.venv-phase8/Scripts/python.exe -m pytest tests/test_gpu_native.py -q
./.venv-phase8/Scripts/python.exe benchmarks/benchmark_phase18_gpu_native.py `
  --households 2875192 --partition-households 250000 --repetitions 9 `
  --output benchmark-results/phase18-gpu-native-full-households.json
./.venv-phase8/Scripts/python.exe scripts/audit_phase18_gpu_capacity.py
./.venv-phase8/Scripts/python.exe -m pytest -q
```

The benchmark exits unsuccessfully if any performance, partition, numeric, or
GPU-only gate fails.

## What Phase 18 proves—and does not prove

It proves:

- a fail-closed GPU-resident state boundary can be enforced in code;
- random draws can be exact across different partitions;
- two dependent choices and an ordered aggregation can remain on the GPU;
- the entire public household table fits this vertical slice;
- the GPU is materially faster than a fused parallel Numba CPU baseline at
  this scale, including the permitted boundary transfers; and
- all stated qualification gates pass on this machine.

It does not prove:

- a complete calibrated ActivitySim model runs GPU-native;
- behavioral validity of the synthetic coefficients;
- bit-identical MNL arithmetic between CPU and GPU;
- enough memory for all persons, tours, trips, timetables, shadow prices, and
  skims simultaneously;
- restartable GPU checkpoints;
- faster file parsing or output writing; or
- performance on another GPU or public model.

## Next implementation sequence

1. Add GPU-native indexed joins, table filters, stable sorts, scatter/gather,
   and categorical encodings with exact or explicitly bounded policies.
2. Implement a bounded hot-skim cache with event-safe eviction and record hit,
   miss, and transfer bytes by model component.
3. Port one calibrated household/person component chain while preserving stable
   ActivitySim entity channels and compare every intermediate table.
4. Add device checkpoints or a documented host-checkpoint boundary so restart
   behavior is reproducible after a failure.
5. Port tours, trips, timetable scheduling, destination sampling, and shadow
   pricing one dependency layer at a time. Unsupported operations must remain
   hard errors during GPU-only qualification.
6. Re-measure high-water memory after every new component and derive the
   production partition size from evidence, not extrapolation.
7. Run interleaved full-model CPU/GPU trials only after the complete calibrated
   chain passes table-by-table equivalence.

Phase 18 is successful because it converts “what if the whole model stayed on
the GPU?” from a slogan into a measured runtime contract. It is the foundation,
not the finish line.
