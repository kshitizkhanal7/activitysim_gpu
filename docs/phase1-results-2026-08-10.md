# Phase 1 proof results - 2026-08-10

## Decision

Phase 1 is complete. ChoiceForge now has a strong fused CPU baseline, a warm
Sharrow production profile, raw timing samples, bootstrap intervals, and
multi-warp CUDA correctness coverage.

The evidence supports this bounded claim:

> On an RTX A4000 and Threadripper PRO 5965WX, the ChoiceForge CUDA kernel was
> 4.19x faster than a 24-core fused Numba implementation, including transfers,
> for a synthetic 10,000-chooser workload shaped like MTC mandatory tour
> scheduling (190 alternatives and 69 features). It produced zero different
> choices and a maximum logsum error below 0.000002.

The evidence does **not** yet show that ChoiceForge beats warm Sharrow at an
ActivitySim component boundary. That requires a real component replay fixture
and expression lowering in Phase 2.

## Hardware and software

- CPU: AMD Ryzen Threadripper PRO 5965WX, 24 physical cores and 48 logical CPUs
- GPU: NVIDIA RTX A4000, 16 GB
- OS: Windows 10 build 26200
- Python 3.11.14, NumPy 1.25.2 with OpenBLAS, Numba 0.66.0, CuPy 13.6.0
- ActivitySim 1.4.0 and Sharrow 2.16.2

The strongest CPU configuration uses 24 Numba threads, one per physical core.
The benchmark also records single-thread and 24-thread NumPy/OpenBLAS results.

## Method

Every backend receives identical float32 features, coefficients, constants,
availability, alternative order, and caller-owned random draws.

1. **NumPy materialized:** optimized OpenBLAS matrix multiplication followed by
   vectorized logsum and sampling. It allocates an `N x A` utility matrix.
2. **Fused Numba serial:** three streaming passes per chooser without an
   `N x A` utility matrix.
3. **Fused Numba parallel:** the same operation over 24 physical CPU cores.
4. **CUDA transfer-inclusive:** copies every input to the GPU, launches the
   fused kernel, and returns choices and logsums.
5. **CUDA resident:** inputs remain in GPU memory.

JIT compilation is excluded. Each method receives 10 full-workload warm-ups
before 11 measured repetitions. The ten warm-ups are necessary because the
Windows/CUDA allocation and staging pools continued settling across the first
several large calls. Raw samples are not trimmed.

Speedup intervals are independent bootstrap intervals for the ratio of sample
medians using 20,000 resamples. They describe timing variability only and do
not establish generality across machines or models.

## Strong baseline: 32 alternatives and 16 features

The best CPU method at every size was fused Numba with 24 threads.

| Choosers | Best CPU | GPU incl. transfers | Speedup, 95% interval | GPU resident | Resident speedup | Choice mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.390 ms | 1.260 ms | 0.31x, 0.25-0.43x | 0.810 ms | 0.48x | 0 |
| 100,000 | 5.560 ms | 4.170 ms | 1.33x, 1.21-1.38x | 1.190 ms | 4.66x | 0 |
| 1,000,000 | 59.260 ms | 34.520 ms | 1.72x, 1.66-1.73x | 7.110 ms | 8.33x | 0 |

Against a strong 24-core fused CPU, the GPU loses at 10,000 choosers, wins
modestly at 100,000, and reaches 1.72x transfer-inclusive speedup at one
million. It does not pass the provisional 2x gate for this shape.

## Scheduling shape: 190 alternatives and 69 features

The shape comes from prototype MTC mandatory tour scheduling: 190
departure-duration alternatives and 69 work-scheduling specification rows. The
data are synthetic; only the dimensions are ActivitySim-derived.

| Choosers | Best CPU | GPU incl. transfers | Speedup, 95% interval | GPU resident | Resident speedup | Choice mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 10.760 ms | 2.560 ms | 4.19x, 4.13-4.23x | 0.970 ms | 11.11x | 0 |
| 100,000 | 109.660 ms | 20.380 ms | 5.38x, 5.22-5.42x | 6.730 ms | 16.30x | 2 |

The wider problem gives the GPU enough arithmetic per transferred byte to beat
the fused CPU decisively. The 10,000-row case passes both the 2x speed gate and
the zero-mismatch gate.

The two 100,000-row divergences are retained. NumPy selected alternatives 130
and 116; both fused Numba and CUDA selected 131 and 117. The draws were only
`1.19e-7` and `2.98e-7` from NumPy cumulative-probability boundaries. Logsums
on both rows matched at reported float32 precision. A documented float64 or
hybrid precision policy is therefore required before framework equivalence.

## CUDA defect found and fixed

The first 190-alternative run exposed a shared-memory race invisible at 32
alternatives. One warp could reuse the reduction buffer before another warp
loaded the maximum, producing large errors. A block-wide barrier now separates
maximum reduction from scratch reuse in both kernels. Regression tests cover
33 and 190 alternatives. The full integration suite passes 15 tests.

## Warm Sharrow profile

The earlier 27.3-minute Sharrow run was compilation and reached approximately
15 GB USS. With the cache complete, three `sharrow: require`, single-process
production runs completed in 61.650, 59.119, and 61.790 seconds. Median model
runtime was 61.650 seconds. Final household, person, tour, and trip files had
identical SHA-256 hashes across all runs.

| Component | Median | Fraction of median total |
|---|---:|---:|
| Trip destination | 15.6 s | 25.3% |
| Mandatory tour scheduling | 5.6 s | 9.1% |
| Trip scheduling | 5.2 s | 8.4% |
| CDAP simulate | 4.6 s | 7.5% |
| School location | 4.2 s | 6.8% |
| Trip mode choice | 2.6 s | 4.2% |
| Tour mode choice simulate | 2.5 s | 4.1% |

Mandatory tour scheduling is a tractable fixed-alternative target, but at 9.1%
of total time it cannot alone produce a 10% runtime reduction. Phase 2 should
build a reusable scheduling backend for mandatory tour scheduling, trip
scheduling, and related components. The first two represent 17.5% of runtime.

## Gate assessment

| Gate | Result |
|---|---|
| Strong optimized CPU baseline | Pass |
| Raw samples and reproducible environment | Pass |
| Warm Sharrow production profile | Pass |
| At least 2x transfer-inclusive on a relevant shape | Pass for scheduling shape |
| Zero choice mismatches | Pass at 10,000; fails by 2 boundary rows at 100,000 |
| Real ActivitySim component superiority | Not tested until Phase 2 |

Phase 1 proves that the GPU advantage is workload-dependent and can be large
for scheduling-shaped arithmetic, while identifying the precision and
integration work still required.

## Reproduce

```powershell
.\.venv-asim\Scripts\python.exe -m pytest -q -p no:cacheprovider

.\.venv-asim\Scripts\python.exe benchmarks\benchmark_phase1_linear_choice.py `
  --sizes 10000 100000 1000000 --alternatives 32 --features 16 `
  --repetitions 11 --warmups 10 --cpu-threads 24 `
  --output benchmark-results\phase1-a4000-32x16.json

.\.venv-asim\Scripts\python.exe benchmarks\benchmark_phase1_linear_choice.py `
  --sizes 10000 100000 --alternatives 190 --features 69 `
  --repetitions 11 --warmups 10 --cpu-threads 24 `
  --output benchmark-results\phase1-a4000-scheduling-shape.json

.\.venv-asim\Scripts\python.exe scripts\summarize_phase1_results.py
```

Raw and derived artifacts are under `benchmark-results/`. Warm ActivitySim
outputs are under `benchmark-data/prototype_mtc/prototype_mtc/`.
