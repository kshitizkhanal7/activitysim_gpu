# Phase 35: resident trip runtime and replication proof

## Result

Phase 35 extends the qualified runtime through two late-model trip components:

- trip scheduling uses a persistent CUDA probability service; and
- trip-destination mode logsums use a reviewed native 11-float/45-integer ABI,
  GPU availability construction, generated CUDA utility, and CUDA nesting.

One complete 50,000-household Prototype MTC run passed every proof gate. It
executed all 34 ActivitySim model steps, changed zero modeled decision cells,
produced all seven final CSVs byte-for-byte identically to its Phase 34 control,
used no fallback, and released 6,198,614,528 resident CUDA bytes once after the
final GPU skim consumer.

Phase 35 has therefore proved coverage and replication. It has **not yet proved
a publishable incremental whole-model speedup over Phase 34**. The available
matched timing was contaminated by an unrelated process holding the GPU at 98%
utilization. Its apparent 2.55x result is rejected in this report rather than
presented as success.

## What changed

### Persistent trip scheduling

ActivitySim's scheduler repeatedly chooses one of 19 time alternatives. The
new service keeps the normalized 1,368-row probability specification resident,
reuses a CUDA workspace, applies the controlled ActivitySim draws, clips and
renormalizes probabilities, and returns only the selected alternative. It does
not reorder trips or replace ActivitySim's retry rules and random ledger.

The qualified run recorded:

| Measure | Value |
|---|---:|
| scheduling calls | 548 |
| chooser rows | 210,110 |
| failed choices handled by ActivitySim retries | 66,899 |
| first-trip calls | 200 |
| resident specification | 1,368 rows x 19 alternatives |
| fallback calls | 0 |
| host-to-device bytes | 4,202,200 |
| device-to-host bytes | 840,440 |
| CUDA kernel time | 0.0288 s |
| complete service time | 3.7520 s |

The kernel itself is fast. Random-ledger work and host indexing are now larger
than CUDA execution, which defines the next optimization boundary.

### Native trip-destination logsum ABI

The former path built a very wide pandas preprocessor table before evaluating
trip-mode utilities. Phase 35 reconstructed the exact public-model inputs from
raw trip, tour, land-use, controlled-draw, and resident skim data. Unknown
source labels, changed draw counts, malformed skim groups, or a changed input
ABI fail closed.

The complete run proved:

| Measure | Value |
|---|---:|
| purpose programs | 30 of 30 |
| directional logsum rows | 4,188,312 |
| dense preprocessor rows read | 0 |
| fallback calls | 0 |
| host ABI bytes formed | 1,792,597,536 |
| host ABI construction | 6.8592 s |
| upload | 0.1759 s |
| availability CUDA kernels | 0.1366 s |
| generated utility CUDA kernels | 2.5009 s |
| nested-logsum CUDA kernels | 0.0917 s |

This is an important architectural result but also exposes the remaining cost:
the runtime still forms 1.79 GB of typed ABI matrices on the CPU. Phase 35
eliminates dense expression evaluation, not all host construction.

## Replication contract

Before production enablement, the native path ran in shadow mode against the
authoritative ActivitySim/Sharrow calculation for all 30 purpose batches. The
largest nested-logsum difference was below 1e-12. A separate 500-household
production qualification found exact decisions.

The final 50,000-household matched verification is stronger:

- decision cells different: **0**;
- decision rows different: **0**;
- byte-identical final output CSVs: **7 of 7**;
- destination-logsum maximum difference: **0**;
- mode-logsum maximum difference: **0**;
- random-ledger mismatches: **0**; and
- silent fallback calls: **0**.

The production runner additionally requires exactly three configured wait-time
draws before consuming randomness. This prevents a future configuration change
from advancing the random stream under an obsolete assumption.

## The rejected timing result

The first Phase 34/35 harness run reported 570.9 versus 223.5 all-model seconds.
That comparison is invalid. During the Phase 34 control, an unrelated process
occupied the GPU and inflated trip mode alone to 318.6 seconds. GPU utilization
was measured at 98%. The load changed between the two halves, violating the
matched-run assumption. The summary is retained as diagnostic evidence, but
the 2.55x number must not be cited as Phase 35 performance. The machine-readable
[`phase35-p35pair1-timing-rejection.json`](../benchmark-results/phase35-p35pair1-timing-rejection.json)
records that exclusion so later analysis cannot silently promote the invalid
timing into a headline result.

The Phase 35 candidate's own trip-destination stage took 27.1 seconds under
that load. Its native preprocessor/random-ledger portion was only 1.22 seconds,
but native host ABI construction moved into the logsum portion. Structural
timers show where time went; they do not substitute for clean paired proof.

## Reproduction

Run three direct Phase 34/35 pairs on an otherwise idle machine:

```powershell
.\scripts\run_phase35_resident_trip_ab.ps1 `
  -Repetitions 3 `
  -Households 50000 `
  -RunTag p35proof
```

The harness starts each process independently, compares every final output,
requires every proof gate, reports every component, and rejects the experiment
if Phase 35 loses any pair. Monitor GPU utilization separately and discard a
pair if another workload overlaps either half.

## Next major opportunity

Phase 36 should generate the 11/45 trip ABI directly on the device from a much
smaller raw state packet. Land-use columns and reusable tour attributes should
remain resident; one CUDA preparation kernel should form utility inputs and
availability flags without materializing 1.79 GB on the host. Scheduling should
likewise move host indexing and controlled probability lookup into a resident
tour-chain state machine while ActivitySim keeps the exact random ledger and
retry semantics.

Success requires clean three-pair Phase 35/36 timing, a regular-ActivitySim
comparison, zero changed decisions, bounded declared diagnostics, no fallback,
explicit H2D/D2H accounting, changed-scenario tests, and one final cache release.
Until that evidence exists, Phase 35 is a replication-qualified architectural
advance, not a completed speed-superiority claim.
