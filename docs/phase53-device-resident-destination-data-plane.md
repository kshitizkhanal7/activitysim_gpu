# Phase 53: upstream compact-owner destination data plane

## Outcome

Phase 53 removes the largest remaining Phase 52 host-side cost. ActivitySim
previously joined every person or tour field onto every sampled destination
row, after which Phase 52 scanned those repeated columns to reconstruct one
record per owner. Phase 53 intercepts both public logsum entry points before
that expansion. It joins the authoritative person/tour data to one sampled row
per owner and gives the compact owner table plus the row-specific destination
vector directly to the existing hash-verified CUDA service.

On the public Prototype MTC Extended workload at 50,000 households and 1,454
zones, three matched Phase 52/53 pairs produced:

| Measurement | Phase 52 median | Phase 53 median | Result |
|---|---:|---:|---:|
| host packet preparation | 4.312 s | 1.300 s | 3.318x; 69.86% lower |
| instrumented destination service | 7.846 s | 4.989 s | 1.573x; 36.41% lower |
| five destination components | 16.2 s | 12.6 s | 1.286x; 22.22% lower |

Phase 53 won all three pairs on all three measured boundaries. Each pair
covered all 19 calls, 201,390 owners, and 4,696,676 sampled destinations.

## Architecture

The two adapters preserve ActivitySim's public model settings, controlled
random generator, sampled destination ordering, and final-choice API:

1. `run_location_logsums` selects the first sampled row for each contiguous
   person ID and joins it to the filtered one-row-per-person table.
2. `run_destination_logsums` selects the first sampled row for each tour ID,
   resolves its person relation once, and joins at tour cardinality.
3. The Phase 53 runtime proves that compact owner IDs exactly equal the sampled
   run IDs, consumes owner fields without dense stability scans, and consumes
   destination IDs from the full sample vector.
4. The Phase 52 four-row kernel, source hash, semantic/native plan caches,
   reusable CUDA workspaces, controlled draws, nested-logit graph, and resident
   final-choice handoff remain unchanged.

This is a data-plane improvement rather than a new arithmetic implementation.
The kernel still records source SHA-256
`599a9704be0992d2863320390cbca0028c7a578ecacf72d69de2e658a5d79906`.

## Replication guarantees

Every qualified shard proves:

- the expected call, owner, and sampled-row cardinalities;
- every call used a Phase 53 upstream compact owner source;
- all repeated owner groups remained contiguous and exactly identified;
- the same four-row fused CUDA program and dense-device-ABI elimination;
- no generic generator or CPU fallback;
- exact school, workplace, and tour destination decisions;
- school/workplace logsum error at most `1.9073486328125e-6` against a
  `1e-5` gate;
- tour destination logsum error at most `1.9073486328125e-6` against a
  `1e-4` gate.

The compact route is fail closed. A missing relationship, duplicated owner
run, reordered owner table, absent model field, changed zone universe, integer
overflow, source/schema change, or unsupported chunked call raises an error.

## Qualification method and machine limitation

The full workload was measured in two deterministic shards. The first runs
from initialization through workplace location (seven calls). The second
resumes the previously verified mandatory-scheduling checkpoint and runs
through at-work destination (twelve calls). Their destination-service and five
component timings are measured values and are added only after their combined
cardinalities reproduce the exact 19-call public workload.

Sharding was necessary because this Windows host repeatedly refused a 79 MiB
pandas allocation inside unchanged mandatory tour scheduling, despite more
than 24 GiB of reported virtual memory being free. The failure occurs after
the seven Phase 53 calls and is outside the changed destination path. A resumed
complete downstream run finished with zero changed modeled decisions, and the
six formal qualification shards all passed. Nevertheless, the monolithic
Phase 53 all-model time is not claimed as measured.

Replacing only the five measured Phase 52 component times gives a conservative
median all-model projection of 133.391 seconds versus the measured Phase 52
median of 136.991 seconds. Against the 205.4-second regular ActivitySim median,
that projection is 1.540x faster (35.06% lower). This projection is clearly
separated from the measured destination claims.

## Evidence and reproduction

The consolidated artifact is
`benchmark-results/phase53-p53final-qualification.json`. Its `success` field
and every proof gate are true. Source reports are
`phase53-p53{pre,post}mandatory-gpu-{2,3,4}.json` with their checkpoint files.

Rebuild the consolidated result:

```powershell
.\.venv-phase8\Scripts\python.exe scripts\build_phase53_qualification.py `
  --output benchmark-results\phase53-p53final-qualification.json `
  --baseline benchmark-results\phase52-p52final-gpu-1.json benchmark-results\phase52-p52final-gpu-2.json benchmark-results\phase52-p52final-gpu-3.json `
  --pre benchmark-results\phase53-p53premandatory-gpu-2.json benchmark-results\phase53-p53premandatory-gpu-3.json benchmark-results\phase53-p53premandatory-gpu-4.json `
  --post benchmark-results\phase53-p53postmandatory-gpu-2.json benchmark-results\phase53-p53postmandatory-gpu-3.json benchmark-results\phase53-p53postmandatory-gpu-4.json
```

Run the full automated suite:

```powershell
.\.venv-phase8\Scripts\python.exe -m pytest -q
```

## Assumptions and claim limits

The result covers this public model, scale, GPU, expression program, zone
mapping, time-period system, skim directions, and current ActivitySim join
semantics. The compact table is authoritative because it is constructed inside
the two public logsum functions from their original source tables, not inferred
from sampled-row values. Other models must qualify their column relations and
cardinalities before enabling this path.

The three-pair design reduces ordinary timing noise but is not randomized. The
five-component gain includes pandas, ActivitySim orchestration, sampling, CUDA,
and final choice. The service metric isolates the changed logsum service more
closely. Only those boundaries are called measured.

## Next major opportunity

Phase 54 should move the remaining controlled wait-table transformation and
sample topology to a truly device-owned packet. The sampling service already
has owner IDs, offsets, and destination IDs; it should publish those buffers to
the logsum service through a versioned lease instead of recreating the vectors
in pandas/NumPy. A numerically qualified CUDA wait transform can consume the
six compact controlled normal draws and resident land-use density bands. The
phase should require exact choices, bounded logsums, zero fallback, three
matched wins, and a measured destination service below four seconds.
