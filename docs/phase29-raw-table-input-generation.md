# Phase 29: declared raw-table source compiler

## Outcome

Phase 29 replaces dense-row discovery with a fail-closed compiler whose inputs
are one row per tour, the land-use table, controlled stochastic inputs,
alternative slots, and resident network skims. The compiler declares every
strict utility input and every skim coordinate before it sees the legacy dense
answer. The dense ActivitySim result is compared afterward as an independent
oracle and contributes zero bytes to plan construction.

This is a larger boundary than Phase 28. It covers:

- all 10 floating and 31 integer strict input columns in each program;
- all 16 grouped skim coordinates across six OD/time directions;
- direct parking rates from land use and free-parking state;
- all 18 road and transit availability formulas on CUDA;
- controlled ride-hail wait-time inputs;
- the six real 315-term utility programs, nested logsums, cache scatter,
  scheduling choice, and timetable mutation; and
- changed raw-table populations as well as changed raw-skim worlds.

## Public benchmark result

The proof uses the public 50,000-household Prototype MTC Extended workload.
Each of three fresh Python/CUDA processes executes five measured complete
resident replays.

| Metric | Process 1 | Process 2 | Process 3 | Median of medians |
|---|---:|---:|---:|---:|
| Raw-table-input-to-calendar graph | 0.210148 s | 0.225311 s | 0.226075 s | **0.225311 s** |
| Raw input generation | 0.010439 s | 0.010328 s | 0.010506 s | **0.010439 s** |
| ActivitySim checkpoint-to-result run | 30.647 s | 30.759 s | 31.021 s | **30.759 s** |

Every replay processes 1,210,124 mode-logsum rows and schedules 81,983 tours.
Every logsum bit and final TDD label matches the qualified reference. The
sealed graph has zero modeled post-seal host-to-device traffic, zero
intermediate modeled device-to-host traffic, and zero modeled CPU fallback.

## The 57-source contract

For each program the compiler produces a manifest with 57 declared sources:

- 41 strict floating/integer input columns;
- origin, destination, and time coordinates for outbound-time skims;
- reversed origin, destination, and time coordinates for return-time skims;
- two-dimensional OD and reversed-OD coordinates; and
- the equivalent inbound/outbound time combinations used by ride-hail toll
  expressions.

Unknown columns, skim directions, periods, zones, or shapes fail compilation.
The compiler does not inspect a dense column and then guess whether it is
constant, tour-specific, or slot-specific.

## Direct raw-table formulas

The one-row-per-tour source is the joined ActivitySim tour/person/household
view before the logsum preprocessor expands it across possible time choices.
The compiler reads age, auto ownership, household size, worker count, value of
time, tour purpose/category, free parking, home zone, and the purpose-specific
destination. It reads terminal time, topology, density, parking price, CBD
area type, population, employment, and acreage from land use.

Ride-hail waits are recreated from the controlled standard-normal inputs and
the published density-band mean/standard-deviation tables using the same
scaled-lognormal transformation and clipping limits. The public configuration
has zero standard deviation for these waits, but changed-scenario qualification
uses nonzero values so the stochastic formula is actually exercised.

## Parking provenance

Phase 28 recovered a parking rate by finding a double-precision multiplier
that regenerated all observed rounded costs. That was exact but still a
qualification technique.

Phase 29 reads the unrounded hourly peak parking rate directly from
`land_use.PRKCST`. A work tour with `free_parking_at_work` receives a zero rate.
The CUDA generator then multiplies the rate by the exact alternative duration
and converts to the strict float input type. No parking answer is recovered
from a dense output.

## All availability fields are semantic

Phase 28 generated 14 availability fields whose public values varied by
tour/time row. Four additional fields happened to be constant in that run and
were therefore represented as constants. That was exact for the benchmark but
weaker for a changed network.

Phase 29 generates all 18 fields from skims:

- SOV, tolled SOV, HOV2, tolled HOV2, HOV3, and tolled HOV3 availability;
- local, light-rail/ferry, express, heavy-rail, and commuter walk transit;
- the matching drive-transit fields, including auto ownership; and
- walk and drive ferry availability.

The formulas retain the public MTC transit scale factor of 100 and the same
comparison/summation structure as the reviewed preprocessor.

## Changed-world qualification

Five independently generated raw-table populations contain 10,000 tours with
changed zones, land-use quantities, value of time, parking prices, free
parking, household/person attributes, density bands, and nonzero stochastic
wait dispersion. An independent implementation checks 23 direct source
outputs per scenario. Every value is exact and all five hashes differ.

Five separate changed raw-skim worlds contain another 8,000 rows. Across them,
all 18 availability formulas are generated and every CUDA result matches an
independent NumPy oracle. No anonymous response pattern is accepted, and all
five result hashes differ.

These synthetic gates prove input responsiveness. They do not claim that five
complete alternative ActivitySim policy models have been run end to end.

## Memory and performance tradeoff

Phase 29 retains 24,849,394 bytes for 503,411,584 bytes of removed dense input
and coordinate arrays, a **20.259x reduction**. The compact state is 22.659%
larger than Phase 28. This increase is deliberate: fields that happened to be
constant in the public sample are stored per tour so a changed population can
change them without recompilation or an invalid constant assumption.

The median complete graph is 6.380% slower than Phase 28. Input generation is
0.010439 seconds of the 0.225311-second graph. The gain is a stronger upstream
contract and scenario validity, not a new whole-model speedup claim.

## Proof gates

The hash-chained summary requires:

- three independent processes and 15 complete public replays;
- 57 declared sources in every one of the six programs;
- zero dense-oracle bytes read while constructing each plan;
- all 18 availability formulas generated per public program;
- parking rates read directly from land use/free-parking state;
- zero anonymous response patterns;
- bit-identical logsums and exact final schedules;
- zero post-seal modeled H2D, intermediate D2H, or CPU fallback; and
- all changed raw-table and changed-skim gates passing.

The complete test suite passes 135 tests.

## Honest boundary and the next ambitious phase

The Phase 29 compiler no longer depends on dense preprocessor values. The
current qualification harness still lets ActivitySim create those dense rows
for two separate reasons: to provide an independent answer key and to expose
the already-compiled strict utility ABI. Consequently, the 30.759-second cold
ActivitySim time is not replaced by the 0.225-second resident graph.

The next major phase should build a native schema/IR bootstrap. It should load
the reviewed utility specification and skim metadata, allocate the strict ABI,
compile kernels, and invoke the Phase 29 raw source compiler without running
ActivitySim's dense preprocessor at all. The legacy path should run only in a
separate qualification process. That phase should also replace the remaining
57-entry boundary-decision map with a shared Sharrow/CUDA arithmetic contract.

## Reproduction

Run `scripts/run_phase22_integrated_scheduling.py` with
`--resident-raw-table-input-report` in three fresh processes, then combine the
reports with `scripts/summarize_phase29_raw_tables.py`. Run
`scripts/qualify_phase29_raw_table_scenarios.py` for changed-world evidence.
The aggregate summary SHA-256 chains every resident, live, changed-world, and
Phase 28 comparison report.
