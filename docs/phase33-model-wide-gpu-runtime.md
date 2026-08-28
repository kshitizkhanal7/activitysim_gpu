# Phase 33: expanded model-wide GPU runtime

## Outcome

Phase 33 expands the qualified CUDA boundary from three ActivitySim consumers
to six. It retains native mandatory scheduling, trip-destination logsums, and
trip-mode utilities, then adds:

1. non-mandatory tour-destination mode logsums;
2. non-mandatory tour scheduling; and
3. primary tour-mode utilities.

The performance result is a direct comparison with regular pinned ActivitySim
using its required Sharrow backend. Three fresh, matched 50,000-household pairs
run all 34 public Prototype MTC model steps and all 1,454 zones.

| Pair | Regular ActivitySim | Phase 33 | Saved | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 225.7 s | 175.6 s | 50.1 s | 22.20% | 1.285x |
| 2 | 206.6 s | 178.1 s | 28.5 s | 13.79% | 1.160x |
| 3 | 204.5 s | 177.0 s | 27.5 s | 13.45% | 1.155x |
| Median | **206.6 s** | **177.0 s** | **29.6 s** | **14.33%** | **1.167x** |

The candidate won every pair. The median run changes from about 3 minutes 27
seconds to 2 minutes 57 seconds. These are sums of ActivitySim's named model
timers, not isolated kernel times.

## What moved to CUDA

Phase 33 is an integrated runtime, not a second model implementation. ActivitySim
still owns model order, tables, sampling, feasibility, controlled random draws,
and publication. ChoiceForge replaces supported arithmetic at explicit seams.

```text
ActivitySim workflow and tables
        |
        +-- mandatory tour scheduling -------- native strict ABI + CUDA choice
        +-- non-mandatory destination -------- generated CUDA logsum evaluator
        +-- non-mandatory scheduling --------- CUDA utility and choice scan
        +-- primary tour mode ---------------- generated CUDA utility evaluator
        +-- trip destination ----------------- generated CUDA logsum evaluator
        +-- trip mode ------------------------ generated CUDA utility evaluator
        |
        +-- all other steps ------------------ regular ActivitySim/Sharrow
```

The read-only CUDA skim cubes remain resident across the six consumers and are
released once after trip mode choice. Phase 33 does not load the separate
Phase 31 native skim file beside ActivitySim's host skim dataset, which would
duplicate more than 6 GB of network data while unported components still need
the ActivitySim representation.

The three new groups prove their own use rather than relying on a configuration
flag. Every successful candidate report contains:

| New group | CUDA calls per run | Rows per run | Fallbacks |
|---|---:|---:|---:|
| non-mandatory destination | 6 | 1,764,152 chooser-alternative rows | 0 |
| non-mandatory scheduling | 7 | 75,428 choosers / 9,250,836 alternatives | 0 |
| primary tour mode | 9 | 160,479 tours | 0 |

Those 22 new calls pass in each of the three candidate processes. The existing
mandatory and trip proof gates also remain active.

## Component results

The following table reports the median timer for every configured model step.
ActivitySim rounds these values to tenths of a second. Small movements in
untargeted rows are run noise and are not GPU speedup claims.

| Component | Regular | Phase 33 | Saved | Reduction | Speedup | GPU role |
|---|---:|---:|---:|---:|---:|---|
| initialize_landuse | 16.1 | 16.2 | -0.1 | -0.62% | 0.994x | untargeted |
| initialize_households | 4.9 | 4.9 | 0.0 | 0.00% | 1.000x | untargeted |
| compute_accessibility | 1.7 | 1.8 | -0.1 | -5.88% | 0.944x | untargeted |
| school_location | 9.1 | 9.1 | 0.0 | 0.00% | 1.000x | untargeted |
| workplace_location | 13.5 | 13.5 | 0.0 | 0.00% | 1.000x | untargeted |
| auto_ownership_simulate | 0.8 | 0.9 | -0.1 | -12.50% | 0.889x | untargeted |
| free_parking | 1.0 | 1.0 | 0.0 | 0.00% | 1.000x | untargeted |
| cdap_simulate | 6.3 | 6.3 | 0.0 | 0.00% | 1.000x | untargeted |
| mandatory_tour_frequency | 1.4 | 1.4 | 0.0 | 0.00% | 1.000x | untargeted |
| mandatory_tour_scheduling | 24.3 | 14.7 | 9.6 | 39.51% | **1.653x** | Phase 32 native GPU retained |
| joint_tour_frequency | 1.3 | 1.2 | 0.1 | 7.69% | 1.083x | untargeted |
| joint_tour_composition | 0.8 | 0.7 | 0.1 | 12.50% | 1.143x | untargeted |
| joint_tour_participation | 2.3 | 2.3 | 0.0 | 0.00% | 1.000x | untargeted |
| joint_tour_destination | 3.2 | 3.2 | 0.0 | 0.00% | 1.000x | untargeted |
| joint_tour_scheduling | 1.4 | 1.3 | 0.1 | 7.14% | 1.077x | untargeted |
| non_mandatory_tour_frequency | 3.4 | 3.4 | 0.0 | 0.00% | 1.000x | untargeted |
| non_mandatory_tour_destination | 13.8 | 11.8 | 2.0 | 14.49% | **1.169x** | Phase 33 generated GPU |
| non_mandatory_tour_scheduling | 9.8 | 6.6 | 3.2 | 32.65% | **1.485x** | Phase 33 scheduling GPU |
| tour_mode_choice_simulate | 5.4 | 5.3 | 0.1 | 1.85% | 1.019x | Phase 33 generated GPU |
| atwork_subtour_frequency | 1.0 | 1.0 | 0.0 | 0.00% | 1.000x | untargeted |
| atwork_subtour_destination | 4.4 | 4.1 | 0.3 | 6.82% | 1.073x | untargeted |
| atwork_subtour_scheduling | 1.6 | 1.6 | 0.0 | 0.00% | 1.000x | untargeted |
| atwork_subtour_mode_choice | 1.1 | 1.1 | 0.0 | 0.00% | 1.000x | untargeted |
| stop_frequency | 3.5 | 3.6 | -0.1 | -2.86% | 0.972x | untargeted |
| trip_purpose | 1.1 | 1.1 | 0.0 | 0.00% | 1.000x | untargeted |
| trip_destination | 41.0 | 27.0 | 14.0 | 34.15% | **1.519x** | Phase 17 generated GPU retained |
| trip_purpose_and_destination | 0.6 | 0.7 | -0.1 | -16.67% | 0.857x | untargeted |
| trip_scheduling | 7.2 | 6.8 | 0.4 | 5.56% | 1.059x | untargeted; no causal GPU claim |
| trip_mode_choice | 9.6 | 10.1 | -0.5 | -5.21% | 0.950x | Phase 17 generated GPU retained |
| write_data_dictionary | 1.7 | 1.7 | 0.0 | 0.00% | 1.000x | untargeted |
| track_skim_usage | 0.5 | 0.4 | 0.1 | 20.00% | 1.250x | untargeted |
| write_trip_matrices | 6.7 | 6.6 | 0.1 | 1.49% | 1.015x | untargeted |
| write_tables | 1.8 | 1.8 | 0.0 | 0.00% | 1.000x | untargeted |
| summarize | 4.6 | 4.6 | 0.0 | 0.00% | 1.000x | untargeted |

The three new target medians save 5.3 seconds in their own component timers.
Tour mode is effectively neutral and trip mode is 0.5 seconds slower, so the
report does not hide overhead that remains. The 29.6-second whole-model median
saving is not obtained by adding independently rounded component medians.

## Replication result

After each candidate exits, an independent verifier compares every published
`final_*.csv` with that pair's fresh regular ActivitySim output. It first checks
file names, row identities, column identities, and missing-value positions.
All decision and substantive columns must be exact. Only two named
diagnostic columns may differ numerically within predeclared bounds.

All three pairs produced:

- zero changed modeled decision cells and zero changed decision rows;
- five final tables byte-identical;
- only `destination_logsum` and `mode_choice_logsum` text differences in tours
  or trips;
- maximum destination-logsum error of 0.0000100000000032 against a 0.0001 gate;
- maximum mode-choice-logsum error of 0.0000038135214631 against a 0.00001 gate;
- all 34 model steps completed; and
- zero CUDA fallbacks.

The diagnostics are intentionally described as bounded, not bit-identical.
They do not alter a modeled destination, time, mode, tour, or trip in this
benchmark. This is benchmark replication, not a universal mathematical proof
for every possible coefficient set or scenario.

## Assumptions and boundaries

The result assumes the checked-in software lock, public Prototype MTC extended
configuration, 50,000 sampled households, all 1,454 zones, and the local RTX
A4000. The ordinary ActivitySim control is optimized Sharrow, not a naive CPU
loop. Both sides execute identical model steps and write complete outputs.

CPU orchestration remains. ActivitySim still builds some chooser tables,
feasible alternatives, stateful timetable inputs, random draws, and output
tables. Destination and mode utilities return compact results to ActivitySim.
Therefore Phase 33 is a faster hybrid runtime, not a CPU-free travel model.

One attempted third candidate ended with Windows access violation
`-1073741819` during ActivitySim's CPU CDAP step, before any Phase 33 kernel or
kernel report. OpenBLAS had emitted its thread-metadata warning, but the precise
native cause is not proven. The incomplete output and logs were archived, never
counted, and the candidate was rerun from a fresh process against the already
completed third control. The retry passed all performance and exactness gates.
The runner now has an explicit `-Resume` mode that reuses only pairs with a
complete timing log, candidate proof, and exactness report; partial outputs
must be archived first.

## Reproduction

```powershell
.\scripts\run_phase33_full_model_ab.ps1 `
  -Repetitions 3 `
  -Households 50000 `
  -RunTag p33model1 `
  -Baseline activitysim
```

The runner refuses to overwrite artifacts. After an externally interrupted or
native-crashed process, archive the incomplete output and use the same command
with `-Resume`. Completed pairs are checked and reused; incomplete work is not
silently accepted.

Primary evidence:

- [`phase33-p33model1-summary.json`](../benchmark-results/phase33-p33model1-summary.json)
- the three `phase33-p33model1-gpu-*.json` live proof reports; and
- the three `phase33-p33model1-exact-*.json` whole-output comparisons;
- [`phase33-p33model1-failure-audit.json`](../benchmark-results/phase33-p33model1-failure-audit.json),
  which records the excluded native crash and hashes its locally archived logs.

## What should come next

The next phase should be one large location-choice and trip-chain expansion,
not a tiny kernel demonstration. School location, workplace location, joint and
at-work destination/mode work, and the sequential trip scheduler are the main
remaining candidates. The best architecture is a device-resident person-tour-
trip state table plus a shared location-choice service, while ActivitySim keeps
its public configuration and output contract.

The acceptance gate should remain ambitious: three full matched pairs, a clear
whole-model win, zero decision changes, bounded declared diagnostics, no silent
fallback, changed-scenario tests, and memory accounting for one authoritative
network representation. A shared CPU/Sharrow/CUDA arithmetic contract is still
needed before benchmark-qualified boundary safeguards can become universal.
