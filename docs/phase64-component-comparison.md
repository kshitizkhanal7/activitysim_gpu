# Phase 64: complete-model and component comparisons

Qualified evidence: `phase64-p64f1-qualification.json`. Six reversed-order
Phase 63/64 pairs, two replacement ordinary CPU controls at 48-thread capacity
and explicit PASSIVE wait policy. All use the same 50,000-household public model.
The original two unspecified-policy CPU controls are retained but excluded.

## Complete systems

| Fresh process | Median elapsed seconds | Observations |
|---|---:|---:|
| Ordinary CPU ActivitySim | 208.93 | 2 |
| Previous hybrid, Phase 63 | 76.17 | 6 |
| Latest hybrid, Phase 64 | 74.95 | 6 |

The hybrid is **2.787x** faster than ordinary CPU, or **64.12% less elapsed time**.
Phase 64 reduces the paired previous-version median by **1.59%**. Every pair
improves both elapsed and charged time. Candidate elapsed times span 74.04-75.76
seconds. The under-70-second and 65-second stretch targets are **not met**.

Separate matched-preparation fresh-worker clocks include input hashing and
reset checks: CPU **206.83 seconds**, hybrid **77.44**, or **2.671x**, with four
processes per engine. Both get the same expression and verified raw-input
preparation. Do not combine that clock with the fresh-process rows above.

## All 34 components

Seconds are medians of ActivitySim's recorded component clocks, which have
limited precision. Ratios for very short steps are especially coarse. Component
medians do not sum to the median complete-process clock: startup and external
runtime validation differ, and the median of a sum need not equal a sum of
medians. These are hybrid-system component ratios, **not GPU-kernel ratios**.
Input preparation, CPU algorithms and output-writing improvements are included.

| Component | CPU s | Phase 63 s | Phase 64 s | CPU / Phase 64 |
|---|---:|---:|---:|---:|
| initialize_landuse | 16.80 | 0.60 | 0.60 | 28.00x |
| initialize_households | 5.10 | 4.50 | 3.05 | 1.67x |
| compute_accessibility | 1.80 | 1.50 | 1.50 | 1.20x |
| school_location | 9.10 | 3.90 | 3.90 | 2.33x |
| workplace_location | 13.00 | 1.70 | 2.10 | 6.19x |
| auto_ownership_simulate | 0.80 | 1.00 | 1.10 | 0.73x |
| free_parking | 1.00 | 0.70 | 0.70 | 1.43x |
| cdap_simulate | 6.85 | 3.45 | 3.40 | 2.01x |
| mandatory_tour_frequency | 1.40 | 1.15 | 1.00 | 1.40x |
| mandatory_tour_scheduling | 23.45 | 6.05 | 5.70 | 4.11x |
| joint_tour_frequency | 1.25 | 1.00 | 0.80 | 1.56x |
| joint_tour_composition | 0.70 | 0.40 | 0.40 | 1.75x |
| joint_tour_participation | 2.20 | 2.05 | 2.00 | 1.10x |
| joint_tour_destination | 2.85 | 1.75 | 1.60 | 1.78x |
| joint_tour_scheduling | 1.40 | 0.90 | 0.90 | 1.56x |
| non_mandatory_tour_frequency | 3.25 | 2.20 | 2.20 | 1.48x |
| non_mandatory_tour_destination | 12.35 | 1.70 | 2.20 | 5.61x |
| non_mandatory_tour_scheduling | 9.60 | 3.20 | 3.25 | 2.95x |
| tour_mode_choice_simulate | 4.95 | 3.00 | 2.95 | 1.68x |
| atwork_subtour_frequency | 1.00 | 0.60 | 0.60 | 1.67x |
| atwork_subtour_destination | 4.00 | 0.80 | 1.45 | 2.76x |
| atwork_subtour_scheduling | 1.50 | 1.10 | 1.10 | 1.36x |
| atwork_subtour_mode_choice | 1.00 | 0.65 | 0.70 | 1.43x |
| stop_frequency | 3.65 | 2.40 | 2.40 | 1.52x |
| trip_purpose | 1.05 | 0.70 | 0.75 | 1.40x |
| trip_destination | 38.50 | 6.95 | 6.70 | 5.75x |
| trip_purpose_and_destination | 0.60 | 0.30 | 0.30 | 2.00x |
| trip_scheduling | 7.25 | 1.10 | 1.10 | 6.59x |
| trip_mode_choice | 10.05 | 4.10 | 3.85 | 2.61x |
| write_data_dictionary | 1.85 | 1.30 | 1.30 | 1.42x |
| track_skim_usage | 0.50 | 0.20 | 0.20 | 2.50x |
| write_trip_matrices | 6.70 | 2.40 | 2.40 | 2.79x |
| write_tables | 1.85 | 1.50 | 1.55 | 1.19x |
| summarize | 4.75 | 2.60 | 2.60 | 1.83x |

Some steps regress against the previous hybrid, including destination work
where the stronger CPU numerical safeguard now executes. Auto ownership is
slower than the ordinary CPU component in this campaign. Do not hide these
rows behind the total speedup or attribute every small difference to a code
change; the component clocks and finite sample sizes limit such conclusions.

## Larger complete models and isolated GPU contribution

Final-code monitored repeats pass at 100,000 households in **109.63 seconds**
and 250,000 in **299.09 seconds**, against retained independent CPU references.
They are single feasibility/output rechecks, not fresh paired CPU speed ratios.
Peak sampled individual-process host RSS is 11.37 and 20.06 billion bytes.

On 19 real mode-reduction batches (603,161 rows), the strongest tested CPU uses
48 threads. GPU-resident reduction is **1.4225x** faster; including transfers
gives a CPU/GPU ratio of **0.9262**, so the GPU loses that comparison. Resident
timing still includes small coefficient transfers and a scalar validity check.

Two reversed whole-model reducer-replacement pairs take median **74.86 seconds**
with GPU reduction and **73.75** with CPU reduction. All outputs pass, but this
does not establish a full-model benefit from that particular GPU reducer.
It retains GPU utility generation and RNG and is not ordinary CPU ActivitySim.
The full-system speedup must not be presented as pure GPU-hardware attribution.

Both ten-scenario hybrid sequences pass: zero active default-pool GPU arrays
before invocations and after final reset; fifth-to-tenth USS growth 174.45 and
188.35 MiB. This is finite sequential qualification, not a leak-free-forever
proof. No new persistent CPU comparison or independent-hardware replication
was performed in Phase 64.
