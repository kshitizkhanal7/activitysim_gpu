# Phase 63: complete CPU/hybrid comparison

Fresh warm-installed workstation processes; not a clean install or GPU-only arithmetic claim. CPU preparation equivalence is separately tested in the 80-model sequence.

Fresh wall clock includes process startup, complete model work, prewarming and live validation. External output audits are outside that clock.

| Clock | Regular CPU48 | Phase 62 hybrid | Phase 63 hybrid |
|---|---:|---:|---:|
| Complete process seconds | 202.82 | 82.19 | 73.94 |
| Charged service seconds | 196.95 | 75.25 | 67.15 |

Regular CPU48 / latest hybrid: 2.743x. Previous / latest hybrid: 1.112x.
Under-70-second fresh target: NOT met. Under-65 stretch: NOT met.

| Model component | CPU48 seconds | Phase 62 seconds | Phase 63 seconds | CPU / latest |
|---|---:|---:|---:|---:|
| initialize_landuse | 13.70 | 0.70 | 0.60 | 22.83x |
| initialize_households | 4.70 | 4.40 | 4.40 | 1.07x |
| compute_accessibility | 1.60 | 1.50 | 1.50 | 1.07x |
| school_location | 9.00 | 4.00 | 3.90 | 2.31x |
| workplace_location | 12.90 | 1.90 | 1.70 | 7.59x |
| auto_ownership_simulate | 0.80 | 1.20 | 1.00 | 0.80x |
| free_parking | 0.90 | 0.80 | 0.70 | 1.29x |
| cdap_simulate | 6.60 | 3.60 | 3.30 | 2.00x |
| mandatory_tour_frequency | 1.35 | 1.40 | 1.00 | 1.35x |
| mandatory_tour_scheduling | 23.75 | 7.15 | 6.05 | 3.93x |
| joint_tour_frequency | 1.10 | 1.05 | 1.00 | 1.10x |
| joint_tour_composition | 0.60 | 0.50 | 0.40 | 1.50x |
| joint_tour_participation | 2.10 | 2.35 | 2.15 | 0.98x |
| joint_tour_destination | 3.15 | 1.85 | 1.80 | 1.75x |
| joint_tour_scheduling | 1.30 | 1.10 | 0.90 | 1.44x |
| non_mandatory_tour_frequency | 3.40 | 2.40 | 2.20 | 1.55x |
| non_mandatory_tour_destination | 12.85 | 1.85 | 1.70 | 7.56x |
| non_mandatory_tour_scheduling | 9.85 | 3.40 | 3.20 | 3.08x |
| tour_mode_choice_simulate | 5.20 | 3.40 | 3.15 | 1.65x |
| atwork_subtour_frequency | 0.90 | 0.90 | 0.55 | 1.64x |
| atwork_subtour_destination | 3.90 | 0.90 | 0.70 | 5.57x |
| atwork_subtour_scheduling | 1.60 | 1.20 | 1.00 | 1.60x |
| atwork_subtour_mode_choice | 0.95 | 0.90 | 0.60 | 1.58x |
| stop_frequency | 3.50 | 2.65 | 2.40 | 1.46x |
| trip_purpose | 0.95 | 0.90 | 0.75 | 1.27x |
| trip_destination | 39.65 | 7.80 | 6.85 | 5.79x |
| trip_purpose_and_destination | 0.50 | 0.50 | 0.30 | 1.67x |
| trip_scheduling | 6.65 | 1.30 | 1.10 | 6.05x |
| trip_mode_choice | 9.25 | 4.10 | 4.05 | 2.28x |
| write_data_dictionary | 1.40 | 1.40 | 1.30 | 1.08x |
| track_skim_usage | 0.30 | 0.30 | 0.10 | 3.00x |
| write_trip_matrices | 6.50 | 2.40 | 2.20 | 2.95x |
| write_tables | 1.70 | 1.70 | 1.50 | 1.13x |
| summarize | 4.35 | 2.80 | 2.60 | 1.67x |

Component logs are rounded to 0.1 seconds before medians. Their median rows need not sum to the median full clock. A component ratio is not a standalone kernel speedup.

Phase 63 includes reusable CPU expression preparation, parsed specification copies, and RSS-only in-step memory tracing with chunking/training disabled. These are disclosed service optimizations; equivalent options are given to CPU in the separate repeated-scenario controls.

## Matched preparation: default 50,000-household scenario only

Eight fresh 50,000-household default-A processes per engine within the balanced sequence campaign. Both receive plans/files/RSS, 48-thread capacity, PASSIVE waiting, private inputs, per-scenario input hashing and reset telemetry. External output audits excluded. This clock differs from the separate fresh-pair harness.

| Engine | Fresh worker median seconds | Measured default-A processes |
|---|---:|---:|
| Regular CPU with Phase 63 preparation | 193.18 | 8 |
| Phase 63 hybrid with the same preparation options | 76.42 | 8 |

Matched-preparation CPU / hybrid: 2.528x. Neither row mixes in the smaller 10,000-household scenario.

Equal preparation does not mean identical implementations everywhere: the hybrid also contains earlier CPU-side, data-layout and output-writing optimizations. This remains a complete-system comparison, not an isolated GPU-hardware experiment.
