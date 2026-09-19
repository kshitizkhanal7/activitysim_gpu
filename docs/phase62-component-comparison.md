# Phase 62: complete CPU/accelerated comparison

Six balanced Phase61/62 pairs; two fresh regular CPU runs each at 1 and 48 Numba threads, separately measured; BLAS/OpenMP one thread. Not a strongest-possible CPU rewrite. Exact output contracts unchanged. Hybrid improvements, not GPU-only attribution.

Separate medians; component clocks are rounded to tenths and row medians need not sum to total medians.

| Clock | CPU 1 | CPU 48 | Phase 61 | Phase 62 | CPU 48 / Phase 62 |
|---|---:|---:|---:|---:|---:|
| process_wall_seconds | 299.83 | 204.84 | 80.63 | 76.81 | 2.667x |
| charged_total_seconds | 293.35 | 198.70 | 73.83 | 70.37 | 2.824x |

## All 34 model steps

| Step | CPU 1 (s) | CPU 48 (s) | Phase 61 (s) | Phase 62 (s) | CPU 48 / Phase 62 |
|---|---:|---:|---:|---:|---:|
| initialize_landuse | 13.95 | 14.30 | 0.60 | 0.70 | 20.429x |
| initialize_households | 4.70 | 4.95 | 4.40 | 4.40 | 1.125x |
| compute_accessibility | 1.60 | 1.60 | 1.50 | 1.50 | 1.067x |
| school_location | 14.75 | 9.25 | 3.60 | 3.60 | 2.569x |
| workplace_location | 29.55 | 13.20 | 1.75 | 1.80 | 7.333x |
| auto_ownership_simulate | 0.80 | 0.80 | 1.10 | 1.10 | 0.727x |
| free_parking | 0.90 | 0.90 | 0.70 | 0.70 | 1.286x |
| cdap_simulate | 6.70 | 6.75 | 4.85 | 3.55 | 1.901x |
| mandatory_tour_frequency | 1.55 | 1.45 | 1.50 | 1.30 | 1.115x |
| mandatory_tour_scheduling | 37.45 | 23.70 | 7.50 | 6.90 | 3.435x |
| joint_tour_frequency | 1.20 | 1.25 | 1.00 | 0.90 | 1.389x |
| joint_tour_composition | 0.60 | 0.70 | 0.50 | 0.40 | 1.750x |
| joint_tour_participation | 2.15 | 2.10 | 2.15 | 2.10 | 1.000x |
| joint_tour_destination | 3.10 | 3.15 | 1.60 | 1.65 | 1.909x |
| joint_tour_scheduling | 1.70 | 1.40 | 1.00 | 0.95 | 1.474x |
| non_mandatory_tour_frequency | 7.00 | 3.30 | 2.50 | 2.30 | 1.435x |
| non_mandatory_tour_destination | 30.05 | 12.70 | 1.75 | 1.80 | 7.056x |
| non_mandatory_tour_scheduling | 17.80 | 9.50 | 3.30 | 3.30 | 2.879x |
| tour_mode_choice_simulate | 5.50 | 5.20 | 3.20 | 3.15 | 1.651x |
| atwork_subtour_frequency | 0.95 | 1.00 | 0.60 | 0.80 | 1.250x |
| atwork_subtour_destination | 7.45 | 3.95 | 0.90 | 0.80 | 4.937x |
| atwork_subtour_scheduling | 1.60 | 1.50 | 1.20 | 1.10 | 1.364x |
| atwork_subtour_mode_choice | 1.00 | 1.10 | 0.75 | 0.70 | 1.571x |
| stop_frequency | 3.30 | 3.60 | 2.95 | 2.50 | 1.440x |
| trip_purpose | 1.00 | 1.00 | 0.90 | 0.85 | 1.176x |
| trip_destination | 64.70 | 39.45 | 7.90 | 7.55 | 5.225x |
| trip_purpose_and_destination | 0.50 | 0.50 | 0.40 | 0.40 | 1.250x |
| trip_scheduling | 6.65 | 6.70 | 1.20 | 1.20 | 5.583x |
| trip_mode_choice | 10.45 | 9.20 | 4.05 | 3.90 | 2.359x |
| write_data_dictionary | 1.55 | 1.50 | 1.30 | 1.30 | 1.154x |
| track_skim_usage | 0.40 | 0.40 | 0.20 | 0.20 | 2.000x |
| write_trip_matrices | 6.60 | 6.45 | 2.40 | 2.40 | 2.688x |
| write_tables | 1.75 | 1.75 | 1.60 | 1.60 | 1.094x |
| summarize | 4.40 | 4.40 | 2.55 | 2.60 | 1.692x |

## Strong equivalent CPU primitive controls

These are isolated calculations on captured live inputs, not full-component or whole-model ratios. Greater than one favors GPU; less than one favors CPU.

- Timetable, encoding and transfers included: CPU/GPU 0.551x.
- Tour-mode reducer, resident inputs: CPU/GPU 0.811x; including transfers: 0.502x.
- Standard normals are a compiled CPU batching improvement, verified bit for bit against original NumPy on live seeded streams; not a GPU result.

The full pipeline retains GPU utility evaluation and live CPU numerical-boundary checks. Independent output audits are outside the clocks; execution, validation/prewarming, reports and process overhead remain charged. CPU 48 denotes Numba capacity, not 48 threads in every library.
