# Phase 61: complete CPU/accelerated comparison

Six balanced Phase60/61 pairs; two fresh regular CPU runs each at 1 and 48 Numba threads, separately measured; BLAS/OpenMP one thread. Not a strongest-possible CPU rewrite. Exact output contracts unchanged. Hybrid improvements, not GPU-only attribution.

Separate medians; component clocks are rounded to tenths and row medians need not sum to total medians.

| Clock | CPU 1 | CPU 48 | Phase 60 | Phase 61 | CPU 48 / Phase 61 |
|---|---:|---:|---:|---:|---:|
| process_wall_seconds | 296.52 | 201.17 | 91.33 | 79.22 | 2.539x |
| charged_total_seconds | 290.65 | 195.30 | 84.67 | 72.82 | 2.682x |

## All 34 model steps

| Step | CPU 1 (s) | CPU 48 (s) | Phase 60 (s) | Phase 61 (s) | CPU 48 / Phase 61 |
|---|---:|---:|---:|---:|---:|
| initialize_landuse | 13.80 | 13.60 | 0.60 | 0.60 | 22.667x |
| initialize_households | 4.55 | 4.70 | 4.40 | 4.40 | 1.068x |
| compute_accessibility | 1.55 | 1.60 | 1.40 | 1.40 | 1.143x |
| school_location | 14.55 | 9.05 | 3.70 | 3.60 | 2.514x |
| workplace_location | 29.50 | 13.10 | 1.70 | 1.70 | 7.706x |
| auto_ownership_simulate | 0.70 | 0.75 | 1.15 | 1.00 | 0.750x |
| free_parking | 0.90 | 0.90 | 0.70 | 0.70 | 1.286x |
| cdap_simulate | 6.50 | 6.70 | 5.00 | 4.55 | 1.473x |
| mandatory_tour_frequency | 1.50 | 1.30 | 1.25 | 1.35 | 0.963x |
| mandatory_tour_scheduling | 37.55 | 23.85 | 8.70 | 7.50 | 3.180x |
| joint_tour_frequency | 1.15 | 1.10 | 1.05 | 0.90 | 1.222x |
| joint_tour_composition | 0.60 | 0.60 | 0.40 | 0.40 | 1.500x |
| joint_tour_participation | 2.00 | 2.10 | 1.90 | 2.10 | 1.000x |
| joint_tour_destination | 3.05 | 3.10 | 1.60 | 1.65 | 1.879x |
| joint_tour_scheduling | 1.60 | 1.20 | 1.25 | 1.00 | 1.200x |
| non_mandatory_tour_frequency | 6.85 | 3.30 | 3.00 | 2.40 | 1.375x |
| non_mandatory_tour_destination | 29.75 | 12.65 | 1.80 | 1.70 | 7.441x |
| non_mandatory_tour_scheduling | 17.65 | 9.25 | 5.00 | 3.30 | 2.803x |
| tour_mode_choice_simulate | 5.40 | 5.10 | 5.35 | 3.25 | 1.569x |
| atwork_subtour_frequency | 0.90 | 0.90 | 0.80 | 0.60 | 1.500x |
| atwork_subtour_destination | 7.30 | 3.90 | 1.10 | 0.75 | 5.200x |
| atwork_subtour_scheduling | 1.50 | 1.50 | 1.40 | 1.10 | 1.364x |
| atwork_subtour_mode_choice | 0.95 | 0.90 | 0.80 | 0.70 | 1.286x |
| stop_frequency | 3.20 | 3.45 | 3.10 | 2.90 | 1.190x |
| trip_purpose | 1.00 | 0.95 | 0.80 | 0.85 | 1.118x |
| trip_destination | 65.00 | 39.35 | 8.60 | 7.90 | 4.981x |
| trip_purpose_and_destination | 0.50 | 0.45 | 0.35 | 0.35 | 1.286x |
| trip_scheduling | 6.55 | 6.60 | 2.00 | 1.20 | 5.500x |
| trip_mode_choice | 10.25 | 9.10 | 5.65 | 3.90 | 2.333x |
| write_data_dictionary | 1.40 | 1.40 | 1.30 | 1.30 | 1.077x |
| track_skim_usage | 0.30 | 0.30 | 0.20 | 0.20 | 1.500x |
| write_trip_matrices | 6.50 | 6.40 | 2.40 | 2.40 | 2.667x |
| write_tables | 1.70 | 1.70 | 1.50 | 1.55 | 1.097x |
| summarize | 4.45 | 4.45 | 3.90 | 2.60 | 1.712x |

## Strong equivalent CPU primitive controls

These are isolated calculations on captured live inputs, not full-component or whole-model ratios. Greater than one favors GPU; less than one favors CPU.

- Timetable, encoding and transfers included: CPU/GPU 0.551x.
- Tour-mode reducer, resident inputs: CPU/GPU 0.811x; including transfers: 0.502x.
- Standard normals are a compiled CPU batching improvement, verified bit for bit against original NumPy on live seeded streams; not a GPU result.

The full pipeline retains GPU utility evaluation and live CPU numerical-boundary checks. Independent output audits are outside the clocks; execution, validation/prewarming, reports and process overhead remain charged. CPU 48 denotes Numba capacity, not 48 threads in every library.
