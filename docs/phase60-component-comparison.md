# Phase 60: all model steps and stronger CPU controls

six balanced Phase59/60 pairs; two fresh regular CPU runs each at 1 and 48 Numba threads, separately measured; BLAS/OpenMP numerical libraries remain one thread; not a strongest-possible CPU rewrite

Seconds, separate medians for each row. Upstream component clocks are rounded to tenths; extra displayed decimals are not additional measurement precision. Row medians need not sum to total medians.

| Clock | CPU 1 thread | CPU 48 threads | Phase 59 | Phase 60 | CPU 48 / Phase 60 |
|---|---:|---:|---:|---:|---:|
| process_wall_seconds | 296.18 | 201.61 | 98.42 | 90.16 | 2.236x |
| charged_total_seconds | 290.45 | 195.85 | 91.92 | 83.46 | 2.347x |

## Every model step

| Step | CPU 1 (s) | CPU 48 (s) | Phase 59 (s) | Phase 60 (s) | CPU 48 / Phase 60 |
|---|---:|---:|---:|---:|---:|
| initialize_landuse | 13.45 | 13.80 | 0.60 | 0.60 | 23.000x |
| initialize_households | 4.60 | 4.65 | 4.30 | 4.30 | 1.081x |
| compute_accessibility | 1.55 | 1.55 | 1.40 | 1.40 | 1.107x |
| school_location | 14.80 | 9.10 | 3.70 | 3.75 | 2.427x |
| workplace_location | 29.85 | 13.10 | 1.70 | 1.70 | 7.706x |
| auto_ownership_simulate | 0.70 | 0.70 | 1.20 | 1.20 | 0.583x |
| free_parking | 0.80 | 0.85 | 0.70 | 0.70 | 1.214x |
| cdap_simulate | 6.40 | 6.60 | 6.30 | 4.90 | 1.347x |
| mandatory_tour_frequency | 1.30 | 1.30 | 1.30 | 1.30 | 1.000x |
| mandatory_tour_scheduling | 37.95 | 23.70 | 10.30 | 8.80 | 2.693x |
| joint_tour_frequency | 1.10 | 1.10 | 1.00 | 1.00 | 1.100x |
| joint_tour_composition | 0.55 | 0.60 | 0.40 | 0.40 | 1.500x |
| joint_tour_participation | 2.10 | 2.05 | 1.85 | 1.90 | 1.079x |
| joint_tour_destination | 3.00 | 3.05 | 1.45 | 1.70 | 1.794x |
| joint_tour_scheduling | 1.30 | 1.20 | 1.20 | 1.20 | 1.000x |
| non_mandatory_tour_frequency | 7.10 | 3.30 | 6.70 | 2.90 | 1.138x |
| non_mandatory_tour_destination | 30.25 | 13.00 | 1.75 | 1.70 | 7.647x |
| non_mandatory_tour_scheduling | 17.45 | 9.35 | 6.30 | 4.90 | 1.908x |
| tour_mode_choice_simulate | 5.35 | 5.10 | 5.40 | 5.40 | 0.944x |
| atwork_subtour_frequency | 0.80 | 0.90 | 0.70 | 0.75 | 1.200x |
| atwork_subtour_destination | 7.20 | 4.00 | 1.10 | 1.10 | 3.636x |
| atwork_subtour_scheduling | 1.80 | 1.40 | 1.40 | 1.40 | 1.000x |
| atwork_subtour_mode_choice | 0.90 | 1.00 | 0.80 | 0.80 | 1.250x |
| stop_frequency | 3.20 | 3.45 | 3.10 | 3.10 | 1.113x |
| trip_purpose | 0.90 | 0.90 | 0.80 | 0.80 | 1.125x |
| trip_destination | 64.50 | 39.50 | 8.60 | 8.75 | 4.514x |
| trip_purpose_and_destination | 0.40 | 0.45 | 0.30 | 0.30 | 1.500x |
| trip_scheduling | 6.50 | 6.60 | 2.00 | 2.05 | 3.220x |
| trip_mode_choice | 10.40 | 9.20 | 5.50 | 5.45 | 1.688x |
| write_data_dictionary | 1.40 | 1.40 | 1.30 | 1.30 | 1.077x |
| track_skim_usage | 0.30 | 0.30 | 0.20 | 0.20 | 1.500x |
| write_trip_matrices | 6.45 | 6.55 | 2.20 | 2.20 | 2.977x |
| write_tables | 1.70 | 1.70 | 1.60 | 1.50 | 1.133x |
| summarize | 4.40 | 4.40 | 4.20 | 3.85 | 1.143x |

CPU 48 means 48 Numba threads, not 48 threads in every library or 48 independent model processes. The Phase 59 control has the same 48-thread pool capacity but masks it to one throughout. Phase 60 uses 48 only in non-mandatory tour frequency and restores one afterward. Both hybrids use four CPU matrix-compression workers. No reports or model steps are removed.

The accelerated system remains hybrid. This phase improves CPU preparation and enables an existing parallel CPU calculation; it does not establish a new GPU-only kernel speedup. Independent output auditing is outside the model clocks, while actual execution, validation and process overhead remain charged as documented.
