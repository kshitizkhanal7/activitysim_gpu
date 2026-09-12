# Phase 59: complete CPU/accelerated timing comparison

warm-cache fresh processes on one workstation; 6 balanced Phase58/59 pairs, then 2 unpaired regular CPU controls; not strongest possible CPU rewrite

Seconds; medians are computed separately for each row. Upstream step timings are rounded to tenths of a second; displayed extra decimal places do not imply finer precision. Component medians do not necessarily add to the full-model median.

| Clock | Regular CPU | Phase 58 hybrid | Phase 59 hybrid | CPU / Phase 59 |
|---|---:|---:|---:|---:|
| process_wall_seconds | 314.901 | 112.515 | 109.839 | 2.867x |
| charged_total_seconds | 308.400 | 104.819 | 102.689 | 3.003x |

## Every model step

| Step | Regular CPU (s) | Phase 58 (s) | Phase 59 (s) | CPU / Phase 59 |
|---|---:|---:|---:|---:|
| initialize_landuse | 14.650 | 0.600 | 0.700 | 20.929x |
| initialize_households | 5.100 | 4.400 | 4.400 | 1.159x |
| compute_accessibility | 1.700 | 1.600 | 1.550 | 1.097x |
| school_location | 15.400 | 4.050 | 4.200 | 3.667x |
| workplace_location | 30.750 | 2.000 | 2.000 | 15.375x |
| auto_ownership_simulate | 0.850 | 1.400 | 1.400 | 0.607x |
| free_parking | 1.000 | 1.000 | 1.000 | 1.000x |
| cdap_simulate | 6.950 | 7.000 | 6.900 | 1.007x |
| mandatory_tour_frequency | 1.500 | 1.500 | 1.500 | 1.000x |
| mandatory_tour_scheduling | 39.050 | 7.150 | 10.800 | 3.616x |
| joint_tour_frequency | 1.300 | 1.200 | 1.400 | 0.929x |
| joint_tour_composition | 0.700 | 0.700 | 0.700 | 1.000x |
| joint_tour_participation | 2.200 | 2.150 | 2.150 | 1.023x |
| joint_tour_destination | 3.250 | 2.050 | 2.000 | 1.625x |
| joint_tour_scheduling | 1.600 | 1.600 | 1.500 | 1.067x |
| non_mandatory_tour_frequency | 7.300 | 7.300 | 7.100 | 1.028x |
| non_mandatory_tour_destination | 31.750 | 2.000 | 2.000 | 15.875x |
| non_mandatory_tour_scheduling | 18.800 | 6.800 | 6.600 | 2.848x |
| tour_mode_choice_simulate | 5.850 | 5.500 | 5.800 | 1.009x |
| atwork_subtour_frequency | 1.050 | 1.000 | 1.000 | 1.050x |
| atwork_subtour_destination | 7.700 | 1.400 | 1.350 | 5.704x |
| atwork_subtour_scheduling | 1.850 | 1.700 | 1.700 | 1.088x |
| atwork_subtour_mode_choice | 1.100 | 1.100 | 1.100 | 1.000x |
| stop_frequency | 3.600 | 3.550 | 3.500 | 1.029x |
| trip_purpose | 1.150 | 1.100 | 1.100 | 1.045x |
| trip_destination | 68.000 | 9.000 | 8.900 | 7.640x |
| trip_purpose_and_destination | 0.600 | 0.600 | 0.600 | 1.000x |
| trip_scheduling | 7.000 | 3.700 | 2.300 | 3.043x |
| trip_mode_choice | 10.950 | 6.200 | 5.900 | 1.856x |
| write_data_dictionary | 1.650 | 1.500 | 1.500 | 1.100x |
| track_skim_usage | 0.450 | 0.400 | 0.400 | 1.125x |
| write_trip_matrices | 6.850 | 6.600 | 2.500 | 2.740x |
| write_tables | 1.950 | 1.800 | 1.800 | 1.083x |
| summarize | 4.800 | 4.600 | 4.700 | 1.021x |

Individual step ratios are descriptive, not separately randomized component benchmarks. Unchanged CPU steps can differ because of timing noise. Full-model improvement combines GPU kernels, CPU algorithm changes, data handling and output compression. The CPU baseline is pinned regular single-process ActivitySim with numerical libraries limited to one thread, not a purpose-built optimized CPU model. The Phase 59 matrix writer uses four CPU compression workers.

Charged time adds recorded validation/prewarming to the model steps. Launch-to-exit also includes process overhead. Independent output verification is outside these clocks. All modeled decisions and matrix values pass; published summary report values pass the declared exact-value contract. Diagnostic logsums retain explicit bounds.
