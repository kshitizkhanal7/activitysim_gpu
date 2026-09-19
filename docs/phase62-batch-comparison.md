# Phase 62: complete repeated-scenario comparison

Two repetitions per strategy, four-strategy order reversed in repetition two; full A-B-A outputs independently audited. All setup, fresh private input copies and scenario input hashing included. One machine, fixed public network and two seeds, not a universal isolation or performance guarantee.

Seconds for all three scenarios, including the first run and process setup. These are batch totals, not single-run times.

| Engine | Three fresh processes | One persistent process | Fresh / persistent |
|---|---:|---:|---:|
| Phase 62 hybrid | 235.65 | 212.34 | 1.110x |
| Regular CPU / 48 threads | 594.42 | 545.74 | 1.089x |

Persistent CPU / persistent hybrid: 2.570x.

The selected worker reuses programs and private raw-input snapshots, not modeled decisions. Numeric input tables are deep-copied for each scenario. GPU skim content caching is implemented and tested but disabled in the selected path because its cost and cache misses outweighed its benefit in development. Fresh single-process results and the 70-second target are reported separately.
