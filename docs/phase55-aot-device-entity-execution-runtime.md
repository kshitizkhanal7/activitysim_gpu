# Phase 55: reviewed AOT device-entity execution runtime

## Outcome

Phase 55 is qualified on the public Prototype MTC Extended 50,000-household
benchmark and the NVIDIA RTX A4000. It improves the already GPU-accelerated
Phase 54 runtime while preserving exact modeled decisions, bounded diagnostic
logsums, and zero fallback.

The final matched three-pair experiment produced:

| Boundary | Phase 54 median | Phase 55 median | Result |
|---|---:|---:|---:|
| 19-call monolithic destination service | not separately reconstructed | 2.249 s | below 2.5 s target |
| five destination components | 10.7 s | 10.1 s | 1.059x, 5.61% lower |
| complete 34-step model | 138.2 s | 135.909 s | 1.017x, 1.66% lower |

Phase 55 won all three full-model pairs and all three five-component pairs.
All three output verification files passed. The five-component median is 0.1
seconds above the under-10-second stretch target, and the full model remains
5.909 seconds above the under-130-second stretch target. Those targets are not
claimed as passed.

The six final cold-process shard runs also compare the instrumented destination
service against Phase 54. Their medians are 3.713 and 2.737 seconds, a 1.356x
speedup. The two-shard number includes two independent process startup costs;
the monolithic full run pays startup once and has a 2.249-second median.

## What changed

### Reviewed plan atlas

The runtime no longer reparses the ten public destination utility programs and
rebuilds their native ABI at every process start. A developer-only capture run
resolves each specification, coefficients, scalar constants, bindings, and
skim sources. It writes a typed JSON plan atlas with a self-digest. Production
mode:

1. verifies the atlas SHA-256;
2. verifies the current public config-file fingerprints;
3. requires the exact purpose/specification key;
4. materializes compact arrays and skim bindings without expression parsing;
5. stops on an absent or changed plan.

The reviewed atlas contains ten plans and has payload digest
`8a26ceb258b7f891517010f1b1a59f380f146ee76d05c5c6bd27d7ac7a187ce8`.
Capture mode is opt-in and is not enabled by either production runner.

### Architecture-specific CUDA binary

The checked Phase 52 CUDA source is compiled ahead of time to an `sm_86` cubin
for this RTX A4000. Production verifies:

- source SHA-256 `599a9704be0992d2863320390cbca0028c7a578ecacf72d69de2e658a5d79906`;
- cubin SHA-256 `9872bc0c9031f5ab29d99065db960652ff5fa5ffdb8733e0f0c8d5e1008d7f06`;
- declared kernel symbol and byte count;
- the active GPU architecture.

The production path loads the reviewed binary and never asks NVRTC to compile
the fused destination kernel. A different GPU architecture must receive and
qualify its own binary; Phase 55 deliberately fails closed instead of silently
recompiling.

### Compact native ABI materialization

`export_native_aot_plan` records data-only bindings, coefficients, scalar
arrays, row-source order, and manifest metadata. `materialize_native_aot_plan`
reconstructs the minimal resident invocation. It allocates only one bootstrap
row because the fused kernel consumes compact owner data directly.

Immutable skim cubes are resolved once per network object and source, then
shared across all ten plans. This removed a repeated first-use lookup from the
AOT materializer and was essential to the final service result.

### CUDA sample compaction and direct lease publication

The destination sampler now sorts each chooser's unique sampled alternatives
and compacts probabilities, counts, and draw positions on CUDA. It publishes
the compact device offsets and destination buffer directly to the existing
versioned lease service. The logsum runtime consumes those same buffers, so no
second host-to-device destination upload occurs.

This kernel preserves ActivitySim's chooser-major, ascending-alternative sample
order. Guarded rows are adjudicated by the existing exact NumPy reference and
written back before compaction. The production path rejects development-only
diagnostics that would require an unreviewed data route.

The compactor is not independently faster in this compatibility runtime. Its
final median sampler time is 2.584 seconds versus 2.182 seconds for Phase 54.
ActivitySim still requires a host pandas sample table, and that publication
cost limits the kernel. The aggregate destination path wins because AOT plan
loading and downstream buffer reuse outweigh this local cost. A future runtime
must remove the host DataFrame contract to turn compaction into a standalone
speedup.

## Replication design

Two independent designs are retained:

- Six cold shards: three Phase 54/55 comparisons covering seven early and
  twelve downstream calls. Together each pair covers 19 calls, 201,390 owners,
  and 4,696,676 sampled rows.
- Three fresh-process full pairs: Phase 54 control followed by Phase 55,
  running all 34 model steps. Temporary pipelines are deleted only after their
  timing CSV, runtime report, and exact verification are preserved.

Every final full candidate reports 19 atlas hits, 19 code-generation bypasses,
19 CUDA compactions, and zero fallback. Every inherited proof gate passes.

The machine-readable acceptance artifact is
`benchmark-results/phase55-p55final3-qualification.json`. Raw full-pair evidence
uses the `phase55-p55final3-*` prefix. Final-code shard evidence uses
`phase55-p55final2-*`; its Phase 54 controls are the previously qualified
`phase54-p54final-*` reports.

## Assumptions and claim boundary

The proof is limited to this public configuration, 50,000 households, 1,454
zones, ten reviewed plan keys, the 315-term/21-alternative destination-logsum
program, current random-stream policy, current sample ordering, unchunked
calls, and `sm_86` hardware. The atlas fingerprints configuration files, but it
does not make arbitrary ActivitySim expressions safe.

Modeled decision columns are exact. Floating diagnostics are required to stay
inside their predeclared limits; bit-identical values are not claimed for every
intermediate CPU/GPU operation. Timing is specific to this machine and current
software environment.

Phase 55 is a replicated incremental gain over Phase 54. It is not a GPU-only
ActivitySim implementation. Initialization, orchestration, pandas publication,
and many model components remain CPU work.

## Next major opportunity

Phase 56 should eliminate the host sample-table compatibility boundary rather
than add another isolated kernel. A device-native entity/sample store should
carry owner IDs, offsets, chosen destinations, probabilities, counts, and
random-state references directly through sampling, logsums, probability, and
final choice. Host materialization should happen once at the model publication
boundary.

Its acceptance criteria should include:

- a general versioned device-table schema rather than ten public-only keys;
- architecture-portable, reproducibly built binary bundles for supported GPUs;
- a sampler that is itself faster than Phase 54 in all three pairs;
- five destination components below 10 seconds median;
- full-model median below 130 seconds;
- exact decisions, bounded diagnostics, zero fallback, and three matched wins.
