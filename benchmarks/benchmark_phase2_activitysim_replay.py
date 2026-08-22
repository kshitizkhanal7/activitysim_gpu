"""Benchmark the lowered real-data ActivitySim scheduling replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from choiceforge.cuda_backend import CudaChoiceBackend, _cupy
from choiceforge.interaction_backend import (
    CudaInteractionBackend,
    choose_terms_numpy,
    offsets_from_ids,
)


def timed(fn, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return value, samples


def tile_batch(data, factor):
    terms = np.tile(data["terms"], (factor, 1))
    counts = np.diff(offsets_from_ids(data["chooser_ids"]))
    tiled_counts = np.tile(counts, factor)
    offsets = np.r_[0, np.cumsum(tiled_counts)].astype(np.int64)
    draws = np.tile(data["draws"], factor)
    positions = np.tile(data["positions"], factor)
    return terms, offsets, draws, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=Path("benchmark-results/phase2-replay"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/phase2-summary.json"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    manifest = json.loads((args.capture / "manifest.json").read_text(encoding="utf-8"))
    mandatory = [b for b in manifest["batches"] if b["trace_label"].startswith("mandatory_tour_scheduling")]
    if len(mandatory) != 6:
        raise RuntimeError(f"expected six mandatory batches, found {len(mandatory)}")

    # Validate every segment/tour-number batch before benchmarking the dominant one.
    gpu_terms = CudaInteractionBackend()
    gpu_exact = CudaChoiceBackend()
    all_validation = []
    for batch_meta in mandatory:
        with np.load(args.capture / batch_meta["file"]) as loaded:
            batch = {k: loaded[k] for k in loaded.files}
        batch_offsets = offsets_from_ids(batch["chooser_ids"])
        reconstructed_batch = (
            batch["terms"].astype(np.float64)
            @ batch["coefficients"].astype(np.float64)
        )
        term_result = gpu_terms.choose_from_terms(
            batch["terms"], batch["coefficients"], batch_offsets, batch["draws"]
        )
        exact_positions = gpu_exact.choose_from_probabilities(
            batch["probabilities"], batch["draws"]
        )
        error = np.abs(reconstructed_batch - batch["utilities"])
        all_validation.append(
            {
                "trace_label": batch_meta["trace_label"],
                "choosers": batch_meta["choosers"],
                "exact_probability_choice_mismatches": int(np.count_nonzero(exact_positions != batch["positions"])),
                "lowered_gpu_choice_mismatches": int(np.count_nonzero(term_result.choices != batch["positions"])),
                "lowered_utility_max_abs_error": float(error.max()),
                "lowered_utility_mean_abs_error": float(error.mean()),
            }
        )

    # The largest batch dominates runtime and retains the canonical 190-alternative shape.
    meta = max(mandatory, key=lambda b: b["interaction_rows"])
    with np.load(args.capture / meta["file"]) as loaded:
        data = {k: loaded[k] for k in loaded.files}
    offsets_native = offsets_from_ids(data["chooser_ids"])

    # Validate the captured front end and ActivitySim's exact probability/draw contract.
    reconstructed = data["terms"].astype(np.float64) @ data["coefficients"].astype(np.float64)
    utility_error = np.abs(reconstructed - data["utilities"])
    exact_gpu_positions = gpu_exact.choose_from_probabilities(
        data["probabilities"], data["draws"]
    )
    exact_choice_mismatches = int(np.count_nonzero(exact_gpu_positions != data["positions"]))

    cpu_warm = choose_terms_numpy(data["terms"], data["coefficients"], offsets_native, data["draws"])
    gpu = gpu_terms
    gpu.choose_from_terms(data["terms"], data["coefficients"], offsets_native, data["draws"])
    cp = _cupy()

    rows = []
    for scale in args.scales:
        terms, offsets, draws, reference_positions = tile_batch(data, scale)
        cpu_result, cpu_samples = timed(
            lambda: choose_terms_numpy(terms, data["coefficients"], offsets, draws), args.repeats
        )
        gpu_result, gpu_inclusive_samples = timed(
            lambda: gpu.choose_from_terms(terms, data["coefficients"], offsets, draws), args.repeats
        )

        d_terms = cp.asarray(terms)
        d_beta = cp.asarray(data["coefficients"])
        d_offsets = cp.asarray(offsets)
        d_draws = cp.asarray(draws, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()

        def resident():
            out = gpu.choose_from_terms(d_terms, d_beta, d_offsets, d_draws, return_device=True)
            cp.cuda.Stream.null.synchronize()
            return out

        _, gpu_resident_samples = timed(resident, args.repeats)
        rows.append(
            {
                "scale": scale,
                "choosers": int(draws.size),
                "interaction_rows": int(terms.shape[0]),
                "input_megabytes": terms.nbytes / 1e6,
                "cpu_ms": statistics.median(cpu_samples),
                "gpu_inclusive_ms": statistics.median(gpu_inclusive_samples),
                "gpu_resident_ms": statistics.median(gpu_resident_samples),
                "inclusive_speedup": statistics.median(cpu_samples) / statistics.median(gpu_inclusive_samples),
                "resident_speedup": statistics.median(cpu_samples) / statistics.median(gpu_resident_samples),
                "cpu_activitysim_choice_mismatches": int(np.count_nonzero(cpu_result.choices != reference_positions)),
                "gpu_activitysim_choice_mismatches": int(np.count_nonzero(gpu_result.choices != reference_positions)),
                "cpu_gpu_choice_mismatches": int(np.count_nonzero(cpu_result.choices != gpu_result.choices)),
            }
        )
        del d_terms, d_beta, d_offsets, d_draws
        cp.get_default_memory_pool().free_all_blocks()

    result = {
        "source": "ActivitySim 1.4 prototype_mtc mandatory_tour_scheduling",
        "batch": meta,
        "mandatory_batches": len(mandatory),
        "mandatory_choosers_total": sum(b["choosers"] for b in mandatory),
        "validation": {
            "exact_probability_choice_mismatches": exact_choice_mismatches,
            "all_batches_exact_probability_choice_mismatches": sum(x["exact_probability_choice_mismatches"] for x in all_validation),
            "all_batches_lowered_gpu_choice_mismatches": sum(x["lowered_gpu_choice_mismatches"] for x in all_validation),
            "lowered_utility_max_abs_error": float(utility_error.max()),
            "lowered_utility_mean_abs_error": float(utility_error.mean()),
            "batches": all_validation,
        },
        "benchmark": rows,
        "notes": [
            "GPU inclusive timing includes host-to-device inputs and device-to-host outputs.",
            "GPU resident timing assumes the lowered term matrix and draws already reside on device.",
            "Scale factors repeat captured real tours and their ActivitySim draws; no synthetic rows are generated.",
            "The CPU comparator uses NumPy BLAS for term aggregation and a Numba ragged choice loop.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
