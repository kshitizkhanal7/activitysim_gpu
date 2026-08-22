"""Validate and benchmark all captured trip-destination simulation batches."""

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
    choose_batched_terms_numpy,
    choose_terms_numpy,
)


def load_batches(root: Path, manifest: dict) -> list[tuple[dict, dict]]:
    result = []
    for meta in manifest["batches"]:
        with np.load(root / meta["file"]) as loaded:
            result.append((meta, {key: loaded[key] for key in loaded.files}))
    return result


def timed(function, repeats: int):
    samples = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        samples.append((time.perf_counter() - started) * 1000)
    return value, samples


def pack_batches(batches):
    terms = np.concatenate([data["terms"] for _, data in batches])
    coefficients = np.stack([data["coefficients"] for _, data in batches])
    offsets = [0]
    row_base = 0
    segments = []
    for segment, (_, data) in enumerate(batches):
        offsets.extend((data["offsets"][1:] + row_base).tolist())
        segments.extend([segment] * (len(data["offsets"]) - 1))
        row_base += len(data["terms"])
    return {
        "terms": terms,
        "coefficients": coefficients,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "segments": np.asarray(segments, dtype=np.int32),
        "draws": np.concatenate([data["draws"] for _, data in batches]),
        "positions": np.concatenate([data["positions"] for _, data in batches]),
        "logsums": np.concatenate([data["logsums"] for _, data in batches]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=Path("benchmark-results/phase6-trip-destination-replay"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/phase6-replay-summary.json"))
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    manifest = json.loads((args.capture / "manifest.json").read_text(encoding="utf-8"))
    batches = load_batches(args.capture, manifest)
    if len(batches) != 30:
        raise RuntimeError(f"expected 30 captured trip-destination batches, found {len(batches)}")

    exact_gpu = CudaChoiceBackend()
    fused_gpu = CudaInteractionBackend()
    term_counts = {data["terms"].shape[1] for _, data in batches}
    if len(term_counts) != 1:
        raise RuntimeError(f"captured batches have incompatible term counts: {term_counts}")
    packed = pack_batches(batches)
    packed_terms = packed["terms"]
    packed_coefficients = packed["coefficients"]
    packed_offsets = packed["offsets"]
    packed_segments = packed["segments"]
    packed_draws = packed["draws"]
    packed_positions = packed["positions"]
    packed_logsums = packed["logsums"]

    packed_cpu = choose_batched_terms_numpy(
        packed_terms, packed_coefficients, packed_offsets, packed_segments, packed_draws
    )
    packed_gpu = fused_gpu.choose_from_batched_terms(
        packed_terms, packed_coefficients, packed_offsets, packed_segments, packed_draws
    )
    validation = []
    for meta, data in batches:
        exact_positions = exact_gpu.choose_from_probabilities(data["probabilities"], data["draws"])
        cpu = choose_terms_numpy(data["terms"], data["coefficients"], data["offsets"], data["draws"])
        gpu = fused_gpu.choose_from_terms(data["terms"], data["coefficients"], data["offsets"], data["draws"])
        reconstructed = data["terms"].astype(np.float64) @ data["coefficients"].astype(np.float64)
        validation.append(
            {
                "trace_label": meta["trace_label"],
                "choosers": meta["choosers"],
                "interaction_rows": meta["interaction_rows"],
                "exact_probability_choice_mismatches": int(np.count_nonzero(exact_positions != data["positions"])),
                "cpu_choice_mismatches": int(np.count_nonzero(cpu.choices != data["positions"])),
                "gpu_choice_mismatches": int(np.count_nonzero(gpu.choices != data["positions"])),
                "cpu_gpu_choice_mismatches": int(np.count_nonzero(cpu.choices != gpu.choices)),
                "utility_max_abs_error": float(np.max(np.abs(reconstructed - data["utilities"]))),
                "gpu_logsum_max_abs_error": float(np.max(np.abs(gpu.logsums - data["logsums"]))),
            }
        )

    def cpu_suite():
        return [
            choose_terms_numpy(data["terms"], data["coefficients"], data["offsets"], data["draws"])
            for _, data in batches
        ]

    def gpu_inclusive_suite():
        return [
            fused_gpu.choose_from_terms(data["terms"], data["coefficients"], data["offsets"], data["draws"])
            for _, data in batches
        ]

    def cpu_batched_suite():
        return choose_batched_terms_numpy(
            packed_terms,
            packed_coefficients,
            packed_offsets,
            packed_segments,
            packed_draws,
        )

    def gpu_batched_inclusive_suite():
        return fused_gpu.choose_from_batched_terms(
            packed_terms,
            packed_coefficients,
            packed_offsets,
            packed_segments,
            packed_draws,
        )

    cpu_suite()
    gpu_inclusive_suite()
    cpu_batched_suite()
    gpu_batched_inclusive_suite()
    _, cpu_samples = timed(cpu_suite, args.repeats)
    _, gpu_inclusive_samples = timed(gpu_inclusive_suite, args.repeats)
    _, cpu_batched_samples = timed(cpu_batched_suite, args.repeats)
    _, gpu_batched_inclusive_samples = timed(gpu_batched_inclusive_suite, args.repeats)

    # Measure the crossover before allocating the resident suite so the memory
    # pool state matches the transfer-inclusive production path above.
    crossover = []
    for batch_count in (1, 2, 3, 5, 10, 15, 20, 30):
        subset = pack_batches(batches[:batch_count])

        def subset_cpu():
            return choose_batched_terms_numpy(
                subset["terms"], subset["coefficients"], subset["offsets"],
                subset["segments"], subset["draws"]
            )

        def subset_gpu():
            return fused_gpu.choose_from_batched_terms(
                subset["terms"], subset["coefficients"], subset["offsets"],
                subset["segments"], subset["draws"]
            )

        _, subset_cpu_samples = timed(subset_cpu, args.repeats)
        _, subset_gpu_samples = timed(subset_gpu, args.repeats)
        subset_cpu_median = statistics.median(subset_cpu_samples)
        subset_gpu_median = statistics.median(subset_gpu_samples)
        crossover.append(
            {
                "batches": batch_count,
                "interaction_rows": len(subset["terms"]),
                "choosers": len(subset["segments"]),
                "cpu_median_ms": subset_cpu_median,
                "gpu_inclusive_median_ms": subset_gpu_median,
                "speedup": subset_cpu_median / subset_gpu_median,
            }
        )

    cp = _cupy()
    resident = [
        {
            "terms": cp.asarray(data["terms"]),
            "coefficients": cp.asarray(data["coefficients"]),
            "offsets": cp.asarray(data["offsets"]),
            "draws": cp.asarray(data["draws"], dtype=cp.float32),
        }
        for _, data in batches
    ]
    cp.cuda.Stream.null.synchronize()

    packed_resident = {
        "terms": cp.asarray(packed_terms),
        "coefficients": cp.asarray(packed_coefficients),
        "offsets": cp.asarray(packed_offsets),
        "segments": cp.asarray(packed_segments),
        "draws": cp.asarray(packed_draws, dtype=cp.float32),
    }

    def gpu_resident_suite():
        result = [
            fused_gpu.choose_from_terms(
                data["terms"], data["coefficients"], data["offsets"], data["draws"], return_device=True
            )
            for data in resident
        ]
        cp.cuda.Stream.null.synchronize()
        return result

    gpu_resident_suite()
    _, gpu_resident_samples = timed(gpu_resident_suite, args.repeats)

    def gpu_batched_resident_suite():
        result = fused_gpu.choose_from_batched_terms(
            packed_resident["terms"],
            packed_resident["coefficients"],
            packed_resident["offsets"],
            packed_resident["segments"],
            packed_resident["draws"],
            return_device=True,
        )
        cp.cuda.Stream.null.synchronize()
        return result

    gpu_batched_resident_suite()
    _, gpu_batched_resident_samples = timed(gpu_batched_resident_suite, args.repeats)

    cpu_median = statistics.median(cpu_samples)
    inclusive_median = statistics.median(gpu_inclusive_samples)
    resident_median = statistics.median(gpu_resident_samples)
    cpu_batched_median = statistics.median(cpu_batched_samples)
    gpu_batched_inclusive_median = statistics.median(gpu_batched_inclusive_samples)
    gpu_batched_resident_median = statistics.median(gpu_batched_resident_samples)
    summary = {
        "phase": "6A",
        "source": "ActivitySim 1.4 prototype_mtc trip_destination simulation boundary",
        "batches": len(batches),
        "choosers": sum(meta["choosers"] for meta, _ in batches),
        "interaction_rows": sum(meta["interaction_rows"] for meta, _ in batches),
        "expanded_megabytes": sum(meta["expanded_megabytes"] for meta, _ in batches),
        "validation": {
            "exact_probability_choice_mismatches": sum(x["exact_probability_choice_mismatches"] for x in validation),
            "cpu_choice_mismatches": sum(x["cpu_choice_mismatches"] for x in validation),
            "gpu_choice_mismatches": sum(x["gpu_choice_mismatches"] for x in validation),
            "cpu_gpu_choice_mismatches": sum(x["cpu_gpu_choice_mismatches"] for x in validation),
            "batched_cpu_choice_mismatches": int(
                np.count_nonzero(packed_cpu.choices != packed_positions)
            ),
            "batched_gpu_choice_mismatches": int(
                np.count_nonzero(packed_gpu.choices != packed_positions)
            ),
            "batched_cpu_gpu_choice_mismatches": int(
                np.count_nonzero(packed_cpu.choices != packed_gpu.choices)
            ),
            "batched_gpu_max_logsum_abs_error": float(
                np.max(np.abs(packed_gpu.logsums - packed_logsums))
            ),
            "max_utility_abs_error": max(x["utility_max_abs_error"] for x in validation),
            "max_gpu_logsum_abs_error": max(x["gpu_logsum_max_abs_error"] for x in validation),
            "batches": validation,
        },
        "suite_benchmark": {
            "repeats": args.repeats,
            "cpu_samples_ms": cpu_samples,
            "gpu_inclusive_samples_ms": gpu_inclusive_samples,
            "gpu_resident_samples_ms": gpu_resident_samples,
            "cpu_median_ms": cpu_median,
            "gpu_inclusive_median_ms": inclusive_median,
            "gpu_resident_median_ms": resident_median,
            "inclusive_speedup": cpu_median / inclusive_median,
            "resident_speedup": cpu_median / resident_median,
            "cpu_batched_samples_ms": cpu_batched_samples,
            "gpu_batched_inclusive_samples_ms": gpu_batched_inclusive_samples,
            "gpu_batched_resident_samples_ms": gpu_batched_resident_samples,
            "cpu_batched_median_ms": cpu_batched_median,
            "gpu_batched_inclusive_median_ms": gpu_batched_inclusive_median,
            "gpu_batched_resident_median_ms": gpu_batched_resident_median,
            "batched_inclusive_speedup": cpu_batched_median / gpu_batched_inclusive_median,
            "batched_resident_speedup": cpu_batched_median / gpu_batched_resident_median,
            "launches_before": len(batches),
            "launches_after": 1,
            "real_prefix_crossover": crossover,
        },
        "notes": [
            "The suite includes all 30 purpose/trip-number batches and preserves their real ragged alternative counts.",
            "Transfer-inclusive GPU timing copies all term inputs and returns choices and logsums for every batch.",
            "The segmented path packs all purpose/trip-number segments into one launch while retaining a coefficient row per segment.",
            "This boundary excludes sampling and the two upstream trip-mode-choice logsum calculations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
