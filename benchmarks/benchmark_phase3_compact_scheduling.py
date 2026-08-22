"""Benchmark compact-input compiled scheduling on real ActivitySim replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import numba

from choiceforge.cuda_backend import _cupy
from choiceforge.interaction_backend import offsets_from_ids
from choiceforge.scheduling_compiler import (
    CompiledCpuSchedulingModel,
    CompiledCudaSchedulingModel,
    SchedulingSchema,
)


def timed(fn, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return value, samples


def load_batch(root, meta):
    with np.load(root / meta["file"]) as loaded:
        return {key: loaded[key] for key in loaded.files}


def model_for(meta, data, gpu):
    schema = SchedulingSchema(
        tuple(meta["chooser_columns"]),
        tuple(meta["row_columns"]),
        tuple(meta["alternative_columns"]),
    )
    cls = CompiledCudaSchedulingModel if gpu else CompiledCpuSchedulingModel
    return cls(meta["compact_expressions"], data["coefficients"], schema)


def tiled(data, scale):
    base_offsets = offsets_from_ids(data["chooser_ids"])
    counts = np.diff(base_offsets)
    return {
        "chooser_values": np.tile(data["chooser_values"], (scale, 1)),
        "row_values": np.tile(data["row_values"], (scale, 1)),
        "alternative_values": data["alternative_values"],
        "alternative_ids": np.tile(data["alternative_ids"], scale),
        "offsets": np.r_[0, np.cumsum(np.tile(counts, scale))].astype(np.int64),
        "draws": np.tile(data["draws"], scale),
        "positions": np.tile(data["positions"], scale),
    }


def invoke(model, x, return_device=False):
    return model.choose(
        x["chooser_values"], x["row_values"], x["alternative_values"],
        x["alternative_ids"], x["offsets"], x["draws"],
        **({"return_device": True} if return_device else {}),
    )


def byte_size(x):
    return sum(x[k].nbytes for k in (
        "chooser_values", "row_values", "alternative_values", "alternative_ids", "offsets", "draws"
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=Path("benchmark-results/phase3-replay"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/phase3-summary.json"))
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 2 or len(manifest["batches"]) != 6:
        raise RuntimeError("Phase 3 requires the six-batch compact capture format")

    validation = []
    compile_start = time.perf_counter()
    models = []
    for meta in manifest["batches"]:
        data = load_batch(args.capture, meta)
        cpu = model_for(meta, data, gpu=False)
        gpu = model_for(meta, data, gpu=True)
        models.append((meta, data, cpu, gpu))
        offsets = offsets_from_ids(data["chooser_ids"])
        cpu_utilities = cpu.utilities(
            data["chooser_values"], data["row_values"], data["alternative_values"],
            data["alternative_ids"], offsets,
        )
        phase2_utilities = data["terms"] @ data["coefficients"]
        cpu_result = cpu.choose(
            data["chooser_values"], data["row_values"], data["alternative_values"],
            data["alternative_ids"], offsets, data["draws"],
        )
        gpu_result = gpu.choose(
            data["chooser_values"], data["row_values"], data["alternative_values"],
            data["alternative_ids"], offsets, data["draws"],
        )
        validation.append({
            "trace_label": meta["trace_label"],
            "choosers": meta["choosers"],
            "cpu_choice_mismatches": int(np.count_nonzero(cpu_result.choices != data["positions"])),
            "gpu_choice_mismatches": int(np.count_nonzero(gpu_result.choices != data["positions"])),
            "cpu_gpu_choice_mismatches": int(np.count_nonzero(cpu_result.choices != gpu_result.choices)),
            "compact_vs_phase2_utility_max_abs_error": float(np.max(np.abs(cpu_utilities - phase2_utilities))),
            "gpu_vs_cpu_logsum_max_abs_error": float(np.max(np.abs(gpu_result.logsums - cpu_result.logsums))),
        })
    compile_and_warm_seconds = time.perf_counter() - compile_start

    meta, data, cpu, gpu = max(models, key=lambda x: x[0]["interaction_rows"])
    cp = _cupy()
    results = []
    for scale in args.scales:
        x = tiled(data, scale)
        cpu_result, cpu_samples = timed(lambda: invoke(cpu, x), args.repeats)
        gpu_result, inclusive_samples = timed(lambda: invoke(gpu, x), args.repeats)

        resident = {
            "chooser_values": cp.asarray(x["chooser_values"]),
            "row_values": cp.asarray(x["row_values"]),
            "alternative_values": cp.asarray(x["alternative_values"]),
            "alternative_ids": cp.asarray(x["alternative_ids"]),
            "offsets": cp.asarray(x["offsets"]),
            "draws": cp.asarray(x["draws"], dtype=cp.float32),
        }
        cp.cuda.Stream.null.synchronize()

        def run_resident():
            answer = invoke(gpu, resident, return_device=True)
            cp.cuda.Stream.null.synchronize()
            return answer

        _, resident_samples = timed(run_resident, args.repeats)
        cpu_median = statistics.median(cpu_samples)
        inclusive_median = statistics.median(inclusive_samples)
        resident_median = statistics.median(resident_samples)
        results.append({
            "scale": scale,
            "choosers": int(x["draws"].size),
            "interaction_rows": int(x["row_values"].shape[0]),
            "compact_input_megabytes": byte_size(x) / 1e6,
            "phase2_term_megabytes": data["terms"].nbytes * scale / 1e6,
            "cpu_ms": cpu_median,
            "gpu_inclusive_ms": inclusive_median,
            "gpu_resident_ms": resident_median,
            "inclusive_speedup": cpu_median / inclusive_median,
            "resident_speedup": cpu_median / resident_median,
            "gpu_activitysim_choice_mismatches": int(np.count_nonzero(gpu_result.choices != x["positions"])),
            "cpu_activitysim_choice_mismatches": int(np.count_nonzero(cpu_result.choices != x["positions"])),
            "cpu_samples_ms": cpu_samples,
            "gpu_inclusive_samples_ms": inclusive_samples,
            "gpu_resident_samples_ms": resident_samples,
        })
        del resident
        cp.get_default_memory_pool().free_all_blocks()

    summary = {
        "source": "ActivitySim 1.4 prototype_mtc mandatory_tour_scheduling compact replay",
        "numba_threads": numba.get_num_threads(),
        "compile_and_six_batch_warm_seconds": compile_and_warm_seconds,
        "validation": {
            "choosers": sum(x["choosers"] for x in validation),
            "cpu_choice_mismatches": sum(x["cpu_choice_mismatches"] for x in validation),
            "gpu_choice_mismatches": sum(x["gpu_choice_mismatches"] for x in validation),
            "cpu_gpu_choice_mismatches": sum(x["cpu_gpu_choice_mismatches"] for x in validation),
            "batches": validation,
        },
        "largest_batch": meta,
        "benchmark": results,
        "notes": [
            "Compilation and first-call warm-up are excluded from steady-state timings and reported separately.",
            "Transfer-inclusive GPU timing moves the complete compact ABI and returns choices and logsums.",
            "The CPU baseline uses a generated parallel Numba expression kernel plus compiled ragged choice.",
            "Scale factors repeat captured real tours and ActivitySim-owned draws; no synthetic feature rows are generated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
