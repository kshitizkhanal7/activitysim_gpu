"""Qualify the Phase 18 GPU-native vertical slice on public MTC households.

This is intentionally a model-shaped vertical slice, not a claim that all of
ActivitySim runs on the GPU.  It proves the runtime contract across feature
construction, entity-stable random draws, two dependent 21-alternative MNL
choices, and a deterministic zone aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from choiceforge.cuda_backend import _cupy
from choiceforge.gpu_native import (
    GpuNativeRuntime,
    entity_uniforms_cpu,
    entity_uniforms_gpu,
    plan_household_partitions,
    segmented_sum_sorted_gpu,
)
from choiceforge.numba_backend import linear_choice_numba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOUSEHOLDS = (
    ROOT
    / "benchmark-data"
    / "phase9-mtc-full"
    / "prototype_mtc_extended"
    / "data_full"
    / "households.csv"
)


def coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Stable synthetic coefficients for a public-data systems benchmark."""

    beta = np.linspace(-0.35, 0.45, 21 * 8, dtype=np.float32).reshape(21, 8)
    beta[:, 0] *= np.float32(0.25)
    constants = np.linspace(0.2, -0.15, 21, dtype=np.float32)
    return beta, constants


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_features(data: dict[str, np.ndarray]) -> np.ndarray:
    # The public MTC file uses negative values as missing-data sentinels.  This
    # systems proof maps them to zero income; it does not infer a socioeconomic
    # value from the sentinel.
    income = np.clip(
        data["income"].astype(np.float32), np.float32(0.0), np.float32(250_000.0)
    )
    hhsize = data["hhsize"].astype(np.float32)
    workers = data["num_workers"].astype(np.float32)
    autos = data["auto_ownership"].astype(np.float32)
    zones = data["TAZ"].astype(np.float32)
    hht = data["HHT"].astype(np.float32)
    return np.ascontiguousarray(
        np.column_stack(
            (
                np.ones(income.size, dtype=np.float32),
                income / np.float32(250_000.0),
                np.minimum(hhsize, np.float32(8.0)) / np.float32(8.0),
                np.minimum(workers, np.float32(5.0)) / np.float32(5.0),
                np.minimum(autos, np.float32(5.0)) / np.float32(5.0),
                np.remainder(zones, np.float32(31.0)) / np.float32(30.0),
                np.minimum(hht, np.float32(7.0)) / np.float32(7.0),
                (income > np.float32(100_000.0)).astype(np.float32),
            )
        ),
        dtype=np.float32,
    )


def device_features(cp, columns):
    income = cp.clip(
        columns["income"].astype(cp.float32), np.float32(0.0), np.float32(250_000.0)
    )
    hhsize = columns["hhsize"].astype(cp.float32)
    workers = columns["num_workers"].astype(cp.float32)
    autos = columns["auto_ownership"].astype(cp.float32)
    zones = columns["TAZ"].astype(cp.float32)
    hht = columns["HHT"].astype(cp.float32)
    features = cp.ascontiguousarray(
        cp.stack(
            (
                cp.ones(income.size, dtype=cp.float32),
                income / np.float32(250_000.0),
                cp.minimum(hhsize, np.float32(8.0)) / np.float32(8.0),
                cp.minimum(workers, np.float32(5.0)) / np.float32(5.0),
                cp.minimum(autos, np.float32(5.0)) / np.float32(5.0),
                cp.remainder(zones, np.float32(31.0)) / np.float32(30.0),
                cp.minimum(hht, np.float32(7.0)) / np.float32(7.0),
                (income > np.float32(100_000.0)).astype(cp.float32),
            ),
            axis=1,
        )
    )
    return {"features": features}


def load_households(path: Path, rows: int) -> dict[str, np.ndarray]:
    names = ["household_id", "TAZ", "income", "hhsize", "HHT", "auto_ownership", "num_workers"]
    frame = pd.read_csv(path, usecols=names, nrows=rows)
    if len(frame) != rows:
        raise ValueError(f"requested {rows:,} households but public file contains {len(frame):,}")
    return {name: frame[name].to_numpy(copy=True) for name in names}


def cpu_pipeline(data: dict[str, np.ndarray], beta: np.ndarray, constants: np.ndarray):
    features = host_features(data)
    first_draws = entity_uniforms_cpu(data["household_id"], seed=18_024, stream=1)
    first = linear_choice_numba(features, beta, constants, first_draws, parallel=True)
    second_features = features.copy()
    second_features[:, 7] = first.choices.astype(np.float32) / np.float32(20.0)
    second_draws = entity_uniforms_cpu(data["household_id"], seed=18_024, stream=2)
    second = linear_choice_numba(second_features, beta, constants, second_draws, parallel=True)
    order = np.argsort(data["TAZ"], kind="stable")
    sorted_zones = data["TAZ"][order]
    starts = np.r_[True, sorted_zones[1:] != sorted_zones[:-1]]
    zone_sums = np.add.reduceat(first.choices[order].astype(np.float32), np.flatnonzero(starts))
    return first.choices, second.choices, second.logsums, sorted_zones[starts], zone_sums


def gpu_pipeline(data: dict[str, np.ndarray], beta: np.ndarray, constants: np.ndarray):
    cp = _cupy()
    pool = cp.get_default_memory_pool()
    active_samples: list[int] = []

    def sample_active() -> None:
        active_samples.append(int(pool.used_bytes()))

    runtime = GpuNativeRuntime()
    total_start = time.perf_counter()
    input_table = runtime.ingress_table("households", data)
    params = runtime.ingress_table("parameters", {"beta": beta, "constants": constants})
    sample_active()
    runtime.seal_ingress()
    cp.cuda.Stream.null.synchronize()
    compute_start = time.perf_counter()

    feature_table = runtime.run_stage(
        "household_feature_graph",
        device_features,
        cp,
        input_table.columns,
        output_table="features",
    )
    sample_active()
    draw1 = runtime.run_stage(
        "entity_rng_stream_1",
        lambda: {"draw": entity_uniforms_gpu(input_table.columns["household_id"], 18_024, 1)},
    )
    first = runtime.linear_choice(
        feature_table.columns["features"],
        params.columns["beta"],
        params.columns["constants"],
        draw1.columns["draw"],
    )
    sample_active()
    second_features = cp.ascontiguousarray(feature_table.columns["features"].copy())
    second_features[:, 7] = first.columns["choice"].astype(cp.float32) / np.float32(20.0)
    draw2 = runtime.run_stage(
        "entity_rng_stream_2",
        lambda: {"draw": entity_uniforms_gpu(input_table.columns["household_id"], 18_024, 2)},
    )
    second = runtime.linear_choice(
        second_features,
        params.columns["beta"],
        params.columns["constants"],
        draw2.columns["draw"],
    )
    sample_active()
    order = cp.argsort(input_table.columns["TAZ"], kind="stable")
    aggregates = runtime.run_stage(
        "deterministic_zone_aggregation",
        segmented_sum_sorted_gpu,
        input_table.columns["TAZ"][order],
        first.columns["choice"][order].astype(cp.float32),
    )
    sample_active()
    cp.cuda.Stream.null.synchronize()
    compute_seconds = time.perf_counter() - compute_start
    runtime.assert_gpu_only()
    first_out = runtime.egress_table(first, ("choice",))
    second_out = runtime.egress_table(second)
    aggregate_out = runtime.egress_table(aggregates)
    total_seconds = time.perf_counter() - total_start
    mask = aggregate_out["is_start"].astype(bool)
    return {
        "first_choice": first_out["choice"],
        "second_choice": second_out["choice"],
        "second_logsum": second_out["logsum"],
        "zone_id": aggregate_out["group_id"][mask],
        "zone_sum": aggregate_out["sum"][mask],
        "compute_seconds": compute_seconds,
        "total_seconds": total_seconds,
        "sampled_device_active_peak_bytes": max(active_samples),
        "telemetry": runtime.telemetry,
    }


def machine() -> dict[str, object]:
    cp = _cupy()
    props = cp.cuda.runtime.getDeviceProperties(0)
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        smi = "unavailable"
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
        "device_total_bytes": int(props["totalGlobalMem"]),
        "nvidia_smi": smi,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=50_000)
    parser.add_argument("--partition-households", type=int, default=17_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--input", type=Path, default=DEFAULT_HOUSEHOLDS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-results" / "phase18-gpu-native.json"
    )
    args = parser.parse_args()

    data = load_households(args.input, args.households)
    beta, constants = coefficients()

    # Compile/JIT warm-up is deliberately outside reported samples.
    cpu_pipeline({name: values[:1024] for name, values in data.items()}, beta, constants)
    gpu_pipeline({name: values[:1024] for name, values in data.items()}, beta, constants)

    cpu_times = []
    gpu_compute_times = []
    gpu_total_times = []
    cpu_result = None
    gpu_result = None
    for _ in range(args.repetitions):
        started = time.perf_counter()
        cpu_result = cpu_pipeline(data, beta, constants)
        cpu_times.append(time.perf_counter() - started)
        gpu_result = gpu_pipeline(data, beta, constants)
        gpu_compute_times.append(gpu_result["compute_seconds"])
        gpu_total_times.append(gpu_result["total_seconds"])

    assert cpu_result is not None and gpu_result is not None
    first_mismatches = int(np.count_nonzero(cpu_result[0] != gpu_result["first_choice"]))
    second_mismatches = int(np.count_nonzero(cpu_result[1] != gpu_result["second_choice"]))
    first_mismatch_rate = first_mismatches / args.households
    second_mismatch_rate = second_mismatches / args.households
    zone_ids_exact = np.array_equal(cpu_result[3], gpu_result["zone_id"])
    zone_sum_max_abs = float(np.max(np.abs(cpu_result[4] - gpu_result["zone_sum"])))
    logsum_max_abs = float(np.max(np.abs(cpu_result[2] - gpu_result["second_logsum"])))

    partition_choices = []
    partition_logsums = []
    for begin, end in plan_household_partitions(args.households, args.partition_households):
        piece = gpu_pipeline({name: values[begin:end] for name, values in data.items()}, beta, constants)
        partition_choices.append(piece["second_choice"])
        partition_logsums.append(piece["second_logsum"])
    partition_choice_exact = np.array_equal(np.concatenate(partition_choices), gpu_result["second_choice"])
    partition_logsum_exact = np.array_equal(np.concatenate(partition_logsums), gpu_result["second_logsum"])

    telemetry = gpu_result["telemetry"]
    cpu_median = statistics.median(cpu_times)
    gpu_compute_median = statistics.median(gpu_compute_times)
    gpu_total_median = statistics.median(gpu_total_times)
    gates = {
        "first_choice_mismatch_rate_le_1e-6": first_mismatch_rate <= 1e-6,
        "dependent_choice_mismatch_rate_le_2e-6": second_mismatch_rate <= 2e-6,
        "zone_ids_exact": zone_ids_exact,
        "zone_sum_max_abs_le_1": zone_sum_max_abs <= 1.0,
        "dependent_logsum_max_abs_le_0_01": logsum_max_abs <= 0.01,
        "partition_choice_bit_exact": partition_choice_exact,
        "partition_logsum_bit_exact": partition_logsum_exact,
        "modeled_cpu_fallbacks_zero": telemetry.modeled_cpu_fallbacks == 0,
        "modeled_host_to_device_bytes_zero": telemetry.modeled_host_to_device_bytes == 0,
        "modeled_device_to_host_bytes_zero": telemetry.modeled_device_to_host_bytes == 0,
        "gpu_compute_faster_than_numba": gpu_compute_median < cpu_median,
        "gpu_total_faster_than_numba": gpu_total_median < cpu_median,
    }
    report = {
        "phase": 18,
        "claim_scope": "public-MTC model-shaped GPU-native vertical slice; not full ActivitySim",
        "boundary": {
            "cpu_allowed": ["CSV/config input", "one-time ingress", "kernel launch", "final egress/output"],
            "gpu_required": [
                "feature graph",
                "counter-based random draws",
                "two dependent 21-alternative MNL choices",
                "zone aggregation",
            ],
        },
        "input": {"path": str(args.input.relative_to(ROOT)), "households": args.households},
        "partition_households": args.partition_households,
        "repetitions": args.repetitions,
        "machine": machine(),
        "reproducibility": {
            "repository_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "households_csv_sha256": sha256(args.input),
            "benchmark_script_sha256": sha256(Path(__file__).resolve()),
            "gpu_runtime_sha256": sha256(ROOT / "src" / "choiceforge" / "gpu_native.py"),
            "coefficient_sha256": hashlib.sha256(beta.tobytes() + constants.tobytes()).hexdigest(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cupy": _cupy().__version__,
            "random_policy": "SplitMix64(entity_id xor seed-mix xor stream-mix), upper 24 bits to float32",
        },
        "timings_seconds": {
            "cpu_numba_samples": cpu_times,
            "gpu_compute_samples": gpu_compute_times,
            "gpu_total_with_transfer_samples": gpu_total_times,
            "cpu_numba_median": cpu_median,
            "gpu_compute_median": gpu_compute_median,
            "gpu_total_with_transfer_median": gpu_total_median,
        },
        "speedup": {
            "gpu_compute_vs_numba": cpu_median / gpu_compute_median,
            "gpu_total_with_transfer_vs_numba": cpu_median / gpu_total_median,
        },
        "correctness": {
            "first_choice_mismatches": first_mismatches,
            "first_choice_mismatch_rate": first_mismatch_rate,
            "dependent_choice_mismatches": second_mismatches,
            "dependent_choice_mismatch_rate": second_mismatch_rate,
            "zone_sum_max_abs_error": zone_sum_max_abs,
            "dependent_logsum_max_abs_error": logsum_max_abs,
            "cpu_gpu_note": (
                "GPU partitioning must be bit-exact. CPU/GPU uses an explicit numerical-equivalence "
                "bound because CUDA expf/FMA and Numba arithmetic are not bit-identical."
            ),
        },
        "telemetry": {
            "input_bytes": telemetry.input_bytes,
            "output_bytes": telemetry.output_bytes,
            "modeled_host_to_device_bytes": telemetry.modeled_host_to_device_bytes,
            "modeled_device_to_host_bytes": telemetry.modeled_device_to_host_bytes,
            "modeled_cpu_fallbacks": telemetry.modeled_cpu_fallbacks,
            "kernel_stages": telemetry.kernel_stages,
            "sampled_device_active_peak_bytes": gpu_result["sampled_device_active_peak_bytes"],
            "sampled_peak_note": "CuPy active allocations sampled after stages; temporary peaks may be higher",
        },
        "gates": gates,
        "qualified": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not report["qualified"]:
        raise SystemExit("Phase 18 qualification failed")


if __name__ == "__main__":
    main()
