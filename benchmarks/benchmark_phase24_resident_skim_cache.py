"""Qualify the Phase 24 bounded resident hot-skim layer on public MTC data.

This is intentionally a cache-layer benchmark, not a whole-model speed claim.
Every valid mandatory-scheduling OD/period row reads every logical skim binding
in the reviewed MTC IR.  Independent CPU and CUDA implementations fold the raw
float32 bits into two 64-bit row hashes, so exactness covers every modeled read
without downloading a 700+ MB intermediate matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from choiceforge.cuda_backend import _cupy
from choiceforge.device_resident_runtime import DeviceResidentRuntime
from choiceforge.resident_skim_cache import ResidentOmxSkimCache


DEFAULT_OMX = (
    ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended"
    / "data_full" / "skims.omx"
)
DEFAULT_PIPELINE = (
    ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended"
    / "o-p17modeproof16-baseline-50000-1" / "pipeline.parquetpipeline"
    / "tours" / "mandatory_tour_scheduling.parquet"
)
DEFAULT_CAPTURE = ROOT / "benchmark-results" / "phase21-logsum-capture2"
DEFAULT_SPEC = (
    ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended"
    / "configs" / "tour_mode_choice.csv"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def load_workload(capture: Path, pipeline: Path):
    manifest = json.loads((capture / "manifest.json").read_text())
    tours = pd.read_parquet(
        pipeline, columns=["origin", "destination", "tour_category"]
    )
    tours = tours.loc[tours.tour_category.astype(str).eq("mandatory")]
    origins = []
    destinations = []
    out_periods = []
    in_periods = []
    source_rows = 0
    dropped_missing_destination = 0
    for item in manifest["batches"]:
        with np.load(capture / item["file"]) as loaded:
            ids = loaded["chooser_ids"].astype(np.int64, copy=False)
            starts = loaded["start"].astype(np.int16, copy=False)
            ends = loaded["end"].astype(np.int16, copy=False)
        source_rows += ids.size
        linked = tours.reindex(ids)
        if linked[["origin", "destination"]].isna().any().any():
            raise ValueError("captured scheduling identity is missing from public tours")
        origin = linked.origin.to_numpy(dtype=np.int64) - 1
        destination = linked.destination.to_numpy(dtype=np.int64) - 1
        valid = (origin >= 0) & (destination >= 0)
        dropped_missing_destination += int(np.count_nonzero(~valid))
        # These are the authoritative five representative times chosen by
        # ActivitySim for this purpose. Their sorted ordinal is the named skim
        # period, including school/university's special 18:00 EV row.
        start_values = np.unique(starts)
        end_values = np.unique(ends)
        if start_values.size != 5 or end_values.size != 5:
            raise ValueError("captured batch does not contain five skim periods")
        out_code = np.searchsorted(start_values, starts).astype(np.int32)
        in_code = np.searchsorted(end_values, ends).astype(np.int32)
        origins.append(origin[valid])
        destinations.append(destination[valid])
        out_periods.append(out_code[valid])
        in_periods.append(in_code[valid])
    return (
        {
            "origin": np.ascontiguousarray(np.concatenate(origins)),
            "destination": np.ascontiguousarray(np.concatenate(destinations)),
            "out_period": np.ascontiguousarray(np.concatenate(out_periods)),
            "in_period": np.ascontiguousarray(np.concatenate(in_periods)),
        },
        {
            "source_rows": int(source_rows),
            "valid_rows": int(sum(x.size for x in origins)),
            "missing_destination_rows_excluded": dropped_missing_destination,
            "batches": len(manifest["batches"]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omx", type=Path, default=DEFAULT_OMX)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--budget-gib", type=float, default=8.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "benchmark-results" / "phase24-resident-skim-cache.json",
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    cp = _cupy()
    from choiceforge.sharrow_ir import specification_ir

    document = specification_ir(pd.read_csv(args.spec, comment="#"))
    workload, workload_meta = load_workload(args.capture, args.pipeline)
    cache = ResidentOmxSkimCache.load(
        args.omx,
        document,
        budget_bytes=int(args.budget_gib * 1024**3),
        keep_host=True,
    )

    runtime = DeviceResidentRuntime()
    runtime.register_device_table("hot_skims", cache.runtime_columns())
    runtime.ingress_table("skim_workload", workload)
    runtime.seal_ingress()

    def gpu_operation(tables):
        data = tables["skim_workload"].columns
        first, second = cache.probe_gpu(
            data["origin"], data["destination"],
            data["out_period"], data["in_period"],
        )
        return {"skim_probe": {"hash1": first, "hash2": second}}

    # Compile and allocate before either measured loop.
    runtime.run_stage(
        "warmup.hot_skim_probe", reads=("hot_skims", "skim_workload"),
        writes=("skim_probe",), operation=gpu_operation,
    )
    runtime.synchronize()
    warm_hash = tuple(
        value.copy() for value in runtime.table("skim_probe").columns.values()
    )
    cache.probe_cpu(**workload)

    cpu_samples = []
    cpu_reference = None
    for _ in range(args.repetitions):
        started = time.perf_counter()
        current = cache.probe_cpu(**workload)
        cpu_samples.append(time.perf_counter() - started)
        if cpu_reference is None:
            cpu_reference = current
        elif not all(np.array_equal(a, b) for a, b in zip(cpu_reference, current)):
            raise AssertionError("CPU hot-skim probe was not repeatable")

    gpu_samples = []
    repeat_mismatches = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        runtime.run_stage(
            f"measured_{repetition}.hot_skim_probe",
            reads=("hot_skims", "skim_workload"), writes=("skim_probe",),
            operation=gpu_operation, replace=True,
        )
        runtime.synchronize()
        gpu_samples.append(time.perf_counter() - started)
        repeat_mismatches.append(
            int(
                cp.count_nonzero(
                    runtime.table("skim_probe").columns["hash1"] != warm_hash[0]
                ).item()
                + cp.count_nonzero(
                    runtime.table("skim_probe").columns["hash2"] != warm_hash[1]
                ).item()
            )
        )

    publication_started = time.perf_counter()
    published = runtime.publish({"skim_probe": ("hash1", "hash2")})["skim_probe"]
    publication_seconds = time.perf_counter() - publication_started
    runtime.assert_resident_contract()
    mismatch_count = int(
        np.count_nonzero(published["hash1"] != cpu_reference[0])
        + np.count_nonzero(published["hash2"] != cpu_reference[1])
    )
    cpu_median = statistics.median(cpu_samples)
    gpu_median = statistics.median(gpu_samples)
    setup_compute_publish = (
        cache.telemetry.device_upload_seconds + gpu_median + publication_seconds
    )
    raw_collection_bytes = int(args.omx.stat().st_size)
    logical_reads = workload_meta["valid_rows"] * cache.telemetry.logical_bindings
    telemetry = runtime.telemetry_dict()
    gates = {
        "reviewed_tour_mode_ir_has_315_terms": len(document["terms"]) == 315,
        "reviewed_ir_has_209_logical_skim_bindings": (
            cache.telemetry.logical_bindings == 209
        ),
        "hot_set_fits_declared_budget": (
            cache.telemetry.resident_float32_bytes <= cache.telemetry.budget_bytes
        ),
        "all_logical_reads_bit_exact": mismatch_count == 0,
        "all_measured_gpu_repeats_bit_exact": all(x == 0 for x in repeat_mismatches),
        "no_postseal_modeled_transfers": (
            telemetry["forbidden_postseal_host_bytes"] == 0
        ),
        "no_modeled_cpu_fallbacks": telemetry["modeled_cpu_fallbacks"] == 0,
        "single_final_publication": telemetry["publication_calls"] == 1,
        "gpu_probe_faster_than_cpu": gpu_median < cpu_median,
    }
    report = {
        "phase": 24,
        "claim_scope": (
            "budgeted resident float32 cache and exact all-binding raw-skim probe "
            "on the public mandatory-scheduling OD/period workload"
        ),
        "not_claimed": (
            "complete mode-choice utility/logsum integration or removal of the "
            "Phase 23 precomputed scheduling-logsum ingress"
        ),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "gpu_memory_bytes": int(cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"]),
            "cupy": cp.__version__,
        },
        "workload": {
            **workload_meta,
            "logical_skim_bindings": cache.telemetry.logical_bindings,
            "physical_device_cubes": cache.telemetry.physical_cubes,
            "logical_skim_reads_per_run": int(logical_reads),
            "hash_words_per_row": 2,
        },
        "cache": {
            **asdict(cache.telemetry),
            "raw_omx_file_bytes": raw_collection_bytes,
            "resident_fraction_of_uncompressed_required_source": (
                cache.telemetry.resident_float32_bytes
                / cache.telemetry.source_float64_bytes
            ),
            "resident_to_compressed_omx_file_ratio": (
                cache.telemetry.resident_float32_bytes / raw_collection_bytes
            ),
            "deduplicated_directional_aliases": (
                cache.telemetry.logical_bindings - cache.telemetry.physical_cubes
            ),
        },
        "timings_seconds": {
            "cpu_hot_cache_samples": cpu_samples,
            "gpu_resident_samples": gpu_samples,
            "cpu_hot_cache_median": cpu_median,
            "gpu_resident_median": gpu_median,
            "one_time_disk_read": cache.telemetry.disk_read_seconds,
            "one_time_device_upload": cache.telemetry.device_upload_seconds,
            "final_publication": publication_seconds,
            "single_run_upload_compute_publication": setup_compute_publish,
        },
        "speedup": {
            "resident_gpu_vs_cpu_hot_cache": cpu_median / gpu_median,
            "upload_inclusive_single_run": cpu_median / setup_compute_publish,
            "ten_runs_upload_amortized": (10 * cpu_median) / (
                cache.telemetry.device_upload_seconds + 10 * gpu_median
                + publication_seconds
            ),
            "hundred_runs_upload_amortized": (100 * cpu_median) / (
                cache.telemetry.device_upload_seconds + 100 * gpu_median
                + publication_seconds
            ),
        },
        "correctness": {
            "cpu_gpu_hash_word_mismatches": mismatch_count,
            "repeat_hash_word_mismatches": repeat_mismatches,
            "published_hash_sha256": array_sha256(
                published["hash1"], published["hash2"]
            ),
        },
        "runtime_telemetry": telemetry,
        "proof_gates": gates,
        "hashes": {
            "raw_omx_sha256": file_sha256(args.omx),
            "strict_ir_sha256": document["sha256"],
            "tour_mode_spec_sha256": file_sha256(args.spec),
            "capture_manifest_sha256": file_sha256(args.capture / "manifest.json"),
            "workload_sha256": array_sha256(*workload.values()),
            "cache_source": file_sha256(
                ROOT / "src" / "choiceforge" / "resident_skim_cache.py"
            ),
            "benchmark_source": file_sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "workload": report["workload"],
        "cache": {k: v for k, v in report["cache"].items() if k != "matrix_sha256"},
        "timings_seconds": {
            k: v for k, v in report["timings_seconds"].items() if not k.endswith("samples")
        },
        "speedup": report["speedup"],
        "correctness": report["correctness"],
        "proof_gates": gates,
    }, indent=2))
    if not all(gates.values()):
        raise SystemExit("Phase 24 resident hot-skim proof gate failed")


if __name__ == "__main__":
    main()
