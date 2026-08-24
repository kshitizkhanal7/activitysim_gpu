"""Qualify Phase 20 mandatory-tour expansion and calibrated scheduling replay.

The row builder consumes the exact mandatory-tour-frequency choices proved by
Phase 19.  The downstream boundary uses a compact capture from the same public
50,000-household ActivitySim run: real feasible alternatives, mode-choice
logsums, timetable primitives, coefficients, draws, and selected positions.
Preparation of those compact scheduling inputs remains ActivitySim CPU work and
is explicitly outside the kernel timing and claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from choiceforge.cuda_backend import _cupy
from choiceforge.gpu_native import ActivitySimRandomLedger
from choiceforge.scheduling_compiler import (
    CompiledCpuSchedulingModel,
    CompiledCudaSchedulingModel,
    SchedulingSchema,
)
from choiceforge.tour_expansion import TOUR_COLUMNS, mandatory_tours_cpu, mandatory_tours_gpu


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended"
PIPELINE = PUBLIC / "o-p17modeproof16-baseline-50000-1" / "pipeline.parquetpipeline"
CAPTURE = ROOT / "benchmark-results" / "phase20-scheduling-replay"
PHASE19 = ROOT / "benchmark-results" / "phase19-calibrated-chain.json"
MTF_ALTERNATIVES = ("work1", "work2", "school1", "school2", "work_and_school")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def timed(operation, repetitions: int) -> tuple[Any, list[float]]:
    samples = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
    return result, samples


def load_tour_inputs(pipeline: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    persons = pd.read_parquet(pipeline / "persons" / "mandatory_tour_frequency.parquet")
    mandatory = persons.cdap_activity.astype(str).eq("M")
    choosers = persons.loc[mandatory]
    choices = pd.Categorical(
        choosers.mandatory_tour_frequency, categories=list(MTF_ALTERNATIVES)
    ).codes.astype(np.int8)
    if np.any(choices < 0):
        raise RuntimeError("public mandatory chooser has an unknown frequency alternative")
    inputs = {
        "person_id": choosers.index.to_numpy(dtype=np.int64, copy=True),
        "household_id": choosers.household_id.to_numpy(dtype=np.int64, copy=True),
        "mtf_choice": choices,
        "is_worker": choosers.is_worker.to_numpy(dtype=np.bool_, copy=True),
        "workplace_zone_id": choosers.workplace_zone_id.to_numpy(dtype=np.int64, copy=True),
        "school_zone_id": choosers.school_zone_id.to_numpy(dtype=np.int64, copy=True),
        "home_zone_id": choosers.home_zone_id.to_numpy(dtype=np.int64, copy=True),
    }
    tours = pd.read_parquet(pipeline / "tours" / "mandatory_tour_frequency.parquet")
    return inputs, tours


def row_mismatches(actual: dict[str, np.ndarray], reference: pd.DataFrame) -> dict[str, int]:
    expected: dict[str, np.ndarray] = {
        "tour_id": reference.index.to_numpy(dtype=np.int64),
        "person_id": reference.person_id.to_numpy(dtype=np.int64),
        "tour_type": np.where(reference.tour_type.astype(str).to_numpy() == "work", 0, 1).astype(np.int8),
        "tour_type_count": reference.tour_type_count.to_numpy(dtype=np.int8),
        "tour_type_num": reference.tour_type_num.to_numpy(dtype=np.int8),
        "tour_num": reference.tour_num.to_numpy(dtype=np.int8),
        "tour_count": reference.tour_count.to_numpy(dtype=np.int8),
        "tour_category": np.zeros(len(reference), dtype=np.int8),
        "number_of_participants": reference.number_of_participants.to_numpy(dtype=np.int8),
        "destination": reference.destination.to_numpy(dtype=np.int64),
        "origin": reference.origin.to_numpy(dtype=np.int64),
        "household_id": reference.household_id.to_numpy(dtype=np.int64),
    }
    return {
        name: int(np.count_nonzero(np.asarray(actual[name]) != expected[name]))
        for name in TOUR_COLUMNS
    }


def load_scheduling(capture: Path):
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 3 or not manifest.get("compact_only"):
        raise RuntimeError("Phase 20 requires compact-only scheduling capture format 3")
    batches = []
    for meta in manifest["batches"]:
        with np.load(capture / meta["file"]) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        schema = SchedulingSchema(
            tuple(meta["chooser_columns"]),
            tuple(meta["row_columns"]),
            tuple(meta["alternative_columns"]),
        )
        cpu = CompiledCpuSchedulingModel(meta["compact_expressions"], data["coefficients"], schema)
        gpu = CompiledCudaSchedulingModel(meta["compact_expressions"], data["coefficients"], schema)
        batches.append((meta, data, cpu, gpu))
    return manifest, batches


def invoke(model, data, *, return_device=False):
    return model.choose(
        data["chooser_values"],
        data["row_values"],
        data["alternative_values"],
        data["alternative_ids"],
        data["offsets"],
        data["draws"],
        **({"return_device": True} if return_device else {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=PIPELINE)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-results" / "phase20-tour-chain.json"
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=ROOT / "benchmark-results" / "phase20-device-checkpoint.json",
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    cp = _cupy()
    inputs, tour_checkpoint = load_tour_inputs(args.pipeline)
    schedule_checkpoint = pd.read_parquet(
        args.pipeline / "tours" / "mandatory_tour_scheduling.parquet"
    )
    phase19 = json.loads(PHASE19.read_text(encoding="utf-8"))
    phase19_exact = all(
        phase19["correctness"][name] == 0
        for name in (
            "gpu_auto_checkpoint_mismatches",
            "gpu_mtf_checkpoint_mismatches",
        )
    )

    cpu_tours = mandatory_tours_cpu(**inputs)
    device_inputs = {name: cp.asarray(value) for name, value in inputs.items()}
    gpu_table = mandatory_tours_gpu(**device_inputs)
    cp.cuda.Stream.null.synchronize()
    gpu_tours = {name: cp.asnumpy(value) for name, value in gpu_table.columns.items()}
    cpu_row_errors = row_mismatches(cpu_tours, tour_checkpoint)
    gpu_row_errors = row_mismatches(gpu_tours, tour_checkpoint)

    # Warm array allocation/JIT paths before collecting row-expansion samples.
    mandatory_tours_gpu(**device_inputs)
    cp.cuda.Stream.null.synchronize()
    _, cpu_row_times = timed(lambda: mandatory_tours_cpu(**inputs), args.repetitions)

    def gpu_rows_resident():
        value = mandatory_tours_gpu(**device_inputs)
        cp.cuda.Stream.null.synchronize()
        return value

    _, gpu_row_resident_times = timed(gpu_rows_resident, args.repetitions)

    def gpu_rows_inclusive():
        uploaded = {name: cp.asarray(value) for name, value in inputs.items()}
        value = mandatory_tours_gpu(**uploaded)
        result = cp.asnumpy(value.columns["tour_id"])
        cp.cuda.Stream.null.synchronize()
        return result

    _, gpu_row_inclusive_times = timed(gpu_rows_inclusive, args.repetitions)

    capture_manifest, batches = load_scheduling(args.capture)
    # Compile and warm every segment before validation/timing.
    for _, data, cpu, gpu in batches:
        invoke(cpu, data)
        invoke(gpu, data)

    schedule_validation = []
    scheduled_ids = []
    scheduled_tdd = []
    cpu_gpu_logsum_error = 0.0
    for meta, data, cpu, gpu in batches:
        cpu_result = invoke(cpu, data)
        gpu_result = invoke(gpu, data)
        cpu_gpu_logsum_error = max(
            cpu_gpu_logsum_error,
            float(np.max(np.abs(cpu_result.logsums - gpu_result.logsums))),
        )
        selected_rows = data["offsets"][:-1] + gpu_result.choices
        tdd = data["alternative_ids"][selected_rows]
        scheduled_ids.append(data["chooser_ids"])
        scheduled_tdd.append(tdd)
        schedule_validation.append(
            {
                "trace_label": meta["trace_label"],
                "choosers": int(meta["choosers"]),
                "interaction_rows": int(meta["interaction_rows"]),
                "cpu_activitysim_choice_mismatches": int(
                    np.count_nonzero(cpu_result.choices != data["positions"])
                ),
                "gpu_activitysim_choice_mismatches": int(
                    np.count_nonzero(gpu_result.choices != data["positions"])
                ),
                "cpu_gpu_choice_mismatches": int(
                    np.count_nonzero(cpu_result.choices != gpu_result.choices)
                ),
            }
        )

    ids = np.concatenate(scheduled_ids)
    tdd = np.concatenate(scheduled_tdd)
    order = np.argsort(ids, kind="stable")
    reference_schedule = schedule_checkpoint.sort_index()
    schedule_ids_match = bool(np.array_equal(ids[order], reference_schedule.index.to_numpy()))
    schedule_tdd_mismatches = int(
        np.count_nonzero(tdd[order] != reference_schedule.tdd.to_numpy())
    )
    expanded_ids_match_scheduling = bool(
        np.array_equal(np.sort(gpu_tours["tour_id"]), ids[order])
    )

    def cpu_schedule():
        return [invoke(cpu, data) for _, data, cpu, _ in batches]

    _, cpu_schedule_times = timed(cpu_schedule, args.repetitions)

    def gpu_schedule_inclusive():
        return [invoke(gpu, data) for _, data, _, gpu in batches]

    _, gpu_schedule_inclusive_times = timed(gpu_schedule_inclusive, args.repetitions)

    resident_batches = []
    for _, data, _, gpu in batches:
        resident_batches.append(
            (
                gpu,
                {
                    "chooser_values": cp.asarray(data["chooser_values"]),
                    "row_values": cp.asarray(data["row_values"]),
                    "alternative_values": cp.asarray(data["alternative_values"]),
                    "alternative_ids": cp.asarray(data["alternative_ids"]),
                    "offsets": cp.asarray(data["offsets"]),
                    "draws": cp.asarray(data["draws"], dtype=cp.float64),
                },
            )
        )
    cp.cuda.Stream.null.synchronize()

    def gpu_schedule_resident():
        value = [invoke(gpu, data, return_device=True) for gpu, data in resident_batches]
        cp.cuda.Stream.null.synchronize()
        return value

    repeated_choices = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        result = gpu_schedule_resident()
        gpu_time = time.perf_counter() - start
        repeated_choices.append([cp.asnumpy(item.choices) for item in result])
        if "gpu_schedule_resident_times" not in locals():
            gpu_schedule_resident_times = []
        gpu_schedule_resident_times.append(gpu_time)
    schedule_repeat_exact = all(
        np.array_equal(repeated_choices[0][batch], trial[batch])
        for trial in repeated_choices[1:]
        for batch in range(len(batches))
    )

    cpu_row_median = statistics.median(cpu_row_times)
    gpu_row_resident_median = statistics.median(gpu_row_resident_times)
    gpu_row_inclusive_median = statistics.median(gpu_row_inclusive_times)
    cpu_schedule_median = statistics.median(cpu_schedule_times)
    gpu_schedule_resident_median = statistics.median(gpu_schedule_resident_times)
    gpu_schedule_inclusive_median = statistics.median(gpu_schedule_inclusive_times)

    random_ledger = ActivitySimRandomLedger()
    random_ledger.reserve("tours", "mandatory_tour_scheduling", 1)
    table_hashes = {name: array_sha256(gpu_tours[name]) for name in TOUR_COLUMNS}
    table_hashes["tdd"] = array_sha256(tdd[order])
    checkpoint_manifest = {
        "format_version": 1,
        "phase": 20,
        "checkpoint_name": "mandatory_tour_scheduling",
        "completed_components": [
            "auto_ownership_simulate",
            "mandatory_tour_frequency",
            "mandatory_tour_row_expansion",
            "mandatory_tour_scheduling_kernel_replay",
        ],
        "device_tables": {
            "mandatory_tours": {
                "rows": int(len(gpu_tours["tour_id"])),
                "columns": {
                    name: {"dtype": str(gpu_tours[name].dtype), "sha256": table_hashes[name]}
                    for name in TOUR_COLUMNS
                },
            },
            "mandatory_schedule": {
                "rows": int(len(tdd)),
                "columns": {"tdd": {"dtype": str(tdd.dtype), "sha256": table_hashes["tdd"]}},
            },
        },
        "random_offsets": random_ledger.snapshot(),
        "source_capture_manifest_sha256": sha256(args.capture / "manifest.json"),
        "restart_limit": (
            "hash-complete proof manifest; the compact scheduling preparation arrays "
            "remain external captured inputs, not a self-contained whole-model restart"
        ),
    }
    args.checkpoint_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_manifest.write_text(
        json.dumps(checkpoint_manifest, indent=2), encoding="utf-8"
    )

    correctness = {
        "phase19_upstream_gpu_choices_exact": phase19_exact,
        "cpu_tour_column_mismatches": cpu_row_errors,
        "gpu_tour_column_mismatches": gpu_row_errors,
        "expanded_tour_ids_equal_scheduling_chooser_ids": expanded_ids_match_scheduling,
        "capture_schedule_ids_equal_checkpoint": schedule_ids_match,
        "gpu_schedule_tdd_checkpoint_mismatches": schedule_tdd_mismatches,
        "cpu_schedule_choice_mismatches": int(
            sum(row["cpu_activitysim_choice_mismatches"] for row in schedule_validation)
        ),
        "gpu_schedule_choice_mismatches": int(
            sum(row["gpu_activitysim_choice_mismatches"] for row in schedule_validation)
        ),
        "cpu_gpu_schedule_choice_mismatches": int(
            sum(row["cpu_gpu_choice_mismatches"] for row in schedule_validation)
        ),
        "cpu_gpu_schedule_logsum_max_abs_error": cpu_gpu_logsum_error,
        "gpu_schedule_repeat_bit_exact": schedule_repeat_exact,
    }
    gates = {
        "upstream_exact": phase19_exact,
        "all_tour_columns_exact": not any(cpu_row_errors.values()) and not any(gpu_row_errors.values()),
        "tour_to_schedule_link_exact": expanded_ids_match_scheduling and schedule_ids_match,
        "all_schedule_choices_exact": schedule_tdd_mismatches == 0
        and correctness["cpu_schedule_choice_mismatches"] == 0
        and correctness["gpu_schedule_choice_mismatches"] == 0,
        "repeatable": schedule_repeat_exact,
        "resident_scheduling_faster_than_cpu": gpu_schedule_resident_median < cpu_schedule_median,
        "transfer_inclusive_scheduling_faster_than_cpu": gpu_schedule_inclusive_median < cpu_schedule_median,
    }
    report = {
        "phase": 20,
        "claim_scope": (
            "exact GPU mandatory-tour row expansion plus calibrated compact scheduling-kernel "
            "replay on the public 50,000-household MTC checkpoint"
        ),
        "exclusions": [
            "scheduling mode-choice-logsum preparation",
            "feasible-alternative and timetable-primitive preparation",
            "a fresh end-to-end ActivitySim run",
        ],
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
        "workload": {
            "households": 50000,
            "mandatory_persons": int(len(inputs["person_id"])),
            "mandatory_tours": int(len(gpu_tours["tour_id"])),
            "scheduling_batches": len(batches),
            "scheduling_interaction_rows": int(
                sum(row[0]["interaction_rows"] for row in batches)
            ),
        },
        "correctness": correctness,
        "scheduling_batches": schedule_validation,
        "timings_seconds": {
            "tour_expansion_cpu_samples": cpu_row_times,
            "tour_expansion_gpu_resident_samples": gpu_row_resident_times,
            "tour_expansion_gpu_transfer_inclusive_samples": gpu_row_inclusive_times,
            "scheduling_cpu_samples": cpu_schedule_times,
            "scheduling_gpu_resident_samples": gpu_schedule_resident_times,
            "scheduling_gpu_transfer_inclusive_samples": gpu_schedule_inclusive_times,
            "tour_expansion_cpu_median": cpu_row_median,
            "tour_expansion_gpu_resident_median": gpu_row_resident_median,
            "tour_expansion_gpu_transfer_inclusive_median": gpu_row_inclusive_median,
            "scheduling_cpu_median": cpu_schedule_median,
            "scheduling_gpu_resident_median": gpu_schedule_resident_median,
            "scheduling_gpu_transfer_inclusive_median": gpu_schedule_inclusive_median,
        },
        "speedup": {
            "tour_expansion_resident_gpu_vs_cpu": cpu_row_median / gpu_row_resident_median,
            "tour_expansion_transfer_inclusive_gpu_vs_cpu": cpu_row_median / gpu_row_inclusive_median,
            "scheduling_resident_gpu_vs_cpu": cpu_schedule_median / gpu_schedule_resident_median,
            "scheduling_transfer_inclusive_gpu_vs_cpu": cpu_schedule_median / gpu_schedule_inclusive_median,
        },
        "proof_gates": gates,
        "hashes": {
            "phase19_result_sha256": sha256(PHASE19),
            "frequency_checkpoint_sha256": sha256(
                args.pipeline / "tours" / "mandatory_tour_frequency.parquet"
            ),
            "scheduling_checkpoint_sha256": sha256(
                args.pipeline / "tours" / "mandatory_tour_scheduling.parquet"
            ),
            "capture_manifest_sha256": sha256(args.capture / "manifest.json"),
            "tour_expansion_source_sha256": sha256(
                ROOT / "src" / "choiceforge" / "tour_expansion.py"
            ),
            "scheduling_compiler_source_sha256": sha256(
                ROOT / "src" / "choiceforge" / "scheduling_compiler.py"
            ),
            "checkpoint_manifest_sha256": sha256(args.checkpoint_manifest),
        },
        "capture": capture_manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"workload": report["workload"], "correctness": correctness, "speedup": report["speedup"], "proof_gates": gates}, indent=2))
    if not all(gates.values()):
        raise SystemExit("Phase 20 proof gate failed")


if __name__ == "__main__":
    main()
