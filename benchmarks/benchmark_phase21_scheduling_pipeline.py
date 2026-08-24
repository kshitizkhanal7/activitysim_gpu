"""Qualify GPU-resident timetable and feasible-row scheduling preparation.

The measured boundary begins with per-tour chooser attributes and the compact
5-by-5 mode-choice-logsum cache.  It includes sequential feasibility filtering,
CSR construction, seven timetable primitives, scheduling choice, and timetable
mutation for all six mandatory-tour batches.  Logsum-cache creation from raw
network skims is named separately and is not hidden inside this timing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import time
from pathlib import Path

import numpy as np

from choiceforge.cuda_backend import _cupy
from choiceforge.gpu_scheduling_pipeline import (
    CompiledCpuSchedulingPreparer,
    GpuSchedulingPreparer,
)
from choiceforge.scheduling_compiler import (
    CompiledCpuSchedulingModel,
    CompiledCudaSchedulingModel,
    SchedulingSchema,
)


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "benchmark-results" / "phase21-scheduling-inputs"
SOURCE = ROOT / "benchmark-results" / "phase20-scheduling-replay"
BASELINE_LOG = (
    ROOT
    / "benchmark-data"
    / "phase9-mtc-full"
    / "prototype_mtc_extended"
    / "o-p17modeproof16-baseline-50000-1"
    / "activitysim.log"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def timed(operation, repetitions):
    samples = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
    return result, samples


def load_inputs(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / manifest["common_file"]) as loaded:
        common = {name: loaded[name] for name in loaded.files}
    batches = []
    for meta in manifest["batches"]:
        with np.load(path / meta["file"]) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        schema = SchedulingSchema(
            tuple(meta["chooser_columns"]),
            tuple(meta["row_columns"]),
            tuple(meta["alternative_columns"]),
        )
        batches.append(
            {
                "meta": meta,
                "data": data,
                "schema": schema,
                "cpu_model": CompiledCpuSchedulingModel(
                    meta["expressions"], data["coefficients"], schema
                ),
                "gpu_model": CompiledCudaSchedulingModel(
                    meta["expressions"], data["coefficients"], schema
                ),
            }
        )
    return manifest, common, batches


def load_source(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    batches = []
    for meta in manifest["batches"]:
        with np.load(path / meta["file"]) as loaded:
            batches.append({name: loaded[name] for name in loaded.files})
    return manifest, batches


def columns(meta):
    names = meta["chooser_columns"]
    return {
        "end_previous_column": names.index("end_previous"),
        "tour_count_column": names.index("tour_count"),
        "tour_num_column": names.index("tour_num"),
    }


def cpu_pipeline(preparer, common, batches, *, retain=False):
    preparer.reset()
    selected_batches = []
    prepared_batches = []
    for batch in batches:
        data = batch["data"]
        prepared = preparer.prepare(
            data["person_rows"],
            data["chooser_values"],
            data["mode_logsum_cache"],
            **columns(batch["meta"]),
        )
        result = batch["cpu_model"].choose(
            prepared.chooser_values,
            prepared.row_values,
            common["alternative_values"],
            prepared.alternative_ids,
            prepared.offsets,
            data["draws"],
        )
        selected = prepared.alternative_ids[prepared.offsets[:-1] + result.choices]
        preparer.assign(data["person_rows"], selected)
        selected_batches.append(selected)
        if retain:
            prepared_batches.append(prepared)
    return np.concatenate(selected_batches), prepared_batches


def gpu_pipeline(preparer, alternatives, batches, cp, *, retain=False):
    preparer.reset()
    selected_batches = []
    prepared_batches = []
    for batch in batches:
        data = batch["device"]
        prepared = preparer.prepare(
            data["person_rows"],
            data["chooser_values"],
            data["mode_logsum_cache"],
            **columns(batch["meta"]),
        )
        result = batch["gpu_model"].choose(
            prepared.chooser_values,
            prepared.row_values,
            alternatives,
            prepared.alternative_ids,
            prepared.offsets,
            data["draws"],
            return_device=True,
        )
        selected = prepared.alternative_ids[prepared.offsets[:-1] + result.choices]
        preparer.assign(data["person_rows"], selected)
        selected_batches.append(selected)
        if retain:
            prepared_batches.append(prepared)
    result = cp.concatenate(selected_batches)
    cp.cuda.Stream.null.synchronize()
    return result, prepared_batches


def mismatch_count(actual, expected) -> int:
    return int(np.count_nonzero(np.asarray(actual) != np.asarray(expected)))


def activitysim_component_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    match = re.search(
        r"time to execute run\.mandatory_tour_scheduling\s*:\s*(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    match = re.search(
        r"time to execute run\.mandatory_tour_scheduling\s*:\s*(\d+(?:\.\d+)?)\s+seconds",
        text,
    )
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase21-scheduling-pipeline.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "benchmark-results" / "phase21-device-checkpoint.json",
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    cp = _cupy()
    manifest, common, batches = load_inputs(args.inputs)
    source_manifest, source_batches = load_source(args.source)
    if len(batches) != len(source_batches):
        raise RuntimeError("Phase 20 and Phase 21 batch counts differ")

    cpu_preparer = CompiledCpuSchedulingPreparer(
        manifest["person_count"], common["alternative_values"]
    )
    alternatives_device = cp.asarray(common["alternative_values"])
    gpu_preparer = GpuSchedulingPreparer(manifest["person_count"], alternatives_device)
    for batch in batches:
        batch["device"] = {
            name: cp.asarray(value)
            for name, value in batch["data"].items()
            if name
            in {"person_rows", "chooser_values", "mode_logsum_cache", "draws"}
        }

    # Warm both Numba and CUDA compilation paths before any validation or timing.
    cpu_pipeline(cpu_preparer, common, batches)
    gpu_pipeline(gpu_preparer, alternatives_device, batches, cp)

    cpu_selected, cpu_prepared = cpu_pipeline(
        cpu_preparer, common, batches, retain=True
    )
    gpu_selected_device, gpu_prepared = gpu_pipeline(
        gpu_preparer, alternatives_device, batches, cp, retain=True
    )
    gpu_selected = cp.asnumpy(gpu_selected_device)

    expected_selected = np.concatenate(
        [batch["data"]["expected_tdd"] for batch in batches]
    )
    preparation = []
    total_mismatches = {
        "cpu_offsets": 0,
        "gpu_offsets": 0,
        "cpu_alternative_ids": 0,
        "gpu_alternative_ids": 0,
        "cpu_row_values": 0,
        "gpu_row_values": 0,
        "cpu_chooser_values": 0,
        "gpu_chooser_values": 0,
    }
    for batch, source, cpu_value, gpu_value in zip(
        batches, source_batches, cpu_prepared, gpu_prepared
    ):
        gpu_host = {
            name: cp.asnumpy(getattr(gpu_value, name))
            for name in ("offsets", "alternative_ids", "row_values", "chooser_values")
        }
        batch_errors = {}
        for name in ("offsets", "alternative_ids", "row_values", "chooser_values"):
            cpu_error = mismatch_count(getattr(cpu_value, name), source[name])
            gpu_error = mismatch_count(gpu_host[name], source[name])
            total_mismatches[f"cpu_{name}"] += cpu_error
            total_mismatches[f"gpu_{name}"] += gpu_error
            batch_errors[f"cpu_{name}"] = cpu_error
            batch_errors[f"gpu_{name}"] = gpu_error
        preparation.append(
            {
                "trace_label": batch["meta"]["trace_label"],
                "choosers": int(batch["data"]["chooser_ids"].size),
                "generated_interaction_rows": int(cpu_value.interaction_rows),
                "expected_interaction_rows": int(batch["meta"]["expected_interaction_rows"]),
                "mismatches": batch_errors,
            }
        )

    cpu_choice_mismatches = mismatch_count(cpu_selected, expected_selected)
    gpu_choice_mismatches = mismatch_count(gpu_selected, expected_selected)
    cpu_gpu_choice_mismatches = mismatch_count(cpu_selected, gpu_selected)

    source_prepared_bytes = 0
    for source in source_batches:
        source_prepared_bytes += sum(
            source[name].nbytes
            for name in ("offsets", "alternative_ids", "row_values")
        )

    # Correctness intentionally retains every generated row. Release those
    # proof arrays before benchmarking the steady-state reusable pipeline.
    del cpu_prepared, gpu_prepared, source_batches
    cp.get_default_memory_pool().free_all_blocks()

    _, cpu_times = timed(
        lambda: cpu_pipeline(cpu_preparer, common, batches)[0], args.repetitions
    )

    def resident_operation():
        return gpu_pipeline(gpu_preparer, alternatives_device, batches, cp)[0]

    # Warm allocator sizes as well as compiled kernels. Dynamic CSR lengths
    # make the first post-validation allocation unrepresentative.
    for _ in range(3):
        resident_operation()
    first_repeat = cp.asnumpy(resident_operation())
    _, gpu_resident_times = timed(resident_operation, args.repetitions)
    second_repeat = cp.asnumpy(resident_operation())

    def inclusive_operation():
        alternatives = cp.asarray(common["alternative_values"])
        inclusive_batches = []
        for batch in batches:
            item = dict(batch)
            item["device"] = {
                name: cp.asarray(batch["data"][name])
                for name in ("person_rows", "chooser_values", "mode_logsum_cache", "draws")
            }
            inclusive_batches.append(item)
        preparer = GpuSchedulingPreparer(manifest["person_count"], alternatives)
        selected, _ = gpu_pipeline(preparer, alternatives, inclusive_batches, cp)
        return cp.asnumpy(selected)

    _, gpu_inclusive_times = timed(inclusive_operation, args.repetitions)
    cpu_median = statistics.median(cpu_times)
    resident_median = statistics.median(gpu_resident_times)
    inclusive_median = statistics.median(gpu_inclusive_times)

    primitive_bytes = common["alternative_values"].nbytes
    for batch in batches:
        primitive_bytes += sum(
            batch["data"][name].nbytes
            for name in ("person_rows", "chooser_values", "mode_logsum_cache", "draws")
        )

    checkpoint = {
        "format_version": 1,
        "phase": 21,
        "checkpoint_name": "mandatory_tour_scheduling_prepared_on_device",
        "completed_components": [
            "mandatory_tour_frequency",
            "mandatory_tour_row_expansion",
            "mandatory_scheduling_feasibility",
            "mandatory_scheduling_timetable_primitives",
            "mandatory_scheduling_choice",
            "mandatory_scheduling_timetable_mutation",
        ],
        "rows": int(expected_selected.size),
        "tdd_sha256": array_sha256(gpu_selected),
        "timetable_sha256": array_sha256(cp.asnumpy(gpu_preparer.windows)),
        "source_input_manifest_sha256": sha256(args.inputs / "manifest.json"),
        "restart_limit": (
            "exact device scheduling restart from compact mode-logsum cache; "
            "raw network-skim-to-logsum cache construction remains a named upstream boundary"
        ),
    }
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2) + "\n")

    activitysim_seconds = activitysim_component_seconds(BASELINE_LOG)
    result = {
        "phase": 21,
        "claim_scope": (
            "six-batch mandatory scheduling from compact per-tour skim-period logsum "
            "cache through feasibility, timetable primitives, choice, and timetable mutation"
        ),
        "named_upstream_boundary": (
            "construction of each tour's 5-by-5 mode-choice-logsum cache from raw network skims"
        ),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
        },
        "workload": {
            "households": 50000,
            "tours": int(expected_selected.size),
            "batches": len(batches),
            "tdd_alternatives": int(common["alternative_values"].shape[0]),
            "generated_interaction_rows": int(sum(x["generated_interaction_rows"] for x in preparation)),
            "timetable_person_rows": int(manifest["person_count"]),
            "timetable_periods": int(gpu_preparer.windows.shape[1]),
            "source_prepared_bytes": source_prepared_bytes,
            "phase21_primitive_bytes": primitive_bytes,
            "input_reduction": source_prepared_bytes / primitive_bytes,
        },
        "correctness": {
            "preparation_mismatches": total_mismatches,
            "cpu_tdd_mismatches": cpu_choice_mismatches,
            "gpu_tdd_mismatches": gpu_choice_mismatches,
            "cpu_gpu_tdd_mismatches": cpu_gpu_choice_mismatches,
            "gpu_repeat_mismatches": mismatch_count(first_repeat, second_repeat),
        },
        "batch_validation": preparation,
        "timings_seconds": {
            "compiled_cpu_samples": cpu_times,
            "gpu_resident_samples": gpu_resident_times,
            "gpu_transfer_inclusive_samples": gpu_inclusive_times,
            "compiled_cpu_median": cpu_median,
            "gpu_resident_median": resident_median,
            "gpu_transfer_inclusive_median": inclusive_median,
        },
        "context_only": {
            "activitysim_component_seconds": activitysim_seconds,
            "comparison_warning": (
                "Not a speedup denominator: the ActivitySim component includes "
                "upstream raw-skim logsum and pandas work outside this benchmark."
            ),
        },
        "speedup": {
            "resident_gpu_vs_compiled_cpu": cpu_median / resident_median,
            "transfer_inclusive_gpu_vs_compiled_cpu": cpu_median / inclusive_median,
        },
        "proof_gates": {
            "all_cpu_preparation_exact": all(v == 0 for k, v in total_mismatches.items() if k.startswith("cpu_")),
            "all_gpu_preparation_exact": all(v == 0 for k, v in total_mismatches.items() if k.startswith("gpu_")),
            "all_cpu_choices_exact": cpu_choice_mismatches == 0,
            "all_gpu_choices_exact": gpu_choice_mismatches == 0,
            "cpu_gpu_choices_exact": cpu_gpu_choice_mismatches == 0,
            "gpu_repeatable": mismatch_count(first_repeat, second_repeat) == 0,
            "resident_gpu_faster_than_compiled_cpu": resident_median < cpu_median,
            "transfer_inclusive_gpu_faster_than_compiled_cpu": inclusive_median < cpu_median,
            "primitive_boundary_smaller_than_prepared_rows": primitive_bytes < source_prepared_bytes,
        },
        "hashes": {
            "phase21_input_manifest_sha256": sha256(args.inputs / "manifest.json"),
            "phase20_source_manifest_sha256": sha256(args.source / "manifest.json"),
            "pipeline_source_sha256": sha256(ROOT / "src" / "choiceforge" / "gpu_scheduling_pipeline.py"),
            "scheduling_compiler_source_sha256": sha256(ROOT / "src" / "choiceforge" / "scheduling_compiler.py"),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
    }
    if not all(result["proof_gates"].values()):
        raise AssertionError(f"Phase 21 proof gate failed: {result['proof_gates']}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "correctness": result["correctness"],
        "timings_seconds": {k: v for k, v in result["timings_seconds"].items() if not k.endswith("samples")},
        "speedup": result["speedup"],
        "proof_gates": result["proof_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
