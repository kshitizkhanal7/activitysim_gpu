"""Qualify a multi-component, device-resident public MTC vertical slice.

The measured graph joins the exact calibrated Phase 19 components to Phase 20
variable-row tour expansion and Phase 21 timetable scheduling. Immutable
compact scheduling logsum caches remain an explicit ingress boundary; no
modeled array crosses to the host between the first and final component.
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

from choiceforge.calibrated_chain import (
    choice_from_probabilities_gpu,
    evaluate_mnl_features,
    gather_by_key_gpu,
    key_rows_gpu,
    mnl_probabilities,
    mnl_utilities,
)
from choiceforge.cuda_backend import _cupy
from choiceforge.device_resident_runtime import DeviceResidentRuntime
from choiceforge.fused_mnl import FusedFixedMnlCudaModel
from choiceforge.gpu_native import DeviceTable, activitysim_uniforms_gpu
from choiceforge.gpu_scheduling_pipeline import (
    CompiledCpuSchedulingPreparer,
    GpuSchedulingPreparer,
)
from choiceforge.tour_expansion import TOUR_COLUMNS, mandatory_tours_cpu, mandatory_tours_gpu


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import benchmark_phase19_calibrated_chain as phase19  # noqa: E402
import benchmark_phase20_tour_chain as phase20  # noqa: E402
import benchmark_phase21_scheduling_pipeline as phase21  # noqa: E402


def file_sha256(path: Path) -> str:
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


def _frame_columns(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: frame[name].to_numpy(copy=True) for name in names}


def prepare_ingress(data, auto_spec, mtf_spec, tour_inputs, common, batches):
    households = data["households"]
    household_columns = {
        "household_id": households.index.to_numpy(copy=True),
        "home_zone_id": households.home_zone_id.to_numpy(copy=True),
    }
    household_columns.update(_frame_columns(households, phase19.AUTO_HOUSEHOLD_COLUMNS))

    household_state = data["household_state"]
    household_state_columns = {
        "household_id": household_state.index.to_numpy(copy=True)
    }
    household_state_columns.update(
        _frame_columns(household_state, phase19.MTF_HOUSEHOLD_COLUMNS)
    )

    persons = data["persons"]
    person_columns = {
        "person_id": persons.index.to_numpy(copy=True),
        "household_id": persons.household_id.to_numpy(copy=True),
        "mandatory": persons.cdap_activity.astype(str).eq("M").to_numpy(np.uint8),
    }
    person_columns.update(_frame_columns(persons, phase19.MTF_PERSON_COLUMNS))

    land_use = data["land_use"]
    land_columns = {"zone_id": land_use.index.to_numpy(copy=True)}
    land_columns.update(_frame_columns(land_use, phase19.AUTO_LAND_USE_COLUMNS))
    accessibility = data["accessibility"]
    access_columns = {"zone_id": accessibility.index.to_numpy(copy=True)}
    access_columns.update(_frame_columns(accessibility, phase19.AUTO_ACCESS_COLUMNS))

    tables: dict[str, dict[str, np.ndarray]] = {
        "households": household_columns,
        "household_state": household_state_columns,
        "persons": person_columns,
        "land_use": land_columns,
        "accessibility": access_columns,
        "auto_parameters": {"coefficients": auto_spec.coefficients},
        "tour_choosers": {
            name: value for name, value in tour_inputs.items() if name != "mtf_choice"
        },
        "schedule_common": {"alternative_values": common["alternative_values"]},
    }
    for number, batch in enumerate(batches):
        tables[f"schedule_batch_{number}"] = {
            name: batch["data"][name]
            for name in (
                "chooser_ids",
                "person_rows",
                "chooser_values",
                "mode_logsum_cache",
                "draws",
                "expected_tdd",
            )
        }
    return tables


def build_runtime(ingress, person_count):
    runtime = DeviceResidentRuntime()
    started = time.perf_counter()
    for name, columns in ingress.items():
        runtime.ingress_table(name, columns)
    runtime.seal_ingress()
    runtime.cp.cuda.Stream.null.synchronize()
    upload_seconds = time.perf_counter() - started
    started = time.perf_counter()
    alternatives = runtime.table("schedule_common").columns["alternative_values"]
    preparer = GpuSchedulingPreparer(person_count, alternatives)
    runtime.cp.cuda.Stream.null.synchronize()
    scheduler_initialization_seconds = time.perf_counter() - started
    return runtime, preparer, upload_seconds, scheduler_initialization_seconds


def compile_topology(runtime):
    """Build static device row maps once for all subsequent scenarios."""

    cp = runtime.cp
    started = time.perf_counter()

    def household_maps(tables):
        households = tables["households"]
        zones = households.columns["home_zone_id"]
        return {
            "household_join_map": {
                "household_row": cp.arange(zones.size, dtype=cp.int64),
                "land_row": key_rows_gpu(tables["land_use"].columns["zone_id"], zones),
                "access_row": key_rows_gpu(
                    tables["accessibility"].columns["zone_id"], zones
                ),
            }
        }

    runtime.run_stage(
        "topology.household_zone_maps",
        reads=("households", "land_use", "accessibility"),
        writes=("household_join_map",),
        operation=household_maps,
    )

    def mandatory_maps(tables):
        persons = tables["persons"]
        person_rows = cp.flatnonzero(persons.columns["mandatory"])
        household_ids = persons.columns["household_id"][person_rows]
        household_state_rows = key_rows_gpu(
            tables["household_state"].columns["household_id"], household_ids
        )
        static_values = cp.column_stack(
            [
                persons.columns[name][person_rows]
                for name in phase19.MTF_PERSON_COLUMNS
            ]
            + [
                tables["household_state"].columns[name][household_state_rows]
                for name in phase19.MTF_HOUSEHOLD_COLUMNS
            ]
        ).astype(cp.float64)
        return {
            "mandatory_person_map": {
                "person_row": person_rows.astype(cp.int64),
                "chooser_id": persons.columns["person_id"][person_rows],
                "static_values": static_values,
                "auto_household_row": key_rows_gpu(
                    tables["households"].columns["household_id"], household_ids
                ),
            }
        }

    runtime.run_stage(
        "topology.mandatory_person_maps",
        reads=("persons", "household_state", "households"),
        writes=("mandatory_person_map",),
        operation=mandatory_maps,
    )

    def tour_map(tables):
        person_rows = tables["mandatory_person_map"].columns["person_row"]
        mandatory_ids = tables["persons"].columns["person_id"][person_rows]
        return {
            "tour_choice_map": {
                "mtf_row": key_rows_gpu(
                    mandatory_ids, tables["tour_choosers"].columns["person_id"]
                )
            }
        }

    runtime.run_stage(
        "topology.tour_choice_map",
        reads=("persons", "mandatory_person_map", "tour_choosers"),
        writes=("tour_choice_map",),
        operation=tour_map,
    )
    runtime.synchronize()
    return time.perf_counter() - started


def execute_graph(runtime, preparer, auto_spec, mtf_model, batches, scenario):
    cp = runtime.cp
    suffix = f"scenario_{scenario}"

    def auto_features(tables):
        households = tables["households"]
        join_map = tables["household_join_map"].columns
        columns = {
            name: households.columns[name] for name in phase19.AUTO_HOUSEHOLD_COLUMNS
        }
        columns.update({
            name: tables["land_use"].columns[name][join_map["land_row"]]
            for name in phase19.AUTO_LAND_USE_COLUMNS
        })
        columns.update({
            name: tables["accessibility"].columns[name][join_map["access_row"]]
            for name in phase19.AUTO_ACCESS_COLUMNS
        })
        return {
            "auto_features": {
                "features": evaluate_mnl_features(
                    auto_spec, columns, cp, phase19.AUTO_CONSTANTS
                )
            }
        }

    runtime.run_stage(
        f"{suffix}.auto_features",
        reads=("households", "land_use", "accessibility", "household_join_map"),
        writes=("auto_features",),
        operation=auto_features,
    )

    def auto_choice(tables):
        utilities = mnl_utilities(
            tables["auto_features"].columns["features"],
            tables["auto_parameters"].columns["coefficients"],
            cp,
        )
        probabilities, logsums = mnl_probabilities(utilities, cp)
        draws = activitysim_uniforms_gpu(
            tables["households"].columns["household_id"],
            "households",
            "auto_ownership_simulate",
        )
        return {
            "auto_result": {
                "choice": choice_from_probabilities_gpu(probabilities, draws),
                "logsum": logsums,
            }
        }

    runtime.run_stage(
        f"{suffix}.auto_choice",
        reads=("auto_features", "auto_parameters", "households"),
        writes=("auto_result",),
        operation=auto_choice,
        replace="auto_result" in runtime.tables,
    )
    runtime.release_tables("auto_features")

    def mtf_choice(tables):
        state = tables["mandatory_person_map"].columns
        chooser_ids = state["chooser_id"]
        draws = activitysim_uniforms_gpu(
            chooser_ids, "persons", "mandatory_tour_frequency"
        )
        result = mtf_model.choose(
            state["static_values"],
            tables["auto_result"].columns["choice"][state["auto_household_row"]],
            draws,
        )
        return {
            "mtf_result": {
                "chooser_id": chooser_ids,
                "choice": result.choices,
                "logsum": result.logsums,
            }
        }

    runtime.run_stage(
        f"{suffix}.mandatory_frequency_choice",
        reads=("mandatory_person_map", "auto_result"),
        writes=("mtf_result",),
        operation=mtf_choice,
        replace="mtf_result" in runtime.tables,
    )
    def tours(tables):
        choosers = tables["tour_choosers"]
        choices = tables["mtf_result"].columns["choice"][
            tables["tour_choice_map"].columns["mtf_row"]
        ]
        table = mandatory_tours_gpu(
            choosers.columns["person_id"],
            choosers.columns["household_id"],
            choices,
            choosers.columns["is_worker"],
            choosers.columns["workplace_zone_id"],
            choosers.columns["school_zone_id"],
            choosers.columns["home_zone_id"],
        )
        return {"mandatory_tours": table}

    runtime.run_stage(
        f"{suffix}.mandatory_tour_expansion",
        reads=("tour_choosers", "mtf_result", "tour_choice_map"),
        writes=("mandatory_tours",),
        operation=tours,
        replace="mandatory_tours" in runtime.tables,
    )

    batch_names = tuple(f"schedule_batch_{i}" for i in range(len(batches)))

    def link_tours(tables):
        generated = tables["mandatory_tours"].columns["tour_id"]
        target = cp.concatenate(
            [tables[name].columns["chooser_ids"] for name in batch_names]
        )
        linked = gather_by_key_gpu(
            generated,
            target,
            {"generated_row": cp.arange(generated.size, dtype=cp.int64)},
        )
        return {
            "tour_schedule_link": {
                "tour_id": target,
                "generated_row": linked["generated_row"],
            }
        }

    runtime.run_stage(
        f"{suffix}.tour_schedule_link",
        reads=("mandatory_tours",) + batch_names,
        writes=("tour_schedule_link",),
        operation=link_tours,
        replace="tour_schedule_link" in runtime.tables,
    )

    def schedule(tables):
        preparer.reset()
        selected_batches = []
        for number, batch in enumerate(batches):
            data = tables[f"schedule_batch_{number}"].columns
            prepared = preparer.prepare(
                data["person_rows"],
                data["chooser_values"],
                data["mode_logsum_cache"],
                **phase21.columns(batch["meta"]),
            )
            result = batch["gpu_model"].choose(
                prepared.chooser_values,
                prepared.row_values,
                tables["schedule_common"].columns["alternative_values"],
                prepared.alternative_ids,
                prepared.offsets,
                data["draws"],
                return_device=True,
            )
            selected = prepared.alternative_ids[
                prepared.offsets[:-1] + result.choices
            ]
            preparer.assign(data["person_rows"], selected)
            selected_batches.append(selected)
        return {
            "schedule_result": {
                "tour_id": tables["tour_schedule_link"].columns["tour_id"],
                "tdd": cp.concatenate(selected_batches),
            },
            "timetable_state": {
                "person_id": cp.arange(preparer.windows.shape[0], dtype=cp.int64),
                "window": preparer.windows.copy(),
                "previous_tdd": preparer.previous_tdd.copy(),
            },
        }

    runtime.run_stage(
        f"{suffix}.mandatory_scheduling",
        reads=("tour_schedule_link", "schedule_common") + batch_names,
        writes=("schedule_result", "timetable_state"),
        operation=schedule,
        replace="schedule_result" in runtime.tables,
    )
    # Both outputs always coexist, so replacement state must be identical.
    if runtime.versions["schedule_result"] != runtime.versions["timetable_state"]:
        raise RuntimeError("schedule and timetable state versions diverged")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=phase19.DEFAULT_PIPELINE)
    parser.add_argument("--inputs", type=Path, default=phase21.INPUTS)
    parser.add_argument("--source", type=Path, default=phase21.SOURCE)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase23-device-resident.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "benchmark-results" / "phase23-device-checkpoint",
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    cp = _cupy()
    data = phase19.load_inputs(args.pipeline)
    auto_spec, mtf_spec = phase19.load_specs()
    mtf_model = FusedFixedMnlCudaModel(
        mtf_spec.expressions,
        mtf_spec.coefficients,
        phase19.MTF_PERSON_COLUMNS + phase19.MTF_HOUSEHOLD_COLUMNS,
    )
    tour_inputs, tour_checkpoint = phase20.load_tour_inputs(args.pipeline)
    manifest, common, batches = phase21.load_inputs(args.inputs)
    ingress = prepare_ingress(
        data, auto_spec, mtf_spec, tour_inputs, common, batches
    )

    if not np.array_equal(
        tour_inputs["person_id"],
        data["persons"].index.to_numpy()[
            data["persons"].cdap_activity.astype(str).eq("M").to_numpy()
        ],
    ):
        raise AssertionError("Phase 19 and Phase 20 mandatory-person order differs")

    cpu_preparer = CompiledCpuSchedulingPreparer(
        manifest["person_count"], common["alternative_values"]
    )

    def cpu_operation():
        # The public spec uses np.where around a protected division. NumPy
        # evaluates both branches eagerly; invalid inactive rows are expected.
        with np.errstate(divide="ignore", invalid="ignore"):
            calibrated = phase19.cpu_pipeline(
                data, auto_spec, mtf_spec, capture=False
            )
        computed_tours = mandatory_tours_cpu(
            tour_inputs["person_id"],
            tour_inputs["household_id"],
            calibrated["mtf_choice"],
            tour_inputs["is_worker"],
            tour_inputs["workplace_zone_id"],
            tour_inputs["school_zone_id"],
            tour_inputs["home_zone_id"],
        )
        selected, _ = phase21.cpu_pipeline(cpu_preparer, common, batches)
        return calibrated, computed_tours, selected, cpu_preparer.windows.copy()

    # Warm CPU compilation and all CUDA kernels/allocator sizes before timing.
    cpu_reference = cpu_operation()
    runtime, preparer, upload_seconds, scheduler_initialization_seconds = build_runtime(
        ingress, manifest["person_count"]
    )
    topology_seconds = compile_topology(runtime)
    setup_seconds = upload_seconds + scheduler_initialization_seconds + topology_seconds
    execute_graph(runtime, preparer, auto_spec, mtf_model, batches, "warmup")
    runtime.synchronize()
    repeat_reference = {
        "auto_choice": runtime.table("auto_result").columns["choice"].copy(),
        "mtf_choice": runtime.table("mtf_result").columns["choice"].copy(),
        "tour_id": runtime.table("mandatory_tours").columns["tour_id"].copy(),
        "tdd": runtime.table("schedule_result").columns["tdd"].copy(),
    }

    cpu_samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        cpu_operation()
        cpu_samples.append(time.perf_counter() - started)

    gpu_samples = []
    repeat_mismatches = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        execute_graph(runtime, preparer, auto_spec, mtf_model, batches, repetition)
        runtime.synchronize()
        gpu_samples.append(time.perf_counter() - started)
        repeat_mismatches.append(
            {
                "auto_choice": int(
                    cp.count_nonzero(
                        runtime.table("auto_result").columns["choice"]
                        != repeat_reference["auto_choice"]
                    ).item()
                ),
                "mtf_choice": int(
                    cp.count_nonzero(
                        runtime.table("mtf_result").columns["choice"]
                        != repeat_reference["mtf_choice"]
                    ).item()
                ),
                "tour_id": int(
                    cp.count_nonzero(
                        runtime.table("mandatory_tours").columns["tour_id"]
                        != repeat_reference["tour_id"]
                    ).item()
                ),
                "tdd": int(
                    cp.count_nonzero(
                        runtime.table("schedule_result").columns["tdd"]
                        != repeat_reference["tdd"]
                    ).item()
                ),
            }
        )

    publish_started = time.perf_counter()
    published = runtime.publish(
        {
            "auto_result": ("choice", "logsum"),
            "mtf_result": ("chooser_id", "choice", "logsum"),
            "mandatory_tours": TOUR_COLUMNS,
            "schedule_result": ("tour_id", "tdd"),
            "timetable_state": ("window", "previous_tdd"),
        }
    )
    publication_seconds = time.perf_counter() - publish_started
    runtime.assert_resident_contract()

    checkpoint_started = time.perf_counter()
    checkpoint_manifest = runtime.checkpoint(
        args.checkpoint,
        tables=(
            "auto_result",
            "mtf_result",
            "mandatory_tours",
            "schedule_result",
            "timetable_state",
        ),
        metadata={
            "phase": 23,
            "public_households": len(data["households"]),
            "named_ingress_boundary": "compact 5x5 scheduling mode-logsum caches",
        },
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    restored = DeviceResidentRuntime.restore(args.checkpoint)
    restored_schedule = restored.publish({"schedule_result": ("tour_id", "tdd")})[
        "schedule_result"
    ]

    auto_expected = data["auto_reference"].to_numpy(dtype=np.int32)
    mtf_expected = pd.Categorical(
        data["mtf_reference"].reindex(published["mtf_result"]["chooser_id"]),
        categories=list(mtf_spec.alternatives),
    ).codes.astype(np.int32)
    tour_errors = phase20.row_mismatches(published["mandatory_tours"], tour_checkpoint)
    expected_tdd = np.concatenate([batch["data"]["expected_tdd"] for batch in batches])
    expected_tour_ids = np.concatenate([batch["data"]["chooser_ids"] for batch in batches])
    correctness = {
        "auto_checkpoint_mismatches": int(
            np.count_nonzero(published["auto_result"]["choice"] != auto_expected)
        ),
        "mandatory_frequency_checkpoint_mismatches": int(
            np.count_nonzero(published["mtf_result"]["choice"] != mtf_expected)
        ),
        "auto_logsum_max_abs_error": float(
            np.max(
                np.abs(
                    published["auto_result"]["logsum"]
                    - cpu_reference[0]["auto_logsum"]
                )
            )
        ),
        "fused_mtf_logsum_max_abs_error": float(
            np.max(
                np.abs(
                    published["mtf_result"]["logsum"]
                    - cpu_reference[0]["mtf_logsum"]
                )
            )
        ),
        "tour_column_mismatches": tour_errors,
        "scheduled_tour_id_mismatches": int(
            np.count_nonzero(published["schedule_result"]["tour_id"] != expected_tour_ids)
        ),
        "tdd_mismatches": int(
            np.count_nonzero(published["schedule_result"]["tdd"] != expected_tdd)
        ),
        "timetable_mismatches_vs_cpu": int(
            np.count_nonzero(published["timetable_state"]["window"] != cpu_reference[3])
        ),
        "restart_tour_id_mismatches": int(
            np.count_nonzero(restored_schedule["tour_id"] != expected_tour_ids)
        ),
        "restart_tdd_mismatches": int(
            np.count_nonzero(restored_schedule["tdd"] != expected_tdd)
        ),
        "repeat_mismatches": repeat_mismatches,
    }
    cpu_median = statistics.median(cpu_samples)
    gpu_median = statistics.median(gpu_samples)
    telemetry = runtime.telemetry_dict()
    final_iteration_stages = telemetry["stages"][-6:]
    gates = {
        "calibrated_choices_exact": (
            correctness["auto_checkpoint_mismatches"] == 0
            and correctness["mandatory_frequency_checkpoint_mismatches"] == 0
        ),
        "fused_mnl_logsums_within_1e_10": (
            correctness["auto_logsum_max_abs_error"] <= 1.0e-10
            and correctness["fused_mtf_logsum_max_abs_error"] <= 1.0e-10
        ),
        "variable_tour_table_exact": all(value == 0 for value in tour_errors.values()),
        "scheduling_exact": (
            correctness["scheduled_tour_id_mismatches"] == 0
            and correctness["tdd_mismatches"] == 0
            and correctness["timetable_mismatches_vs_cpu"] == 0
        ),
        "restart_exact": (
            correctness["restart_tour_id_mismatches"] == 0
            and correctness["restart_tdd_mismatches"] == 0
        ),
        "all_measured_repeats_bit_exact": all(
            all(value == 0 for value in item.values())
            for item in correctness["repeat_mismatches"]
        ),
        "no_postseal_modeled_transfers": (
            telemetry["forbidden_postseal_host_bytes"] == 0
        ),
        "no_modeled_cpu_fallbacks": telemetry["modeled_cpu_fallbacks"] == 0,
        "single_final_publication": telemetry["publication_calls"] == 1,
        "gpu_resident_faster_than_cpu": gpu_median < cpu_median,
        "all_stage_device_timings_recorded": all(
            item["device_seconds"] is not None for item in final_iteration_stages
        ),
    }
    report = {
        "phase": 23,
        "claim_scope": (
            "one sealed device graph for calibrated auto ownership, mandatory tour "
            "frequency, variable-row mandatory tours, ID linkage, and six-batch "
            "timetable scheduling"
        ),
        "named_ingress_boundary": (
            "upstream location/CDAP state and compact 5x5 scheduling mode-logsum caches"
        ),
        "not_claimed": (
            "a complete ActivitySim model or raw-network-skim generation inside the "
            "resident runtime"
        ),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
        },
        "workload": {
            "households": len(data["households"]),
            "persons": len(data["persons"]),
            "mandatory_persons": int(published["mtf_result"]["choice"].size),
            "mandatory_tours": int(published["schedule_result"]["tdd"].size),
            "scheduling_batches": len(batches),
            "scheduling_interaction_rows": int(
                sum(meta["expected_interaction_rows"] for meta in manifest["batches"])
            ),
            "published_columns": 21,
            "fused_mtf_expressions": len(mtf_spec.expressions),
        },
        "timings_seconds": {
            "cpu_modeled_samples": cpu_samples,
            "gpu_resident_samples": gpu_samples,
            "cpu_modeled_median": cpu_median,
            "gpu_resident_median": gpu_median,
            "one_time_input_upload": upload_seconds,
            "one_time_scheduler_initialization": scheduler_initialization_seconds,
            "one_time_device_topology_compile": topology_seconds,
            "one_time_setup_total": setup_seconds,
            "final_publication": publication_seconds,
            "checkpoint_write": checkpoint_seconds,
            "single_run_setup_compute_publication": (
                setup_seconds + gpu_median + publication_seconds
            ),
        },
        "speedup": {
            "resident_gpu_vs_cpu_modeled": cpu_median / gpu_median,
            "transfer_inclusive_gpu_vs_cpu_modeled": cpu_median
            / (setup_seconds + gpu_median + publication_seconds),
            "ten_repeated_runs_amortized": (10 * cpu_median)
            / (setup_seconds + 10 * gpu_median + publication_seconds),
            "hundred_repeated_runs_amortized": (100 * cpu_median)
            / (setup_seconds + 100 * gpu_median + publication_seconds),
        },
        "correctness": correctness,
        "telemetry": telemetry,
        "checkpoint": {
            "tables": list(checkpoint_manifest["tables"]),
            "archive_sha256": checkpoint_manifest["state_archive_sha256"],
            "schedule_tdd_sha256": array_sha256(
                published["schedule_result"]["tdd"]
            ),
        },
        "proof_gates": gates,
        "hashes": {
            "phase21_input_manifest": file_sha256(args.inputs / "manifest.json"),
            "phase20_source_manifest": file_sha256(args.source / "manifest.json"),
            "runtime_source": file_sha256(
                ROOT / "src" / "choiceforge" / "device_resident_runtime.py"
            ),
            "benchmark_source": file_sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "correctness": correctness,
                "timings_seconds": {
                    key: value
                    for key, value in report["timings_seconds"].items()
                    if not key.endswith("samples")
                },
                "speedup": report["speedup"],
                "proof_gates": gates,
            },
            indent=2,
        )
    )
    if not all(gates.values()):
        raise SystemExit("Phase 23 device-resident proof gate failed")


if __name__ == "__main__":
    main()
