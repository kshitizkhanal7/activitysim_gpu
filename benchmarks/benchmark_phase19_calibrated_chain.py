"""Qualify a calibrated public MTC household-to-person chain on the GPU.

The replay starts from published ActivitySim checkpoints immediately before
auto ownership and mandatory tour frequency.  Auto ownership is recomputed and
fed into the person model on the device; no checkpoint auto choice is used by
the GPU path.  Upstream school/work location and CDAP state remain frozen
public reference inputs, so this is a calibrated two-component replay rather
than a complete ActivitySim model run.
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
from typing import Any

import numpy as np
import pandas as pd

from choiceforge.calibrated_chain import (
    ResolvedMnlSpec,
    choice_from_probabilities_cpu,
    choice_from_probabilities_gpu,
    evaluate_mnl_features,
    gather_by_key_gpu,
    mnl_probabilities,
    mnl_utilities,
    resolve_activitysim_mnl_spec,
)
from choiceforge.cuda_backend import _cupy
from choiceforge.gpu_native import (
    DeviceTable,
    GpuNativeRuntime,
    activitysim_uniforms_cpu,
    activitysim_uniforms_gpu,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended"
DEFAULT_PIPELINE = PUBLIC_ROOT / "o-p17modeproof16-baseline-50000-1" / "pipeline.parquetpipeline"
CONFIG = PUBLIC_ROOT / "configs"

AUTO_CONSTANTS = {
    "ID_SAN_FRANCISCO": 1,
    "ID_SAN_MATEO": 2,
    "ID_SANTA_CLARA": 3,
    "ID_ALAMEDA": 4,
    "ID_CONTRA_COSTA": 5,
    "ID_SOLANO": 6,
    "ID_NAPA": 7,
    "ID_SONOMA": 8,
    "ID_MARIN": 9,
}
AUTO_HOUSEHOLD_COLUMNS = (
    "num_drivers",
    "num_children_16_to_17",
    "num_children_5_to_15",
    "num_college_age",
    "num_young_adults",
    "num_young_children",
    "num_workers",
    "income_in_thousands",
    "hh_work_auto_savings_ratio",
)
AUTO_LAND_USE_COLUMNS = ("county_id", "density_index")
AUTO_ACCESS_COLUMNS = (
    "auPkRetail",
    "auOpRetail",
    "trPkRetail",
    "trOpRetail",
    "nmRetail",
)
MTF_PERSON_COLUMNS = (
    "age",
    "distance_to_school",
    "distance_to_work",
    "female",
    "nonstudent_to_school",
    "ptype",
    "roundtrip_auto_time_to_school",
    "roundtrip_auto_time_to_work",
    "school_zone_id",
    "student_is_employed",
    "workplace_zone_id",
)
MTF_HOUSEHOLD_COLUMNS = (
    "home_is_urban",
    "income_in_thousands",
    "non_family",
    "num_drivers",
    "num_non_workers",
    "num_under16_not_at_school",
    "num_young_children",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_specs() -> tuple[ResolvedMnlSpec, ResolvedMnlSpec]:
    auto = resolve_activitysim_mnl_spec(
        "auto_ownership",
        CONFIG / "auto_ownership.csv",
        CONFIG / "auto_ownership_coefficients.csv",
    )
    mtf = resolve_activitysim_mnl_spec(
        "mandatory_tour_frequency",
        CONFIG / "mandatory_tour_frequency.csv",
        CONFIG / "mandatory_tour_frequency_coefficients.csv",
    )
    return auto, mtf


def load_inputs(pipeline: Path) -> dict[str, Any]:
    households = pd.read_parquet(pipeline / "households" / "workplace_location.parquet")
    household_state = pd.read_parquet(pipeline / "households" / "cdap_simulate.parquet")
    persons = pd.read_parquet(pipeline / "persons" / "cdap_simulate.parquet")
    land_use = pd.read_parquet(pipeline / "land_use" / "initialize_landuse.parquet")
    accessibility = pd.read_parquet(
        pipeline / "accessibility" / "compute_accessibility.parquet"
    )
    auto_reference = pd.read_parquet(
        pipeline / "households" / "auto_ownership_simulate.parquet"
    )["auto_ownership"].reindex(households.index)
    mtf_reference = pd.read_parquet(
        pipeline / "persons" / "mandatory_tour_frequency.parquet"
    )["mandatory_tour_frequency"].reindex(persons.index)
    return {
        "households": households,
        "household_state": household_state,
        "persons": persons,
        "land_use": land_use,
        "accessibility": accessibility,
        "auto_reference": auto_reference,
        "mtf_reference": mtf_reference,
    }


def _cpu_lookup(source: pd.DataFrame, keys: np.ndarray, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: source[name].reindex(keys).to_numpy() for name in columns}


def cpu_pipeline(
    data: dict[str, Any],
    auto_spec: ResolvedMnlSpec,
    mtf_spec: ResolvedMnlSpec,
    *,
    capture: bool,
) -> dict[str, Any]:
    households = data["households"]
    zones = households["home_zone_id"].to_numpy()
    auto_columns = {
        name: households[name].to_numpy() for name in AUTO_HOUSEHOLD_COLUMNS
    }
    auto_columns.update(_cpu_lookup(data["land_use"], zones, AUTO_LAND_USE_COLUMNS))
    auto_columns.update(_cpu_lookup(data["accessibility"], zones, AUTO_ACCESS_COLUMNS))
    auto_features = evaluate_mnl_features(auto_spec, auto_columns, np, AUTO_CONSTANTS)
    auto_utilities = mnl_utilities(auto_features, auto_spec.coefficients, np)
    auto_probs, auto_logsums = mnl_probabilities(auto_utilities, np)
    auto_draws = activitysim_uniforms_cpu(
        households.index.to_numpy(), "households", "auto_ownership_simulate"
    )
    auto_choices = choice_from_probabilities_cpu(auto_probs, auto_draws)

    persons = data["persons"]
    mandatory = persons["cdap_activity"].astype(str).eq("M").to_numpy()
    chooser_ids = persons.index.to_numpy()[mandatory]
    household_ids = persons["household_id"].to_numpy()[mandatory]
    mtf_columns = {
        name: persons[name].to_numpy()[mandatory] for name in MTF_PERSON_COLUMNS
    }
    mtf_columns.update(
        _cpu_lookup(data["household_state"], household_ids, MTF_HOUSEHOLD_COLUMNS)
    )
    auto_by_household = pd.Series(auto_choices, index=households.index)
    mtf_columns["auto_ownership"] = auto_by_household.reindex(household_ids).to_numpy()
    mtf_features = evaluate_mnl_features(mtf_spec, mtf_columns, np)
    mtf_utilities = mnl_utilities(mtf_features, mtf_spec.coefficients, np)
    mtf_probs, mtf_logsums = mnl_probabilities(mtf_utilities, np)
    mtf_draws = activitysim_uniforms_cpu(
        chooser_ids, "persons", "mandatory_tour_frequency"
    )
    mtf_choices = choice_from_probabilities_cpu(mtf_probs, mtf_draws)
    result = {
        "auto_choice": auto_choices,
        "auto_logsum": auto_logsums,
        "mtf_chooser_id": chooser_ids,
        "mtf_choice": mtf_choices,
        "mtf_logsum": mtf_logsums,
    }
    if capture:
        result.update(
            auto_features=auto_features,
            auto_utilities=auto_utilities,
            auto_probabilities=auto_probs,
            auto_draws=auto_draws,
            mtf_features=mtf_features,
            mtf_utilities=mtf_utilities,
            mtf_probabilities=mtf_probs,
            mtf_draws=mtf_draws,
        )
    return result


def _ingress_columns(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: frame[name].to_numpy(copy=True) for name in names}


def gpu_pipeline(
    data: dict[str, Any],
    auto_spec: ResolvedMnlSpec,
    mtf_spec: ResolvedMnlSpec,
    *,
    capture: bool,
) -> dict[str, Any]:
    cp = _cupy()
    runtime = GpuNativeRuntime()
    total_start = time.perf_counter()
    households = data["households"]
    household_input = {"household_id": households.index.to_numpy(copy=True)}
    household_input["home_zone_id"] = households["home_zone_id"].to_numpy(copy=True)
    household_input.update(_ingress_columns(households, AUTO_HOUSEHOLD_COLUMNS))
    household_table = runtime.ingress_table("households", household_input)

    household_state = data["household_state"]
    state_input = {"household_id": household_state.index.to_numpy(copy=True)}
    state_input.update(_ingress_columns(household_state, MTF_HOUSEHOLD_COLUMNS))
    household_state_table = runtime.ingress_table("household_state", state_input)

    persons = data["persons"]
    person_input = {
        "person_id": persons.index.to_numpy(copy=True),
        "household_id": persons["household_id"].to_numpy(copy=True),
        "mandatory": persons["cdap_activity"].astype(str).eq("M").to_numpy(np.uint8),
    }
    person_input.update(_ingress_columns(persons, MTF_PERSON_COLUMNS))
    person_table = runtime.ingress_table("persons", person_input)

    land_use = data["land_use"]
    land_input = {"zone_id": land_use.index.to_numpy(copy=True)}
    land_input.update(_ingress_columns(land_use, AUTO_LAND_USE_COLUMNS))
    land_table = runtime.ingress_table("land_use", land_input)
    accessibility = data["accessibility"]
    access_input = {"zone_id": accessibility.index.to_numpy(copy=True)}
    access_input.update(_ingress_columns(accessibility, AUTO_ACCESS_COLUMNS))
    access_table = runtime.ingress_table("accessibility", access_input)
    auto_parameter_table = runtime.ingress_table(
        "auto_parameters", {"coefficients": auto_spec.coefficients}
    )
    mtf_parameter_table = runtime.ingress_table(
        "mtf_parameters", {"coefficients": mtf_spec.coefficients}
    )
    runtime.seal_ingress()
    cp.cuda.Stream.null.synchronize()
    compute_start = time.perf_counter()

    def auto_feature_stage() -> dict[str, Any]:
        columns = {
            name: household_table.columns[name] for name in AUTO_HOUSEHOLD_COLUMNS
        }
        columns.update(
            gather_by_key_gpu(
                land_table.columns["zone_id"],
                household_table.columns["home_zone_id"],
                {name: land_table.columns[name] for name in AUTO_LAND_USE_COLUMNS},
            )
        )
        columns.update(
            gather_by_key_gpu(
                access_table.columns["zone_id"],
                household_table.columns["home_zone_id"],
                {name: access_table.columns[name] for name in AUTO_ACCESS_COLUMNS},
            )
        )
        return {"features": evaluate_mnl_features(auto_spec, columns, cp, AUTO_CONSTANTS)}

    auto_features = runtime.run_stage("auto_gpu_joins_and_29_expressions", auto_feature_stage)

    def auto_mnl_stage() -> dict[str, Any]:
        utilities = mnl_utilities(
            auto_features.columns["features"],
            auto_parameter_table.columns["coefficients"],
            cp,
        )
        probabilities, logsums = mnl_probabilities(utilities, cp)
        draws = activitysim_uniforms_gpu(
            household_table.columns["household_id"],
            "households",
            "auto_ownership_simulate",
        )
        choices = choice_from_probabilities_gpu(probabilities, draws)
        return {
            "utility": utilities,
            "probability": probabilities,
            "draw": draws,
            "choice": choices,
            "logsum": logsums,
        }

    auto_result = runtime.run_stage("calibrated_auto_ownership_mnl", auto_mnl_stage)

    def mtf_feature_stage() -> dict[str, Any]:
        mask = person_table.columns["mandatory"].astype(cp.bool_)
        chooser_ids = person_table.columns["person_id"][mask]
        household_ids = person_table.columns["household_id"][mask]
        columns = {name: person_table.columns[name][mask] for name in MTF_PERSON_COLUMNS}
        columns.update(
            gather_by_key_gpu(
                household_state_table.columns["household_id"],
                household_ids,
                {
                    name: household_state_table.columns[name]
                    for name in MTF_HOUSEHOLD_COLUMNS
                },
            )
        )
        columns["auto_ownership"] = gather_by_key_gpu(
            household_table.columns["household_id"],
            household_ids,
            {"auto_ownership": auto_result.columns["choice"]},
        )["auto_ownership"]
        features = evaluate_mnl_features(mtf_spec, columns, cp)
        return {"chooser_id": chooser_ids, "features": features}

    mtf_features = runtime.run_stage(
        "household_to_person_gpu_join_and_98_expressions", mtf_feature_stage
    )

    def mtf_mnl_stage() -> dict[str, Any]:
        utilities = mnl_utilities(
            mtf_features.columns["features"],
            mtf_parameter_table.columns["coefficients"],
            cp,
        )
        probabilities, logsums = mnl_probabilities(utilities, cp)
        draws = activitysim_uniforms_gpu(
            mtf_features.columns["chooser_id"],
            "persons",
            "mandatory_tour_frequency",
        )
        choices = choice_from_probabilities_gpu(probabilities, draws)
        return {
            "choice": choices,
            "logsum": logsums,
            "utility": utilities,
            "probability": probabilities,
            "draw": draws,
        }

    mtf_result = runtime.run_stage("calibrated_mandatory_tour_frequency_mnl", mtf_mnl_stage)
    cp.cuda.Stream.null.synchronize()
    compute_seconds = time.perf_counter() - compute_start
    runtime.assert_gpu_only()
    final_auto = runtime.egress_table(auto_result, ("choice", "logsum"))
    final_mtf = runtime.egress_table(mtf_result, ("choice", "logsum"))
    chooser_ids = runtime.egress_table(mtf_features, ("chooser_id",))["chooser_id"]
    result = {
        "auto_choice": final_auto["choice"],
        "auto_logsum": final_auto["logsum"],
        "mtf_chooser_id": chooser_ids,
        "mtf_choice": final_mtf["choice"],
        "mtf_logsum": final_mtf["logsum"],
        "compute_seconds": compute_seconds,
        "telemetry": runtime.telemetry,
    }
    if capture:
        result.update(
            auto_features=runtime.egress_table(auto_features)["features"],
            auto_utilities=runtime.egress_table(auto_result, ("utility",))["utility"],
            auto_probabilities=runtime.egress_table(auto_result, ("probability",))["probability"],
            auto_draws=runtime.egress_table(auto_result, ("draw",))["draw"],
            mtf_features=runtime.egress_table(mtf_features, ("features",))["features"],
            mtf_utilities=runtime.egress_table(mtf_result, ("utility",))["utility"],
            mtf_probabilities=runtime.egress_table(mtf_result, ("probability",))["probability"],
            mtf_draws=runtime.egress_table(mtf_result, ("draw",))["draw"],
        )
    result["total_seconds"] = time.perf_counter() - total_start
    return result


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _reference_choice_indices(
    values: pd.Series, alternatives: tuple[str, ...], chooser_ids: np.ndarray
) -> np.ndarray:
    selected = values.reindex(chooser_ids)
    return pd.Categorical(selected, categories=list(alternatives)).codes.astype(np.int32)


def machine() -> dict[str, Any]:
    cp = _cupy()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": properties["name"].decode()
        if isinstance(properties["name"], bytes)
        else properties["name"],
        "device_total_bytes": int(properties["totalGlobalMem"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "phase19-calibrated-chain.json",
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")

    data = load_inputs(args.pipeline)
    auto_spec, mtf_spec = load_specs()
    # Compile CUDA and Numba/CuPy internals outside reported samples.
    gpu_pipeline(data, auto_spec, mtf_spec, capture=False)
    cpu_reference = cpu_pipeline(data, auto_spec, mtf_spec, capture=True)
    gpu_reference = gpu_pipeline(data, auto_spec, mtf_spec, capture=True)

    cpu_times: list[float] = []
    gpu_compute_times: list[float] = []
    gpu_total_times: list[float] = []
    repeated_gpu_choices: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        cpu_pipeline(data, auto_spec, mtf_spec, capture=False)
        cpu_times.append(time.perf_counter() - start)
        gpu_result = gpu_pipeline(data, auto_spec, mtf_spec, capture=False)
        gpu_compute_times.append(gpu_result["compute_seconds"])
        gpu_total_times.append(gpu_result["total_seconds"])
        repeated_gpu_choices.append((gpu_result["auto_choice"], gpu_result["mtf_choice"]))

    auto_checkpoint = data["auto_reference"].to_numpy(dtype=np.int32)
    mtf_checkpoint = _reference_choice_indices(
        data["mtf_reference"], mtf_spec.alternatives, cpu_reference["mtf_chooser_id"]
    )
    correctness = {
        "cpu_auto_checkpoint_mismatches": int(
            np.count_nonzero(cpu_reference["auto_choice"] != auto_checkpoint)
        ),
        "gpu_auto_checkpoint_mismatches": int(
            np.count_nonzero(gpu_reference["auto_choice"] != auto_checkpoint)
        ),
        "cpu_mtf_checkpoint_mismatches": int(
            np.count_nonzero(cpu_reference["mtf_choice"] != mtf_checkpoint)
        ),
        "gpu_mtf_checkpoint_mismatches": int(
            np.count_nonzero(gpu_reference["mtf_choice"] != mtf_checkpoint)
        ),
        "auto_feature_max_abs_error": _max_abs(
            cpu_reference["auto_features"], gpu_reference["auto_features"]
        ),
        "auto_utility_max_abs_error": _max_abs(
            cpu_reference["auto_utilities"], gpu_reference["auto_utilities"]
        ),
        "auto_probability_max_abs_error": _max_abs(
            cpu_reference["auto_probabilities"], gpu_reference["auto_probabilities"]
        ),
        "auto_draws_bit_exact": bool(
            np.array_equal(cpu_reference["auto_draws"], gpu_reference["auto_draws"])
        ),
        "mtf_feature_max_abs_error": _max_abs(
            cpu_reference["mtf_features"], gpu_reference["mtf_features"]
        ),
        "mtf_utility_max_abs_error": _max_abs(
            cpu_reference["mtf_utilities"], gpu_reference["mtf_utilities"]
        ),
        "mtf_probability_max_abs_error": _max_abs(
            cpu_reference["mtf_probabilities"], gpu_reference["mtf_probabilities"]
        ),
        "mtf_draws_bit_exact": bool(
            np.array_equal(cpu_reference["mtf_draws"], gpu_reference["mtf_draws"])
        ),
        "gpu_repeat_auto_choice_bit_exact": all(
            np.array_equal(repeated_gpu_choices[0][0], item[0])
            for item in repeated_gpu_choices[1:]
        ),
        "gpu_repeat_mtf_choice_bit_exact": all(
            np.array_equal(repeated_gpu_choices[0][1], item[1])
            for item in repeated_gpu_choices[1:]
        ),
    }
    cpu_median = statistics.median(cpu_times)
    gpu_compute_median = statistics.median(gpu_compute_times)
    gpu_total_median = statistics.median(gpu_total_times)
    telemetry = gpu_reference["telemetry"]
    gates = {
        "independent_cpu_reconstructs_both_checkpoints_exactly": (
            correctness["cpu_auto_checkpoint_mismatches"] == 0
            and correctness["cpu_mtf_checkpoint_mismatches"] == 0
        ),
        "gpu_reconstructs_both_checkpoints_exactly": (
            correctness["gpu_auto_checkpoint_mismatches"] == 0
            and correctness["gpu_mtf_checkpoint_mismatches"] == 0
        ),
        "all_expression_features_bit_exact": (
            correctness["auto_feature_max_abs_error"] == 0.0
            and correctness["mtf_feature_max_abs_error"] == 0.0
        ),
        "activitysim_random_draws_bit_exact": (
            correctness["auto_draws_bit_exact"] and correctness["mtf_draws_bit_exact"]
        ),
        "utilities_within_1e_10": (
            correctness["auto_utility_max_abs_error"] <= 1.0e-10
            and correctness["mtf_utility_max_abs_error"] <= 1.0e-10
        ),
        "probabilities_within_1e_12": (
            correctness["auto_probability_max_abs_error"] <= 1.0e-12
            and correctness["mtf_probability_max_abs_error"] <= 1.0e-12
        ),
        "repeat_choices_bit_exact": (
            correctness["gpu_repeat_auto_choice_bit_exact"]
            and correctness["gpu_repeat_mtf_choice_bit_exact"]
        ),
        "modeled_cpu_fallbacks_zero": telemetry.modeled_cpu_fallbacks == 0,
        "modeled_host_to_device_bytes_zero": telemetry.modeled_host_to_device_bytes == 0,
        "modeled_device_to_host_bytes_zero": telemetry.modeled_device_to_host_bytes == 0,
        "gpu_compute_faster_than_cpu_reference": gpu_compute_median < cpu_median,
        "gpu_total_faster_than_cpu_reference": gpu_total_median < cpu_median,
    }

    input_files = [
        args.pipeline / "households" / "workplace_location.parquet",
        args.pipeline / "households" / "auto_ownership_simulate.parquet",
        args.pipeline / "households" / "cdap_simulate.parquet",
        args.pipeline / "persons" / "cdap_simulate.parquet",
        args.pipeline / "persons" / "mandatory_tour_frequency.parquet",
        CONFIG / "auto_ownership.csv",
        CONFIG / "auto_ownership_coefficients.csv",
        CONFIG / "mandatory_tour_frequency.csv",
        CONFIG / "mandatory_tour_frequency_coefficients.csv",
    ]
    report = {
        "phase": 19,
        "claim_scope": (
            "calibrated public MTC checkpoint replay of auto ownership feeding mandatory "
            "tour frequency; upstream location and CDAP state are frozen checkpoint inputs"
        ),
        "public_workload": {
            "households": len(data["households"]),
            "persons": len(data["persons"]),
            "mandatory_person_choosers": len(cpu_reference["mtf_chooser_id"]),
            "auto_expressions": len(auto_spec.expressions),
            "mtf_expressions": len(mtf_spec.expressions),
            "alternatives_per_component": 5,
            "dependency": "GPU auto-ownership choice is joined by household_id into GPU MTF",
        },
        "boundary": {
            "cpu_allowed": [
                "read public Parquet/CSV",
                "decode CDAP M input flag",
                "resolve configuration coefficients",
                "one-time ingress",
                "kernel launch and scalar validation",
                "final result egress",
            ],
            "gpu_required": [
                "zone lookups",
                "127 published expression evaluations",
                "both utility and probability calculations",
                "ActivitySim-compatible MT19937 draws",
                "both choices",
                "household-to-person dependency join",
            ],
        },
        "machine": machine(),
        "timings_seconds": {
            "cpu_reference_samples": cpu_times,
            "gpu_compute_samples": gpu_compute_times,
            "gpu_total_with_transfer_samples": gpu_total_times,
            "cpu_reference_median": cpu_median,
            "gpu_compute_median": gpu_compute_median,
            "gpu_total_with_transfer_median": gpu_total_median,
        },
        "speedup": {
            "gpu_compute_vs_cpu_reference": cpu_median / gpu_compute_median,
            "gpu_total_with_transfer_vs_cpu_reference": cpu_median / gpu_total_median,
        },
        "correctness": correctness,
        "telemetry": {
            "input_bytes": telemetry.input_bytes,
            "output_bytes_in_capture_run": telemetry.output_bytes,
            "modeled_host_to_device_bytes": telemetry.modeled_host_to_device_bytes,
            "modeled_device_to_host_bytes": telemetry.modeled_device_to_host_bytes,
            "modeled_cpu_fallbacks": telemetry.modeled_cpu_fallbacks,
            "kernel_stages": telemetry.kernel_stages,
        },
        "reproducibility": {
            "repository_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "input_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in input_files
            },
            "benchmark_script_sha256": sha256(Path(__file__).resolve()),
            "calibrated_runtime_sha256": sha256(
                ROOT / "src" / "choiceforge" / "calibrated_chain.py"
            ),
            "gpu_runtime_sha256": sha256(ROOT / "src" / "choiceforge" / "gpu_native.py"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cupy": _cupy().__version__,
            "random_policy": (
                "ActivitySim hash32 channel+step+entity seed; NumPy RandomState MT19937 "
                "first random_sample double, generated bit-exactly on CUDA"
            ),
        },
        "gates": gates,
        "qualified": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not report["qualified"]:
        raise SystemExit("Phase 19 qualification failed")


if __name__ == "__main__":
    main()
