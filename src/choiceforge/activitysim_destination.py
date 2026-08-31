"""Optimized ActivitySim trip-destination helpers.

The public prototype computes two trip-mode-choice logsums for every sampled
destination. ActivitySim normally runs the identical model twice: origin to
sampled stop, then sampled stop to the half-tour destination. This module
stacks those two directions and evaluates the mode-choice model once.
"""

from __future__ import annotations

import logging
import hashlib
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
_STRICT_IR_CACHE = {}
_TRIP_DESTINATION_STAGE_TELEMETRY = []
_TRIP_NATIVE_LOGSUM_TELEMETRY = []
_TRIP_NATIVE_CUBE_CACHE = {}
_TRIP_LOGSUM_CONTRACT_CACHE = {}
_TRIP_SIMULATION_SPEC_CACHE = {}
_PHASE42_CONTRACT_HITS = 0
_PHASE42_CONTRACT_MISSES = 0
_PHASE42_COMPACT_BUNDLES = 0
_PHASE42_SIMULATION_SPEC_HITS = 0
_PHASE42_SIMULATION_SPEC_MISSES = 0
_PHASE43_COMPACT_DRAW_ROWS = 0
_PHASE43_EXPANDED_DRAW_ROWS_AVOIDED = 0
_PHASE43_RNG_CALLS = 0
_PHASE43_CHOICE_DRAW_ROWS = 0
_PHASE43_CHOICE_DRAWS_CONSUMED = 0


@dataclass(frozen=True)
class Phase42DirectionalFrame:
    """Zero-copy paired OD/DP view consumed by the normalized GPU runtime."""

    base: pd.DataFrame
    origin: np.ndarray
    destination: np.ndarray
    phase42_compact_directional: bool = True
    phase43_compact_draws: bool = False

    def __len__(self):
        return int(self.origin.size)

    @property
    def index(self):
        return np.concatenate(
            (self.base.index.to_numpy(copy=False), self.base.index.to_numpy(copy=False))
        )


def reset_trip_destination_stage_telemetry():
    """Clear process-local timings for the trip-number batched path."""
    global _PHASE42_CONTRACT_HITS, _PHASE42_CONTRACT_MISSES, _PHASE42_COMPACT_BUNDLES
    global _PHASE42_SIMULATION_SPEC_HITS, _PHASE42_SIMULATION_SPEC_MISSES
    global _PHASE43_COMPACT_DRAW_ROWS, _PHASE43_EXPANDED_DRAW_ROWS_AVOIDED
    global _PHASE43_RNG_CALLS, _PHASE43_CHOICE_DRAW_ROWS
    global _PHASE43_CHOICE_DRAWS_CONSUMED
    _TRIP_DESTINATION_STAGE_TELEMETRY.clear()
    _TRIP_NATIVE_LOGSUM_TELEMETRY.clear()
    _PHASE42_CONTRACT_HITS = 0
    _PHASE42_CONTRACT_MISSES = 0
    _PHASE42_COMPACT_BUNDLES = 0
    _PHASE42_SIMULATION_SPEC_HITS = 0
    _PHASE42_SIMULATION_SPEC_MISSES = 0
    _PHASE43_COMPACT_DRAW_ROWS = 0
    _PHASE43_EXPANDED_DRAW_ROWS_AVOIDED = 0
    _PHASE43_RNG_CALLS = 0
    _PHASE43_CHOICE_DRAW_ROWS = 0
    _PHASE43_CHOICE_DRAWS_CONSUMED = 0


def clear_trip_native_cube_cache():
    """Release Phase 35's aliases of shared resident CUDA skim cubes."""
    _TRIP_NATIVE_CUBE_CACHE.clear()
    from choiceforge.trip_logsum_native import clear_trip_device_state_cache

    clear_trip_device_state_cache()


def phase42_compiler_telemetry():
    """Report the generalized compiler and compact-boundary proof counters."""
    from choiceforge.arithmetic_abi import (
        PHASE42_NUMERIC_ABI_SHA256,
        PHASE42_NUMERIC_COMPILER_VERSION,
    )
    from choiceforge.native_abi_bootstrap import native_codegen_cache_stats

    return {
        "compiler_version": PHASE42_NUMERIC_COMPILER_VERSION,
        "numeric_abi_sha256": PHASE42_NUMERIC_ABI_SHA256,
        "logsum_contract_cache_entries": len(_TRIP_LOGSUM_CONTRACT_CACHE),
        "logsum_contract_cache_hits": _PHASE42_CONTRACT_HITS,
        "logsum_contract_cache_misses": _PHASE42_CONTRACT_MISSES,
        "compact_directional_bundles": _PHASE42_COMPACT_BUNDLES,
        "simulation_spec_cache_entries": len(_TRIP_SIMULATION_SPEC_CACHE),
        "simulation_spec_cache_hits": _PHASE42_SIMULATION_SPEC_HITS,
        "simulation_spec_cache_misses": _PHASE42_SIMULATION_SPEC_MISSES,
        "native_codegen_cache": native_codegen_cache_stats(),
    }


def phase43_runtime_telemetry():
    """Report compact controlled-random state without exposing model values."""
    return {
        "compact_draw_rows": _PHASE43_COMPACT_DRAW_ROWS,
        "expanded_draw_rows_avoided": _PHASE43_EXPANDED_DRAW_ROWS_AVOIDED,
        "rng_calls": _PHASE43_RNG_CALLS,
        "normal_draws_per_row": 3,
        "choice_draw_rows": _PHASE43_CHOICE_DRAW_ROWS,
        "choice_draws_consumed": _PHASE43_CHOICE_DRAWS_CONSUMED,
    }


def trip_destination_stage_telemetry():
    """Return aggregate timings without exposing the mutable event list."""
    events = list(_TRIP_DESTINATION_STAGE_TELEMETRY)
    totals = {
        "calls": len(events),
        "purposes": int(sum(item["purposes"] for item in events)),
        "trip_rows": int(sum(item["trip_rows"] for item in events)),
        "sample_rows": int(sum(item["sample_rows"] for item in events)),
    }
    for name in ("sampling", "preparation", "preprocessor", "logsums", "simulation", "total"):
        totals[f"{name}_seconds"] = float(
            sum(item[f"{name}_seconds"] for item in events)
        )
    totals["events"] = events
    totals["native_logsum"] = list(_TRIP_NATIVE_LOGSUM_TELEMETRY)
    return totals


def _candidate_sink_metadata(choosers, trace_label, *, required):
    """Read scheduling-only row identity only when a device sink needs it."""
    if not required:
        return {}
    return {
        "chooser_ids": np.asarray(choosers.index, dtype=np.int64),
        "start": np.asarray(choosers["start"], dtype=np.int16),
        "end": np.asarray(choosers["end"], dtype=np.int16),
        "out_period": np.asarray(choosers["out_period"].astype(str)),
        "in_period": np.asarray(choosers["in_period"].astype(str)),
    }


def _cached_strict_ir(spec_frame):
    """Compile an immutable model specification once per process."""
    from choiceforge.sharrow_ir import specification_ir

    started = time.perf_counter()
    row_hash = pd.util.hash_pandas_object(spec_frame, index=True).values.tobytes()
    column_bytes = "\0".join(map(str, spec_frame.columns)).encode("utf-8")
    key = hashlib.sha256(column_bytes + row_hash).hexdigest()
    cache_hit = key in _STRICT_IR_CACHE
    if cache_hit:
        document = _STRICT_IR_CACHE[key]
    else:
        document = specification_ir(spec_frame)
        # The public trip-mode workflow has ten purpose-specific coefficient
        # documents. Keep enough entries for all purposes across trip-number
        # batches instead of evicting them in a ten-item cycle.
        if len(_STRICT_IR_CACHE) >= 32:
            _STRICT_IR_CACHE.pop(next(iter(_STRICT_IR_CACHE)))
        _STRICT_IR_CACHE[key] = document
    return document, cache_hit, (time.perf_counter() - started) * 1000


class DestinationBatchUnsupported(RuntimeError):
    """Raised only by preflight checks before any random draws are consumed."""


def _purpose_invariant_preprocessor(spec, coefficient_sets) -> bool:
    """Return whether one locals dictionary is valid for every purpose."""
    if len(coefficient_sets) < 2:
        return True
    keys = set().union(*(values.keys() for values in coefficient_sets))
    varying = {
        key
        for key in keys
        if len({repr(values.get(key)) for values in coefficient_sets}) > 1
    }
    if not varying:
        return True
    expressions = "\n".join(str(value) for value in spec["expression"])
    return not any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(key))}(?![A-Za-z0-9_])", expressions)
        for key in varying
    )


def _single_preprocessor_settings(logsum_settings):
    if isinstance(logsum_settings, dict):
        preprocessor = logsum_settings.get("preprocessor")
    else:
        preprocessor = getattr(logsum_settings, "preprocessor", None)
    if preprocessor is None:
        return None
    if isinstance(preprocessor, list):
        if len(preprocessor) != 1:
            return None
        preprocessor = preprocessor[0]
    if hasattr(preprocessor, "dict"):
        preprocessor = preprocessor.dict()
    if not isinstance(preprocessor, dict) or not preprocessor.get("SPEC"):
        return None
    return preprocessor


def _controlled_random_draw_count(state, logsum_settings):
    """Return the configured broadcast-draw count for the reviewed preprocessor."""
    from activitysim.core import assign

    preprocessor = _single_preprocessor_settings(logsum_settings)
    if preprocessor is None:
        raise DestinationBatchUnsupported("trip logsum preprocessor is not singular")
    spec_name = preprocessor["SPEC"]
    if not spec_name.endswith(".csv"):
        spec_name += ".csv"
    spec = assign.read_assignment_spec(
        state.filesystem.get_config_file_path(spec_name)
    )
    marker = "rng.lognormal_for_df(df,"
    return sum(marker in str(expression) for expression in spec["expression"])


def _combined_preprocessor(
    state,
    frames,
    combined,
    locals_dict,
    skims,
    logsum_settings,
    trace_label,
    raw_capture=None,
):
    """Annotate stacked directions while preserving ActivitySim RNG draws.

    ActivitySim rewrites all ``rng.lognormal_for_df`` expressions and obtains
    their normal variates in one keyed draw.  We reproduce the two original
    draws (OD first, DP second), then expose the stacked variates to a single
    evaluation of the assignment spec.  This preserves both values and random
    channel advancement while avoiding a second pass through the deterministic
    expressions and skim lookups.
    """
    from activitysim.core import assign, expressions, simulate

    preprocessor = _single_preprocessor_settings(logsum_settings)
    if preprocessor is None:
        return False

    spec_name = preprocessor["SPEC"]
    if not spec_name.endswith(".csv"):
        spec_name += ".csv"
    spec = assign.read_assignment_spec(
        state.filesystem.get_config_file_path(spec_name)
    )
    marker = "rng.lognormal_for_df(df,"
    random_rows = [i for i in spec.index if marker in spec.loc[i, "expression"]]
    n_randoms = len(random_rows)

    combined_locals = locals_dict.copy()
    combined_locals.update(skims)
    for table_name in preprocessor.get("TABLES") or []:
        combined_locals[table_name] = state.get_dataframe(table_name)

    if n_randoms:
        rng = state.get_rn_generator()
        directional_draws = []
        for frame in frames:
            draws = rng.normal_for_df(frame, broadcast=True, size=n_randoms)
            directional_draws.append(pd.DataFrame(draws, index=frame.index))
        combined_locals["random_draws"] = pd.concat(directional_draws, axis=0)
        if raw_capture is not None:
            raw_capture["random_draws"] = np.asarray(
                combined_locals["random_draws"], dtype=np.float64
            )

        def rng_lognormal(draws, mu, sigma, broadcast=True, scale=False):
            if scale:
                x = 1 + ((sigma * sigma) / (mu * mu))
                mu = np.log(mu / np.sqrt(x))
                sigma = np.sqrt(np.log(x))
            if not broadcast:
                raise ValueError("combined preprocessing requires broadcast draws")
            return np.exp(draws * sigma + mu)

        combined_locals["rng_lognormal"] = rng_lognormal
        for random_number, row in enumerate(random_rows):
            spec.loc[row, "expression"] = spec.loc[row, "expression"].replace(
                marker, f"rng_lognormal(random_draws[{random_number}],"
            )

    simulate.set_skim_wrapper_targets(combined, skims)
    results, _, _ = assign.assign_variables(
        state,
        spec,
        combined,
        combined_locals,
        df_alias=preprocessor.get("DF") or "df",
        trace_label=trace_label,
    )
    expressions.assign_in_place(
        combined,
        results,
        state.settings.downcast_int,
        state.settings.downcast_float,
    )
    return True


def _native_trip_logsum_values(state, bundle, combined_skims, draws):
    """Execute the reviewed raw-trip ABI and return host nested logsums."""
    from choiceforge.cuda_skims import cuda_cube_from_activitysim
    from choiceforge.native_abi_bootstrap import NativeSkimCube, compile_native_strict_abi
    from choiceforge.nested_logit import mtc21_nested_logsums_cuda
    from choiceforge.trip_logsum_native import TripLogsumNativePlan

    phase42_compiler = (
        os.environ.get("CHOICEFORGE_PHASE42_NUMERIC_COMPILER", "0") == "1"
    )
    if phase42_compiler:
        document = bundle["logsum_document"]
        if document is None:
            raise ValueError("Phase 42 bundle is missing its compiled numeric IR")
        ir_cache_hit = bool(bundle["logsum_document_cache_hit"])
        ir_compile_ms = 0.0
    else:
        from choiceforge.sharrow_ir import specification_ir

        ir_started = time.perf_counter()
        document = specification_ir(bundle["logsum_spec"].reset_index())
        ir_cache_hit = False
        ir_compile_ms = (time.perf_counter() - ir_started) * 1000.0
    scalar_environment = state.get_global_constants().copy()
    scalar_environment.update(bundle["locals"])

    def cube_loader(source):
        _, direction, key = source
        wrapper_name = "od_skims" if direction == "od_skims_reverse" else direction
        if wrapper_name not in combined_skims:
            raise ValueError(f"trip native skim direction {direction!r} is absent")
        wrapper = combined_skims[wrapper_name]
        cache_key = (id(getattr(wrapper, "dataset", wrapper)), wrapper_name, key)
        cached = _TRIP_NATIVE_CUBE_CACHE.get(cache_key)
        if cached is None:
            cached = cuda_cube_from_activitysim(wrapper, key)
            _TRIP_NATIVE_CUBE_CACHE[cache_key] = cached
        data, dest_count, time_count, rank = cached
        return NativeSkimCube(data, dest_count, time_count, rank)

    phase36_device = (
        os.environ.get("CHOICEFORGE_PHASE36_DEVICE_TRIP_ABI", "0") == "1"
    )
    phase36_shadow = (
        os.environ.get("CHOICEFORGE_PHASE36_DEVICE_TRIP_ABI_SHADOW", "0") == "1"
    )
    phase37_fused = (
        os.environ.get("CHOICEFORGE_PHASE37_FUSED_TRIP_UTILITY", "0") == "1"
    )
    phase37_shadow = (
        os.environ.get("CHOICEFORGE_PHASE37_FUSED_TRIP_UTILITY_SHADOW", "0") == "1"
    )
    phase38_normalized = (
        os.environ.get("CHOICEFORGE_PHASE38_NORMALIZED_TRIP_STATE", "0") == "1"
    )
    phase38_shadow = (
        os.environ.get("CHOICEFORGE_PHASE38_NORMALIZED_TRIP_STATE_SHADOW", "0") == "1"
    )
    if phase37_fused and not phase36_device:
        raise ValueError("Phase 37 fused trip utility requires the Phase 36 raw contract")
    if phase38_normalized and not phase37_fused:
        raise ValueError("Phase 38 normalized trip state requires Phase 37 fusion")
    native = compile_native_strict_abi(
        document,
        scalar_environment,
        cube_loader,
        rows=len(bundle["combined"]),
        minimal_row_state=bool(phase37_fused and not phase37_shadow),
        cache_codegen=phase42_compiler,
    )
    plan = TripLogsumNativePlan(
        native.invocation, document=document, bindings=native.bindings
    )
    populate_arguments = (
        bundle["combined"], state.get_dataframe("land_use"),
        state.get_dataframe("tours"), bundle["locals"], draws,
    )
    shadow_metrics = {}
    if phase38_normalized and phase38_shadow:
        cp = plan.cp
        reference_utilities, _ = plan.populate_fused(*populate_arguments)
        reference_copy = cp.array(reference_utilities, copy=True)
        utilities, telemetry = plan.populate_normalized(*populate_arguments)
        both_nan = cp.isnan(reference_copy) & cp.isnan(utilities)
        equal = (reference_copy == utilities) | both_nan
        difference = cp.where(equal, 0.0, cp.abs(reference_copy - utilities))
        mismatches = int(cp.count_nonzero(~equal).get())
        max_abs = float(cp.max(difference).get()) if difference.size else 0.0
        shadow_metrics = {
            "phase38_shadow_utility_mismatches": mismatches,
            "phase38_shadow_utility_max_abs_difference": max_abs,
        }
        if not np.isfinite(max_abs) or max_abs > 1.0e-5:
            raise AssertionError(
                "Phase 38 normalized utility shadow mismatch "
                f"count={mismatches} max_abs={max_abs:.3e}"
            )
    elif phase38_normalized:
        utilities, telemetry = plan.populate_normalized(*populate_arguments)
    elif phase37_fused and phase37_shadow:
        cp = plan.cp
        reference_utilities, _ = plan.populate_device(*populate_arguments)
        reference_copy = cp.array(reference_utilities, copy=True)
        utilities, telemetry = plan.populate_fused(*populate_arguments)
        both_nan = cp.isnan(reference_copy) & cp.isnan(utilities)
        equal = (reference_copy == utilities) | both_nan
        difference = cp.where(equal, 0.0, cp.abs(reference_copy - utilities))
        mismatches = int(cp.count_nonzero(~equal).get())
        max_abs = float(cp.max(difference).get()) if difference.size else 0.0
        shadow_metrics = {
            "phase37_shadow_utility_mismatches": mismatches,
            "phase37_shadow_utility_max_abs_difference": max_abs,
        }
        if not np.isfinite(max_abs) or max_abs > 1.0e-5:
            raise AssertionError(
                "Phase 37 fused utility shadow mismatch "
                f"count={mismatches} max_abs={max_abs:.3e}"
            )
    elif phase37_fused:
        utilities, telemetry = plan.populate_fused(*populate_arguments)
    elif phase36_device and phase36_shadow:
        cp = plan.cp
        reference_utilities, _ = plan.populate(*populate_arguments)
        reference_copy = cp.array(reference_utilities, copy=True)
        utilities, telemetry = plan.populate_device(*populate_arguments)
        both_nan = cp.isnan(reference_copy) & cp.isnan(utilities)
        equal = (reference_copy == utilities) | both_nan
        difference = cp.where(equal, 0.0, cp.abs(reference_copy - utilities))
        mismatches = int(cp.count_nonzero(~equal).get())
        max_abs = float(cp.max(difference).get()) if difference.size else 0.0
        shadow_metrics = {
            "abi_shadow_utility_mismatches": mismatches,
            "abi_shadow_utility_max_abs_difference": max_abs,
        }
        if max_abs > 1.0e-5:
            raise AssertionError(
                "Phase 36 device ABI utility shadow mismatch "
                f"count={mismatches} max_abs={max_abs:.3e}"
            )
    elif phase36_device:
        utilities, telemetry = plan.populate_device(*populate_arguments)
    else:
        utilities, telemetry = plan.populate(*populate_arguments)
    logsums, nested = mtc21_nested_logsums_cuda(
        utilities,
        bundle["nest_spec"],
        tuple(document["alternatives"]),
        return_telemetry=True,
        numeric_policy="activitysim_pandas_float64",
    )
    _TRIP_NATIVE_LOGSUM_TELEMETRY.append(
        {
            "trace_label": str(bundle["trace_label"]),
            "rows": telemetry.rows,
            "compact_host_bytes": telemetry.compact_host_bytes,
            "host_build_seconds": telemetry.host_build_seconds,
            "upload_seconds": telemetry.upload_seconds,
            "availability_kernel_seconds": telemetry.availability_kernel_seconds,
            "utility_kernel_seconds": telemetry.utility_kernel_seconds,
            "nested_kernel_seconds": nested.kernel_ms / 1000.0,
            "dense_preprocessor_rows_read": 0,
            "fallback_calls": 0,
            "backend": telemetry.backend,
            "device_preparation_kernel_seconds": (
                telemetry.device_preparation_kernel_seconds
            ),
            "compact_device_input_bytes": telemetry.compact_device_input_bytes,
            "dense_host_abi_bytes_avoided": telemetry.dense_host_abi_bytes_avoided,
            "resident_land_bytes": telemetry.resident_land_bytes,
            "dense_device_abi_bytes_eliminated": (
                telemetry.dense_device_abi_bytes_eliminated
            ),
            "coordinate_device_bytes_eliminated": (
                telemetry.coordinate_device_bytes_eliminated
            ),
            "fused_kernel_seconds": telemetry.fused_kernel_seconds,
            "minimal_bootstrap_bytes": telemetry.minimal_bootstrap_bytes,
            "normalized_trip_rows": telemetry.normalized_trip_rows,
            "normalized_state_rows": telemetry.normalized_state_rows,
            "normalized_row_bytes": telemetry.normalized_row_bytes,
            "normalized_state_bytes": telemetry.normalized_state_bytes,
            "phase37_compact_bytes_eliminated": (
                telemetry.phase37_compact_bytes_eliminated
            ),
            "resident_workspace_hits": telemetry.resident_workspace_hits,
            "resident_workspace_arrays": telemetry.resident_workspace_arrays,
            "normalized_contract_valid": telemetry.normalized_contract_valid,
            "strict_ir_cache_hit": ir_cache_hit,
            "strict_ir_compile_ms": ir_compile_ms,
            "native_codegen_cache_hit": native.manifest.get(
                "codegen_cache_hit", False
            ),
            "native_codegen_cache_key": native.manifest.get("codegen_cache_key"),
            **shadow_metrics,
        }
    )
    return np.asarray(logsums)


def compute_logsums_combined(
    state,
    primary_purpose,
    trips: pd.DataFrame,
    destination_sample,
    tours_merged: pd.DataFrame,
    model_settings,
    skim_hotel,
    trace_label: str,
    *,
    fallback=None,
):
    """Compute both directional logsums in one ActivitySim/Sharrow call.

    Unsupported three-zone path-builder models retain ActivitySim's original
    implementation because their wrappers carry additional directional state.
    """
    from activitysim.abm.models import trip_destination as td
    from activitysim.core import chunk, config, expressions, los, simulate, tracing

    network_los = state.get_injectable("network_los")
    if network_los.zone_system == los.THREE_ZONE:
        if fallback is None:
            raise NotImplementedError("combined three-zone trip-destination logsums")
        return fallback(
            state,
            primary_purpose,
            trips,
            destination_sample,
            tours_merged,
            model_settings,
            skim_hotel,
            trace_label,
        )

    started = time.perf_counter()
    trace_label = tracing.extend_trace_label(trace_label, "compute_logsums_combined")
    trips_merged = pd.merge(
        trips, tours_merged, left_on="tour_id", right_index=True, how="left"
    )
    if not trips_merged.index.equals(trips.index):
        raise AssertionError("trip merge changed chooser order")
    choosers = pd.merge(
        destination_sample,
        trips_merged.reset_index(),
        left_index=True,
        right_on="trip_id",
        how="left",
        suffixes=("", "_r"),
    ).set_index("trip_id")
    if not choosers.index.equals(destination_sample.index):
        raise AssertionError("destination merge changed alternative order")

    origin_column = "_choiceforge_origin"
    destination_column = "_choiceforge_destination"
    logsum_settings = state.filesystem.read_model_settings(model_settings.LOGSUM_SETTINGS)
    coefficients = state.filesystem.get_segment_coefficients(logsum_settings, primary_purpose)
    nest_spec = config.get_logit_model_settings(logsum_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)
    logsum_spec = state.filesystem.read_model_spec(file_name=logsum_settings["SPEC"])
    logsum_spec = simulate.eval_coefficients(
        state, logsum_spec, coefficients, estimator=None
    )
    locals_dict = dict(config.get_model_constants(logsum_settings))
    locals_dict.update(coefficients)

    skim_dict = network_los.get_default_skim_dict()
    original_skims = skim_hotel.logsum_skims()
    od_skims = {
        "ORIGIN": model_settings.TRIP_ORIGIN,
        "DESTINATION": model_settings.ALT_DEST_COL_NAME,
        "odt_skims": original_skims["odt_skims"],
        "dot_skims": original_skims["dot_skims"],
        "od_skims": original_skims["od_skims"],
        "timeframe": "trip",
    }
    dp_skims = {
        "ORIGIN": model_settings.ALT_DEST_COL_NAME,
        "DESTINATION": model_settings.PRIMARY_DEST,
        "odt_skims": original_skims["dpt_skims"],
        "dot_skims": original_skims["pdt_skims"],
        "od_skims": original_skims["dp_skims"],
        "timeframe": "trip",
    }

    od = choosers.copy()
    dp = choosers.copy()
    od[origin_column] = od[model_settings.TRIP_ORIGIN].to_numpy()
    od[destination_column] = od[model_settings.ALT_DEST_COL_NAME].to_numpy()
    dp[origin_column] = dp[model_settings.ALT_DEST_COL_NAME].to_numpy()
    dp[destination_column] = dp[model_settings.PRIMARY_DEST].to_numpy()
    combined = pd.concat((od, dp), axis=0)

    combined_skims = {
        "ORIGIN": origin_column,
        "DESTINATION": destination_column,
        "odt_skims": skim_dict.wrap_3d(
            orig_key=origin_column,
            dest_key=destination_column,
            dim3_key="trip_period",
        ),
        "dot_skims": skim_dict.wrap_3d(
            orig_key=destination_column,
            dest_key=origin_column,
            dim3_key="trip_period",
        ),
        "od_skims": skim_dict.wrap(origin_column, destination_column),
        "timeframe": "trip",
    }
    combined_trace = tracing.extend_trace_label(trace_label, "combined")
    with chunk.chunk_log(
        state,
        tracing.extend_trace_label(combined_trace, "annotate_preprocessor"),
        base=True,
    ):
        used_combined_preprocessor = _combined_preprocessor(
            state,
            (od, dp),
            combined,
            locals_dict,
            combined_skims,
            logsum_settings,
            combined_trace,
        )

    # Conservative fallback for uncommon models with multiple preprocessors or
    # nonstandard settings. It retains the original keyed-RNG call sequence.
    if not used_combined_preprocessor:
        directional_frames = []
        for direction, frame, direction_skims in (
            ("od", od, od_skims),
            ("dp", dp, dp_skims),
        ):
            direction_locals = locals_dict.copy()
            direction_locals.update(direction_skims)
            direction_trace = tracing.extend_trace_label(trace_label, direction)
            expressions.annotate_preprocessors(
                state,
                frame,
                direction_locals,
                direction_skims,
                logsum_settings,
                direction_trace,
            )
            directional_frames.append(frame)
        combined = pd.concat(directional_frames, axis=0)

    combined_locals = locals_dict.copy()
    combined_locals.update(combined_skims)
    logsums = simulate.simple_simulate_logsums(
        state,
        combined,
        logsum_spec,
        nest_spec,
        skims=combined_skims,
        locals_d=combined_locals,
        chunk_size=state.settings.chunk_size,
        trace_label=trace_label,
        chunk_tag="trip_destination.compute_logsums_combined",
        explicit_chunk_size=model_settings.explicit_chunk,
    )
    count = len(destination_sample)
    destination_sample["od_logsum"] = np.asarray(logsums.iloc[:count])
    destination_sample["dp_logsum"] = np.asarray(logsums.iloc[count:])
    logger.info(
        "%s ChoiceForge combined directional logsums rows=%d total=%.3fms",
        trace_label,
        len(combined),
        (time.perf_counter() - started) * 1000,
    )
    return destination_sample


def _trip_logsum_contract(state, primary_purpose, model_settings, trace_label):
    """Cache immutable purpose-specific model compilation inputs.

    ActivitySim reads and resolves the same ten specifications once for each
    trip number.  Phase 42 makes the already-hashed model contract the cache
    key and reuses it; chooser rows and random draws are never cached.
    """
    from activitysim.core import config, simulate, tracing

    global _PHASE42_CONTRACT_HITS, _PHASE42_CONTRACT_MISSES
    key = (
        id(state.filesystem),
        str(model_settings.LOGSUM_SETTINGS),
        str(primary_purpose),
    )
    cached = _TRIP_LOGSUM_CONTRACT_CACHE.get(key)
    if cached is not None:
        _PHASE42_CONTRACT_HITS += 1
        return cached, True
    from choiceforge.sharrow_ir import specification_ir

    logsum_settings = state.filesystem.read_model_settings(
        model_settings.LOGSUM_SETTINGS
    )
    coefficients = state.filesystem.get_segment_coefficients(
        logsum_settings, primary_purpose
    )
    nest_spec = config.get_logit_model_settings(logsum_settings)
    nest_spec = simulate.eval_nest_coefficients(
        nest_spec, coefficients, trace_label
    )
    logsum_spec = state.filesystem.read_model_spec(file_name=logsum_settings["SPEC"])
    logsum_spec = simulate.eval_coefficients(
        state, logsum_spec, coefficients, estimator=None
    )
    locals_dict = dict(config.get_model_constants(logsum_settings))
    locals_dict.update(coefficients)
    document = specification_ir(logsum_spec.reset_index())
    cached = {
        "logsum_settings": logsum_settings,
        "coefficients": coefficients,
        "nest_spec": nest_spec,
        "logsum_spec": logsum_spec,
        "locals": locals_dict,
        "document": document,
    }
    _TRIP_LOGSUM_CONTRACT_CACHE[key] = cached
    _PHASE42_CONTRACT_MISSES += 1
    return cached, False


def _phase42_simulation_spec(original, *args, **kwargs):
    """Cache only the immutable trip-destination final-choice specification."""
    global _PHASE42_SIMULATION_SPEC_HITS, _PHASE42_SIMULATION_SPEC_MISSES
    if kwargs.get("spec_id") != "SPEC" or kwargs.get("segment_name") is None:
        return original(*args, **kwargs)
    state = args[0] if args else kwargs.get("state")
    key = (
        id(getattr(state, "filesystem", state)),
        str(kwargs.get("spec_file_name")),
        str(kwargs.get("coefficients_file_name")),
        str(kwargs.get("segment_name")),
    )
    cached = _TRIP_SIMULATION_SPEC_CACHE.get(key)
    if cached is not None:
        _PHASE42_SIMULATION_SPEC_HITS += 1
        return cached
    value = original(*args, **kwargs)
    _TRIP_SIMULATION_SPEC_CACHE[key] = value
    _PHASE42_SIMULATION_SPEC_MISSES += 1
    return value


def _phase43_compact_draws_for_bundles(
    state, bundles, draw_count, *, include_choice_draws=False
):
    """Generate ActivitySim's OD/DP draws on unique trip rows only.

    ``normal_for_df(..., broadcast=True)`` internally deduplicates the input
    index, advances each trip channel by ``draw_count``, and then expands the
    values back to every sampled alternative. The normalized native runtime
    needs only the pre-expansion state, so Phase 43 supplies that index directly.
    """
    global _PHASE43_COMPACT_DRAW_ROWS, _PHASE43_EXPANDED_DRAW_ROWS_AVOIDED
    global _PHASE43_RNG_CALLS, _PHASE43_CHOICE_DRAW_ROWS
    indexes = []
    expanded_rows = 0
    for bundle in bundles:
        sample_ids = bundle["destination_sample"].index.to_numpy(copy=False)
        if sample_ids.size == 0:
            raise DestinationBatchUnsupported("Phase 43 compact draws require sampled rows")
        starts = np.flatnonzero(np.r_[True, sample_ids[1:] != sample_ids[:-1]])
        trip_index = bundle["trips"].index
        if not np.array_equal(sample_ids[starts], trip_index.to_numpy(copy=False)):
            raise DestinationBatchUnsupported(
                "Phase 43 sampled alternatives are not contiguous in chooser order"
            )
        indexes.append(trip_index)
        expanded_rows += 2 * len(bundle["destination_sample"])
    index_name = indexes[0].name
    if any(index.name != index_name for index in indexes):
        raise DestinationBatchUnsupported("Phase 43 chooser index channels differ")
    unique_index = pd.Index(
        np.concatenate([index.to_numpy(copy=False) for index in indexes]),
        name=index_name,
    )
    if unique_index.has_duplicates:
        raise DestinationBatchUnsupported("Phase 43 purpose trip rows overlap")
    unique_frame = pd.DataFrame(index=unique_index)
    rng = state.get_rn_generator()
    directional = [
        np.asarray(
            rng.normal_for_df(unique_frame, broadcast=False, size=draw_count),
            dtype=np.float64,
        )
        for _direction in range(2)
    ]
    if any(values.shape != (len(unique_frame), draw_count) for values in directional):
        raise DestinationBatchUnsupported("Phase 43 RNG returned an unexpected shape")
    choice_draws = None
    if include_choice_draws:
        choice_draws = np.asarray(
            rng.random_for_df(unique_frame), dtype=np.float64
        ).reshape(-1)
        if choice_draws.shape != (len(unique_frame),):
            raise DestinationBatchUnsupported(
                "Phase 43 choice RNG returned an unexpected shape"
            )
        _PHASE43_CHOICE_DRAW_ROWS += len(unique_frame)
        _PHASE43_RNG_CALLS += 1
    results = []
    cursor = 0
    for index in indexes:
        stop = cursor + len(index)
        results.append(np.concatenate((directional[0][cursor:stop], directional[1][cursor:stop])))
        if choice_draws is not None:
            bundles[len(results) - 1]["compact_choice_draws"] = choice_draws[
                cursor:stop
            ]
        cursor = stop
    compact_rows = 2 * len(unique_frame)
    _PHASE43_COMPACT_DRAW_ROWS += compact_rows
    _PHASE43_EXPANDED_DRAW_ROWS_AVOIDED += int(expanded_rows - compact_rows)
    _PHASE43_RNG_CALLS += 2
    return results


def _phase43_compact_directional_draws(state, bundle, draw_count):
    """Single-bundle adapter retained for focused contract tests."""
    return _phase43_compact_draws_for_bundles(
        state, [bundle], draw_count
    )[0]


def _prepare_logsum_bundle(
    state,
    primary_purpose,
    trips,
    destination_sample,
    tours_merged,
    model_settings,
    skim_hotel,
    trace_label,
):
    """Lower one purpose segment to directional chooser frames."""
    from activitysim.core import config, simulate, tracing

    global _PHASE42_COMPACT_BUNDLES
    phase42_compact = (
        os.environ.get("CHOICEFORGE_PHASE42_NUMERIC_COMPILER", "0") == "1"
        and os.environ.get(
            "CHOICEFORGE_PHASE35_NATIVE_TRIP_LOGSUM_PRODUCTION", "0"
        ) == "1"
    )
    phase43_compact_draws = (
        os.environ.get("CHOICEFORGE_PHASE43_COMPACT_TRIP_STATE", "0") == "1"
    )

    trips_merged = pd.merge(
        trips, tours_merged, left_on="tour_id", right_index=True, how="left"
    )
    if not trips_merged.index.equals(trips.index):
        raise AssertionError("trip merge changed chooser order")
    if phase42_compact:
        choosers = destination_sample.join(
            trips_merged, how="left", rsuffix="_r", sort=False
        )
    else:
        choosers = pd.merge(
            destination_sample,
            trips_merged.reset_index(),
            left_index=True,
            right_on="trip_id",
            how="left",
            suffixes=("", "_r"),
        ).set_index("trip_id")
    if not choosers.index.equals(destination_sample.index):
        raise AssertionError("destination merge changed alternative order")

    bundle_trace = tracing.extend_trace_label(
        trace_label, "compute_logsums_tripnum_batched"
    )
    if phase42_compact:
        contract, contract_cache_hit = _trip_logsum_contract(
            state, primary_purpose, model_settings, bundle_trace
        )
        logsum_settings = contract["logsum_settings"]
        coefficients = contract["coefficients"]
        nest_spec = contract["nest_spec"]
        logsum_spec = contract["logsum_spec"]
        locals_dict = contract["locals"]
        logsum_document = contract["document"]
    else:
        logsum_settings = state.filesystem.read_model_settings(
            model_settings.LOGSUM_SETTINGS
        )
        coefficients = state.filesystem.get_segment_coefficients(
            logsum_settings, primary_purpose
        )
        nest_spec = config.get_logit_model_settings(logsum_settings)
        nest_spec = simulate.eval_nest_coefficients(
            nest_spec, coefficients, bundle_trace
        )
        logsum_spec = state.filesystem.read_model_spec(file_name=logsum_settings["SPEC"])
        logsum_spec = simulate.eval_coefficients(
            state, logsum_spec, coefficients, estimator=None
        )
        locals_dict = dict(config.get_model_constants(logsum_settings))
        locals_dict.update(coefficients)
        logsum_document = None
        contract_cache_hit = False

    origin_column = "_choiceforge_origin"
    destination_column = "_choiceforge_destination"
    if phase42_compact:
        combined_origin = np.concatenate(
            (
                choosers[model_settings.TRIP_ORIGIN].to_numpy(copy=False),
                choosers[model_settings.ALT_DEST_COL_NAME].to_numpy(copy=False),
            )
        )
        combined_destination = np.concatenate(
            (
                choosers[model_settings.ALT_DEST_COL_NAME].to_numpy(copy=False),
                choosers[model_settings.PRIMARY_DEST].to_numpy(copy=False),
            )
        )
        combined = Phase42DirectionalFrame(
            base=choosers,
            origin=combined_origin,
            destination=combined_destination,
            phase43_compact_draws=phase43_compact_draws,
        )
        # ActivitySim's keyed RNG reads canonical row identity, not unrelated
        # chooser columns.  A narrow frame preserves the exact ledger while
        # avoiding two more copies of every sampled-alternative column.
        random_frame = pd.DataFrame(index=choosers.index.copy())
        directional_frames = (random_frame, random_frame)
        od = dp = None
        _PHASE42_COMPACT_BUNDLES += 1
    else:
        od = choosers.copy()
        dp = choosers.copy()
        od[origin_column] = od[model_settings.TRIP_ORIGIN].to_numpy()
        od[destination_column] = od[model_settings.ALT_DEST_COL_NAME].to_numpy()
        dp[origin_column] = dp[model_settings.ALT_DEST_COL_NAME].to_numpy()
        dp[destination_column] = dp[model_settings.PRIMARY_DEST].to_numpy()
        combined = pd.concat((od, dp), axis=0)
        directional_frames = (od, dp)
    return {
        "purpose": primary_purpose,
        "trips": trips,
        "destination_sample": destination_sample,
        "od": od,
        "dp": dp,
        "combined": combined,
        "directional_frames": directional_frames,
        "logsum_settings": logsum_settings,
        "coefficients": coefficients,
        "nest_spec": nest_spec,
        "logsum_spec": logsum_spec,
        "locals": locals_dict,
        "logsum_document": logsum_document,
        "logsum_document_cache_hit": contract_cache_hit,
        "trace_label": bundle_trace,
    }


def _generic_logsum_skims(state):
    skim_dict = state.get_injectable("network_los").get_default_skim_dict()
    origin = "_choiceforge_origin"
    destination = "_choiceforge_destination"
    return {
        "ORIGIN": origin,
        "DESTINATION": destination,
        "odt_skims": skim_dict.wrap_3d(
            orig_key=origin, dest_key=destination, dim3_key="trip_period"
        ),
        "dot_skims": skim_dict.wrap_3d(
            orig_key=destination, dest_key=origin, dim3_key="trip_period"
        ),
        "od_skims": skim_dict.wrap(origin, destination),
        "timeframe": "trip",
    }


def _evaluate_logsum_bundle(state, bundle, combined_skims, model_settings):
    from activitysim.core import simulate

    locals_dict = bundle["locals"].copy()
    locals_dict.update(combined_skims)
    nested_backend = getattr(model_settings, "DESTINATION_NESTED_LOGIT_BACKEND", None)
    if nested_backend == "choiceforge_cuda_mtc21":
        logsums = _simple_simulate_mtc21_logsums_cuda(
            state,
            bundle["combined"],
            bundle["logsum_spec"],
            bundle["nest_spec"],
            combined_skims,
            locals_dict,
            bundle["trace_label"],
            model_settings.explicit_chunk,
        )
    else:
        logsums = simulate.simple_simulate_logsums(
            state,
            bundle["combined"],
            bundle["logsum_spec"],
            bundle["nest_spec"],
            skims=combined_skims,
            locals_d=locals_dict,
            chunk_size=state.settings.chunk_size,
            trace_label=bundle["trace_label"],
            chunk_tag="trip_destination.compute_logsums_tripnum_batched",
            explicit_chunk_size=model_settings.explicit_chunk,
        )
    count = len(bundle["destination_sample"])
    bundle["destination_sample"]["od_logsum"] = np.asarray(logsums.iloc[:count])
    bundle["destination_sample"]["dp_logsum"] = np.asarray(logsums.iloc[count:])


def _simple_simulate_mtc21_logsums_cuda(
    state,
    choosers,
    spec,
    nest_spec,
    skims,
    locals_dict,
    trace_label,
    explicit_chunk_size,
    *,
    device_logsum_sink=None,
    resident_invocation_sink=None,
    materialize_device_sink_result=False,
):
    """Run generated CUDA utility and nesting with an optional device sink.

    The default preserves the ActivitySim dataframe return contract.  When a
    sink is supplied, the modeled logsum vector is delivered on-device and the
    dataframe path receives a neutral placeholder; a downstream GPU scheduler
    must then become authoritative for the choice.
    """
    from activitysim.core import simulate
    from choiceforge.nested_logit import mtc21_nested_logsums_cuda

    original_reducer = simulate.compute_nested_exp_utilities
    shadow_remaining = int(os.environ.get("CHOICEFORGE_UTILITY_SHADOW_BATCHES", "0"))
    strict_remaining = int(os.environ.get("CHOICEFORGE_STRICT_CPU_BATCHES", "0"))
    strict_report_dir = os.environ.get("CHOICEFORGE_STRICT_CPU_REPORT_DIR")
    strict_require_exact = os.environ.get("CHOICEFORGE_STRICT_CPU_REQUIRE_EXACT", "0") == "1"
    strict_report_sequence = 0
    strict_cuda_remaining = int(os.environ.get("CHOICEFORGE_STRICT_CUDA_BATCHES", "0"))
    strict_cuda_report_dir = os.environ.get("CHOICEFORGE_STRICT_CUDA_REPORT_DIR")
    strict_cuda_report_sequence = 0
    strict_cuda_candidate = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_CANDIDATE", "0") == "1"
    )
    strict_cuda_candidate_max_rows = int(
        os.environ.get("CHOICEFORGE_STRICT_CUDA_MAX_ROWS", "100000")
    )
    strict_cuda_tile_rows = int(
        os.environ.get("CHOICEFORGE_STRICT_CUDA_TILE_ROWS", "1")
    )
    strict_cuda_locality = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_LOCALITY", "0") == "1"
        or strict_cuda_tile_rows > 1
    )
    strict_cuda_cooperative = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_COOPERATIVE_SKIMS", "0") == "1"
        or strict_cuda_tile_rows > 1
    )
    strict_cuda_compact_inputs = (
        os.environ.get(
            "CHOICEFORGE_STRICT_CUDA_COMPACT_INPUTS",
            "1" if strict_cuda_locality else "0",
        ) == "1"
    )
    strict_cuda_grouped_indices = (
        os.environ.get(
            "CHOICEFORGE_STRICT_CUDA_GROUPED_INDICES",
            "1" if strict_cuda_locality else "0",
        ) == "1"
    )
    strict_cuda_sparse_coefficients = (
        os.environ.get(
            "CHOICEFORGE_STRICT_CUDA_SPARSE_COEFFICIENTS",
            "0",
        ) == "1"
    )
    strict_cuda_expression_float32 = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_EXPRESSION_FLOAT32", "0") == "1"
    )
    strict_cuda_persistent_plan = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_PERSISTENT_PLAN", "0") == "1"
    )
    strict_cuda_reuse_buffers = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS", "0") == "1"
    )
    strict_cuda_sharrow_fma = (
        os.environ.get("CHOICEFORGE_STRICT_CUDA_SHARROW_FMA", "0") == "1"
    )
    candidate_phase = (
        17 if strict_cuda_persistent_plan else (16 if strict_cuda_locality else 15)
    )
    phase15_report_dir = os.environ.get("CHOICEFORGE_PHASE15_REPORT_DIR")
    phase15_run_id = os.environ.get("CHOICEFORGE_PHASE15_RUN_ID", "")
    phase16_report_dir = os.environ.get("CHOICEFORGE_PHASE16_REPORT_DIR")
    phase16_run_id = os.environ.get("CHOICEFORGE_PHASE16_RUN_ID", "")
    phase17_report_dir = os.environ.get("CHOICEFORGE_PHASE17_REPORT_DIR")
    phase17_run_id = os.environ.get("CHOICEFORGE_PHASE17_RUN_ID", "")
    candidate_report_dir = {
        15: phase15_report_dir,
        16: phase16_report_dir,
        17: phase17_report_dir,
    }[candidate_phase]
    candidate_run_id = {
        15: phase15_run_id,
        16: phase16_run_id,
        17: phase17_run_id,
    }[candidate_phase]
    phase15_report_sequence = 0
    candidate_queue = []
    captured_flow = {}

    def write_phase15_report(payload):
        """Write one deterministic device-resident candidate record."""
        if not candidate_report_dir:
            return
        from pathlib import Path
        import json
        import re

        nonlocal phase15_report_sequence
        phase15_report_sequence += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", trace_label).strip("-")
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate_run_id).strip("-")
        prefix = f"{safe_run_id}_" if safe_run_id else ""
        filename = (
            Path(candidate_report_dir)
            / f"{prefix}batch_{phase15_report_sequence:03d}_{safe_label}.json"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def strict_cuda_inputs(call_spec, dataframe, call_locals):
        """Build the shared strict document and typed real-batch environment."""
        from choiceforge.cuda_skims import cuda_wrapper_from_activitysim

        spec_frame = call_spec.reset_index()
        if "Expression" not in spec_frame:
            raise ValueError("ActivitySim spec reset did not expose Expression")
        targeted = [
            value for value in skims.values()
            if getattr(value, "df", None) is not None
        ]
        if not targeted:
            raise ValueError("strict CUDA candidate found no targeted skim wrapper")
        strict_locals = state.get_global_constants().copy()
        strict_locals.update(call_locals or {})
        # The GPU binder needs array identity and dtype, not pandas indexing.
        # Materialize each zero-copy view once so a compiled plan can validate
        # its ABI without repeatedly constructing Series objects.
        column_arrays = {
            column: dataframe[column].to_numpy(copy=False)
            for column in dataframe.columns
        }
        environment = {"df": column_arrays, **strict_locals}
        environment.update({
            name: cuda_wrapper_from_activitysim(value)
            for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims", "odr_skims", "dor_skims"}
        })
        if "od_skims" in skims:
            environment["od_skims_reverse"] = cuda_wrapper_from_activitysim(
                skims["od_skims"], reverse=True
            )
        environment.update(column_arrays)
        document, ir_cache_hit, ir_compile_ms = _cached_strict_ir(spec_frame)
        return document, environment, ir_cache_hit, ir_compile_ms

    def strict_cuda_comparison(raw_utilities):
        """Require exact shared-IR CPU/CUDA equality on a real batch.

        The generated values are qualification evidence only. Sharrow's
        utilities remain authoritative until a later production enablement
        phase passes the complete-model gate.
        """
        from pathlib import Path
        import re

        from choiceforge.sharrow_cuda import (
            compare_strict_cpu_cuda,
            evaluate_strict_cuda,
        )
        from choiceforge.sharrow_ir import (
            evaluate_strict_cpu,
            write_comparison_report,
        )

        spec_frame = spec.reset_index()
        if "Expression" not in spec_frame:
            raise ValueError("ActivitySim spec reset did not expose Expression")
        targeted = [value for value in skims.values() if getattr(value, "df", None) is not None]
        if not targeted:
            raise ValueError("strict CUDA gate found no targeted Sharrow skim wrapper")
        dataframe = targeted[0].df
        strict_locals = state.get_global_constants().copy()
        strict_locals.update(locals_dict)
        environment = {"df": dataframe, **strict_locals}
        environment.update({
            name: value for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims"}
        })
        for column in dataframe.columns:
            environment[column] = dataframe[column].to_numpy(copy=False)
        document, ir_cache_hit, ir_compile_ms = _cached_strict_ir(spec_frame)
        cpu = evaluate_strict_cpu(
            document,
            environment,
            rows=len(dataframe),
            expression_dtype=(
                "float32" if strict_cuda_expression_float32 else "float64"
            ),
        )
        cuda = evaluate_strict_cuda(
            document,
            environment,
            rows=len(dataframe),
            locality_tile_rows=strict_cuda_tile_rows,
            locality_optimized=strict_cuda_cooperative,
            compact_inputs=strict_cuda_compact_inputs,
            group_skim_indices=strict_cuda_grouped_indices,
            sparse_zero_coefficients=strict_cuda_sparse_coefficients,
            expression_float32=strict_cuda_expression_float32,
            persistent_plan=strict_cuda_persistent_plan,
            reuse_buffers=strict_cuda_reuse_buffers,
            fused_utility_accumulation=strict_cuda_sharrow_fma,
        )
        report = compare_strict_cpu_cuda(
            cpu, cuda, row_labels=raw_utilities.index.to_numpy(copy=False)
        )
        report["trace_label"] = trace_label
        report["activitysim_authoritative"] = True
        report["comparison_mode"] = "require_exact"
        report["ir_cache_hit"] = ir_cache_hit
        report["ir_compile_ms"] = ir_compile_ms
        nonlocal strict_cuda_report_sequence
        strict_cuda_report_sequence += 1
        if strict_cuda_report_dir:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", trace_label).strip("-")
            filename = Path(strict_cuda_report_dir) / f"batch_{strict_cuda_report_sequence:03d}_{safe_label}.json"
            write_comparison_report(report, filename)
        logger.info(
            "%s strict-cuda gate exact=%s rows=%d terms=%d alternatives=%d "
            "compiled=%s kernel=%.3fms",
            trace_label,
            report["exact_gate_passed"],
            report["rows"],
            report["terms"],
            report["alternatives"],
            report["kernel"]["compiled_this_call"],
            report["kernel"]["kernel_ms"],
        )
        if not report["exact_gate_passed"]:
            raise AssertionError(
                "strict IR CPU/CUDA exact gate failed; see Phase 14 report"
            )

    def strict_cpu_comparison(raw_utilities):
        """Run the Phase 13 strict CPU oracle beside Sharrow and report.

        ActivitySim remains authoritative.  Exact-gate enforcement is opt-in;
        observation mode records expected semantic differences without ever
        changing a production utility or random draw.
        """
        from pathlib import Path
        import re

        from choiceforge.sharrow_ir import (
            compare_strict_to_sharrow,
            evaluate_strict_cpu,
            write_comparison_report,
        )

        if not captured_flow:
            raise RuntimeError("strict CPU gate did not capture the Sharrow flow")
        spec_frame = spec.reset_index()
        if "Expression" not in spec_frame:
            raise ValueError("ActivitySim spec reset did not expose Expression")
        targeted = [value for value in skims.values() if getattr(value, "df", None) is not None]
        if not targeted:
            raise ValueError("strict CPU gate found no targeted Sharrow skim wrapper")
        dataframe = targeted[0].df
        strict_locals = state.get_global_constants().copy()
        strict_locals.update(locals_dict)
        environment = {"df": dataframe, **strict_locals}
        environment.update({
            name: value for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims"}
        })
        for column in dataframe.columns:
            environment[column] = dataframe[column].to_numpy(copy=False)
        document, _, _ = _cached_strict_ir(spec_frame)
        strict = evaluate_strict_cpu(document, environment, rows=len(dataframe))
        sharrow_features = np.asarray(
            captured_flow["flow"].load(
                source=captured_flow["tree"], dtype=np.float32
            )
        )
        report = compare_strict_to_sharrow(
            strict,
            sharrow_features,
            raw_utilities.to_numpy(dtype=np.float32, copy=False),
            row_labels=raw_utilities.index.to_numpy(copy=False),
            trace_label=trace_label,
        )
        report["activitysim_authoritative"] = True
        report["sharrow_flow_hash"] = str(
            getattr(captured_flow["flow"], "flow_hash", "unknown")
        )
        report["comparison_mode"] = "require_exact" if strict_require_exact else "observe"
        nonlocal strict_report_sequence
        strict_report_sequence += 1
        if strict_report_dir:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", trace_label).strip("-")
            filename = Path(strict_report_dir) / f"batch_{strict_report_sequence:03d}_{safe_label}.json"
            write_comparison_report(report, filename)
        logger.info(
            "%s strict-cpu gate exact=%s rows=%d terms=%d alternatives=%d "
            "divergent_terms=%d divergent_alternatives=%d",
            trace_label,
            report["exact_gate_passed"],
            report["rows"],
            report["terms"],
            report["alternatives"],
            report["feature_comparison"]["divergent_terms"],
            report["utility_comparison"]["divergent_alternatives"],
        )
        if strict_require_exact and not report["exact_gate_passed"]:
            raise AssertionError(
                "strict CPU/Sharrow exact gate failed; see Phase 13 comparison report"
            )

    def shadow_gpu_utilities(raw_utilities):
        """Compare a real Sharrow batch with the device utility compiler.

        This executes only when explicitly requested by an environment variable
        and never supplies values to ActivitySim, preserving the existing
        byte-identical nested-logsum route during proof collection.
        """
        from choiceforge.activitysim_expression import lower_activitysim_utility_spec
        from choiceforge.cuda_backend import _cupy
        from choiceforge.cuda_skims import activitysim_cuda_environment, cuda_wrapper_from_activitysim
        from choiceforge.destination_utility import LoweredDestinationUtility

        spec_frame = spec.reset_index()
        if "Expression" not in spec_frame:
            # ActivitySim's expression index normally retains this name.  Do
            # not infer a column if a framework revision changes that contract.
            raise ValueError("ActivitySim spec reset did not expose Expression")
        cuda_wrappers = {
            name: cuda_wrapper_from_activitysim(value)
            for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims"}
        }
        shadow_locals = state.get_global_constants().copy()
        shadow_locals.update(locals_dict)
        environment = activitysim_cuda_environment(
            next(iter(cuda_wrappers.values())).dataframe, shadow_locals, **cuda_wrappers
        )
        model64, feature64 = lower_activitysim_utility_spec(
            spec_frame, environment, xp=_cupy(), dtype=_cupy().float64
        )
        features = feature64.astype(_cupy().float32)
        model = LoweredDestinationUtility(
            model64.feature_names, model64.alternative_names, model64.coefficients,
            model64.constants, compute_dtype="float32",
        )
        gpu = model.cuda(features, ordered=True)
        cpu = raw_utilities.to_numpy(dtype=np.float64, copy=False)
        # Split a failed shadow comparison into framework-expression versus
        # device-execution error.  The CPU AST uses the exact currently
        # targeted Sharrow wrappers, while the GPU AST uses their device gather
        # adapters.  This diagnostic never supplies a value to ActivitySim.
        dataframe = next(iter(cuda_wrappers.values())).dataframe
        cpu_environment = {"df": dataframe, **shadow_locals}
        cpu_environment.update(cuda_wrappers)
        cpu_environment.update({
            name: value for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims"}
        })
        for column in dataframe.columns:
            if np.asarray(dataframe[column]).dtype.kind in "biuf":
                cpu_environment[column] = dataframe[column].to_numpy(copy=False)
        cpu_model64, cpu_feature64 = lower_activitysim_utility_spec(
            spec_frame, cpu_environment, dtype=np.float64
        )
        cpu_features = cpu_feature64.astype(np.float32)
        cpu_model = LoweredDestinationUtility(
            cpu_model64.feature_names, cpu_model64.alternative_names,
            cpu_model64.coefficients, cpu_model64.constants, compute_dtype="float32",
        )
        cpu_ast = cpu_model.cpu_reference(cpu_features, ordered=True)
        if captured_flow:
            sharrow_features = np.asarray(
                captured_flow["flow"].load(
                    source=captured_flow["tree"], dtype=np.float32
                )
            )
            if sharrow_features.shape == cpu_features.shape:
                feature_delta = np.abs(sharrow_features - cpu_features)
                row, feature = np.unravel_index(np.nanargmax(feature_delta), feature_delta.shape)
                logger.warning(
                    "%s utility-shadow feature_ast_vs_sharrow max_abs=%.3e row=%s feature=%s actual=%.12g expected=%.12g",
                    trace_label, float(feature_delta[row, feature]), raw_utilities.index[row],
                    cpu_model.feature_names[feature], cpu_features[row, feature], sharrow_features[row, feature],
                )
            else:
                logger.warning(
                    "%s utility-shadow feature capture shape mismatch sharrow=%s ast=%s",
                    trace_label, sharrow_features.shape, cpu_features.shape,
                )
        if tuple(model.alternative_names) != tuple(raw_utilities.columns):
            raise ValueError("lowered alternative order differs from Sharrow utility order")
        if gpu.shape != cpu.shape:
            raise ValueError("lowered utility shape differs from Sharrow utility shape")
        max_abs = float(np.max(np.abs(gpu - cpu)))
        if not np.allclose(gpu, cpu, rtol=1e-11, atol=1e-11, equal_nan=True):
            def detail(label, actual, expected):
                delta = np.abs(actual - expected)
                row, alternative = np.unravel_index(np.nanargmax(delta), delta.shape)
                logger.warning(
                    "%s utility-shadow %s max_abs=%.3e row=%s alternative=%s actual=%.12g expected=%.12g",
                    trace_label, label, float(delta[row, alternative]), raw_utilities.index[row],
                    raw_utilities.columns[alternative], actual[row, alternative], expected[row, alternative],
                )
            detail("cpu_ast_vs_sharrow", cpu_ast, cpu)
            detail("gpu_ast_vs_cpu_ast", gpu, cpu_ast)
            raise AssertionError(f"GPU utility shadow mismatch max_abs={max_abs:.3e}")
        logger.info(
            "%s ChoiceForge utility-shadow PASS rows=%d features=%d max_abs=%.3e",
            trace_label, gpu.shape[0], features.shape[1], max_abs,
        )

    def cuda_reducer(raw_utilities, numeric_nest):
        nonlocal shadow_remaining, strict_remaining, strict_cuda_remaining
        try:
            if strict_cuda_remaining > 0:
                strict_cuda_remaining -= 1
                strict_cuda_comparison(raw_utilities)
            if strict_remaining > 0:
                strict_remaining -= 1
                strict_cpu_comparison(raw_utilities)
            if shadow_remaining > 0:
                shadow_remaining -= 1
                shadow_gpu_utilities(raw_utilities)
            if strict_cuda_candidate and candidate_queue:
                entry = candidate_queue.pop(0)
                if entry["rows"] != len(raw_utilities):
                    raise ValueError("strict CUDA candidate row queue mismatch")
                if tuple(entry["alternatives"]) != tuple(raw_utilities.columns):
                    raise ValueError("strict CUDA candidate alternative queue mismatch")
                logsums, nested = mtc21_nested_logsums_cuda(
                    entry["utilities"],
                    numeric_nest,
                    raw_utilities.columns,
                    return_telemetry=True,
                    return_device=device_logsum_sink is not None,
                    numeric_policy="activitysim_pandas_float64",
                )
                if device_logsum_sink is not None:
                    sink_metadata = {
                        "trace_label": trace_label,
                        "chooser_ids": entry["chooser_ids"],
                        "start": entry["start"],
                        "end": entry["end"],
                        "out_period": entry["out_period"],
                        "in_period": entry["in_period"],
                    }
                    device_logsum_sink(
                        logsums,
                        sink_metadata,
                    )
                    if resident_invocation_sink is not None:
                        resident_invocation_sink(
                            entry["resident_invocation"],
                            numeric_nest,
                            tuple(raw_utilities.columns),
                            sink_metadata,
                            logsums,
                        )
                telemetry = entry["telemetry"]
                write_phase15_report({
                    "phase": candidate_phase,
                    "trace_label": trace_label,
                    "rows": len(raw_utilities),
                    "terms": telemetry.terms,
                    "alternatives": telemetry.alternatives,
                    "candidate_used": True,
                    "fallback_used": False,
                    "device_resident_utility_handoff": True,
                    "utility_device_to_host_bytes": 0,
                    "nested_host_to_device_bytes": 0,
                    "logsum_device_sink_used": device_logsum_sink is not None,
                    "logsum_device_to_host_bytes": (
                        0 if device_logsum_sink is not None else int(logsums.nbytes)
                    ),
                    "input_bytes": telemetry.input_bytes,
                    "binding_resolve_ms": telemetry.binding_resolve_ms,
                    "host_pack_ms": telemetry.host_pack_ms,
                    "input_upload_ms": telemetry.input_upload_ms,
                    "coefficient_upload_ms": telemetry.coefficient_upload_ms,
                    "kernel_ms": telemetry.kernel_ms,
                    "nested_kernel_ms": nested.kernel_ms,
                    "nested_download_ms": nested.device_to_host_ms,
                    "compiled_this_call": telemetry.compiled_this_call,
                    "coefficient_cache_hit": telemetry.coefficient_cache_hit,
                    "cache_key": telemetry.cache_key,
                    "source_sha256": telemetry.source_sha256,
                    "tile_rows": telemetry.tile_rows,
                    "dense_row_inputs": telemetry.dense_row_inputs,
                    "scalar_inputs": telemetry.scalar_inputs,
                    "unique_skim_bindings": telemetry.unique_skim_bindings,
                    "skim_reference_uses": telemetry.skim_reference_uses,
                    "float_input_sources": [
                        ":".join(map(str, source))
                        for source in entry["resident_invocation"].float_input_sources
                    ],
                    "int_input_sources": [
                        ":".join(map(str, source))
                        for source in entry["resident_invocation"].int_input_sources
                    ],
                    "skim_input_sources": [
                        ":".join(map(str, source))
                        for source in entry["resident_invocation"].skim_input_sources
                    ],
                    "skim_loads_avoided_per_row": telemetry.skim_loads_avoided_per_row,
                    "skim_index_groups": telemetry.skim_index_groups,
                    "grouped_skim_indices": telemetry.grouped_skim_indices,
                    "active_coefficients": telemetry.active_coefficients,
                    "zero_coefficient_ops_skipped_per_row": (
                        telemetry.zero_coefficient_ops_skipped_per_row
                    ),
                    "sparse_zero_coefficients": telemetry.sparse_zero_coefficients,
                    "expression_dtype": telemetry.expression_dtype,
                    "persistent_plan": telemetry.persistent_plan,
                    "plan_cache_hit": telemetry.plan_cache_hit,
                    "plan_build_ms": telemetry.plan_build_ms,
                    "reusable_workspace": telemetry.reusable_workspace,
                    "workspace_cache_hit": telemetry.workspace_cache_hit,
                    "fused_utility_accumulation": telemetry.fused_utility_accumulation,
                    "ir_cache_hit": entry["ir_cache_hit"],
                    "ir_compile_ms": entry["ir_compile_ms"],
                    "skim_binding_cache_hits": entry["skim_cache_delta"]["binding_hits"],
                    "skim_binding_cache_misses": entry["skim_cache_delta"]["binding_misses"],
                    "skim_array_uploads": entry["skim_cache_delta"]["array_uploads"],
                })
                logger.info(
                    "%s ChoiceForge strict candidate phase=%d tile_rows=%d rows=%d "
                    "resolve=%.3fms pack=%.3fms "
                    "upload=%.3fms plan=%.3fms coefficient=%.3fms utility=%.3fms "
                    "nested=%.3fms download=%.3fms",
                    trace_label,
                    candidate_phase,
                    telemetry.tile_rows,
                    telemetry.rows,
                    telemetry.binding_resolve_ms,
                    telemetry.host_pack_ms,
                    telemetry.input_upload_ms,
                    telemetry.plan_build_ms,
                    telemetry.coefficient_upload_ms,
                    telemetry.kernel_ms,
                    nested.kernel_ms,
                    nested.device_to_host_ms,
                )
                if device_logsum_sink is not None and materialize_device_sink_result:
                    from choiceforge.cuda_backend import _cupy

                    dataframe_logsums = _cupy().asnumpy(logsums)
                else:
                    dataframe_logsums = logsums
                return pd.DataFrame(
                    {
                        "root": (
                            np.ones(len(raw_utilities), dtype=np.float64)
                            if device_logsum_sink is not None
                            and not materialize_device_sink_result
                            else np.exp(dataframe_logsums)
                        )
                    },
                    index=raw_utilities.index,
                )
            materialize_started = time.perf_counter()
            utilities = raw_utilities.to_numpy(copy=False)
            materialize_ms = (time.perf_counter() - materialize_started) * 1000
            logsums, telemetry = mtc21_nested_logsums_cuda(
                utilities,
                numeric_nest,
                raw_utilities.columns,
                return_telemetry=True,
            )
            logger.info(
                "%s ChoiceForge nested-logit rows=%d utility_bytes=%.3fMB "
                "materialize=%.3fms h2d=%.3fms kernel=%.3fms d2h=%.3fms",
                trace_label,
                telemetry.rows,
                telemetry.input_bytes / 1_000_000,
                materialize_ms,
                telemetry.host_to_device_ms,
                telemetry.kernel_ms,
                telemetry.device_to_host_ms,
            )
            # eval_nl_logsums consumes only the root column unless tracing is on.
            return pd.DataFrame(
                {"root": np.exp(logsums)}, index=raw_utilities.index
            )
        except Exception as exc:
            if strict_cuda_candidate and "entry" in locals():
                try:
                    sharrow_values, _, _ = original_apply_flow(
                        *entry["fallback_args"], **entry["fallback_kwargs"]
                    )
                    if sharrow_values is None:
                        raise RuntimeError("Sharrow fallback returned no utilities")
                    sharrow_frame = pd.DataFrame(
                        sharrow_values,
                        index=raw_utilities.index,
                        columns=raw_utilities.columns,
                    )
                    write_phase15_report({
                        "phase": candidate_phase,
                        "trace_label": trace_label,
                        "rows": len(raw_utilities),
                        "candidate_used": False,
                        "fallback_used": True,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    })
                    logger.warning(
                        "%s strict CUDA candidate fell back to Sharrow: %s",
                        trace_label,
                        exc,
                    )
                    return original_reducer(sharrow_frame, numeric_nest)
                except Exception:
                    logger.exception(
                        "%s strict CUDA candidate Sharrow fallback failed",
                        trace_label,
                    )
                    raise
            logger.warning(
                "%s ChoiceForge CUDA nested-logit fallback: %s", trace_label, exc,
                exc_info=bool(os.environ.get("CHOICEFORGE_UTILITY_SHADOW_BATCHES")),
            )
            return original_reducer(raw_utilities, numeric_nest)

    simulate.compute_nested_exp_utilities = cuda_reducer
    original_apply_flow = None
    if shadow_remaining or strict_remaining or strict_cuda_candidate:
        from activitysim.core import flow as activitysim_flow
        original_apply_flow = activitysim_flow.apply_flow

        def capture_apply_flow(*args, **kwargs):
            if strict_cuda_candidate:
                from choiceforge.cuda_backend import _cupy
                from choiceforge.cuda_skims import cuda_dataset_cache_stats
                from choiceforge.sharrow_cuda import evaluate_strict_cuda

                try:
                    call_spec = args[1] if len(args) > 1 else kwargs["spec"]
                    dataframe = args[2] if len(args) > 2 else kwargs["choosers"]
                    call_locals = (
                        args[3] if len(args) > 3 else kwargs.get("locals_d", {})
                    )
                    if (
                        strict_cuda_candidate_max_rows > 0
                        and len(dataframe) > strict_cuda_candidate_max_rows
                    ):
                        write_phase15_report({
                            "phase": candidate_phase,
                            "trace_label": trace_label,
                            "rows": len(dataframe),
                            "candidate_used": False,
                            "fallback_used": True,
                            "fallback_reason": (
                                "row_policy: "
                                f"rows={len(dataframe)} exceeds "
                                f"max_rows={strict_cuda_candidate_max_rows}"
                            ),
                        })
                        logger.info(
                            "%s strict CUDA candidate policy fallback rows=%d max_rows=%d",
                            trace_label,
                            len(dataframe),
                            strict_cuda_candidate_max_rows,
                        )
                        return original_apply_flow(*args, **kwargs)
                    cache_before = cuda_dataset_cache_stats()
                    document, environment, ir_cache_hit, ir_compile_ms = strict_cuda_inputs(
                        call_spec, dataframe, call_locals
                    )
                    generated = evaluate_strict_cuda(
                        document,
                        environment,
                        rows=len(dataframe),
                        return_device=True,
                        capture_features=False,
                        locality_tile_rows=strict_cuda_tile_rows,
                        locality_optimized=strict_cuda_cooperative,
                        compact_inputs=strict_cuda_compact_inputs,
                        group_skim_indices=strict_cuda_grouped_indices,
                        sparse_zero_coefficients=strict_cuda_sparse_coefficients,
                        expression_float32=strict_cuda_expression_float32,
                        persistent_plan=strict_cuda_persistent_plan,
                        reuse_buffers=strict_cuda_reuse_buffers,
                        fused_utility_accumulation=strict_cuda_sharrow_fma,
                        # Keep the immutable ABI descriptor even when no
                        # downstream scheduling sink is installed. Phase 35
                        # uses its declared sources to replace dense trip
                        # preprocessing; the arrays already belong to this
                        # invocation, so this does not make another copy.
                        capture_resident_invocation=True,
                    )
                    cache_after = cuda_dataset_cache_stats()
                    cache_delta = {
                        key: cache_after[key] - cache_before[key]
                        for key in ("binding_hits", "binding_misses", "array_uploads")
                    }
                    candidate_entry = {
                        "rows": len(dataframe),
                        "alternatives": tuple(document["alternatives"]),
                        "utilities": generated.utilities,
                        "resident_invocation": generated.resident_invocation,
                        "telemetry": generated.telemetry,
                        "ir_cache_hit": ir_cache_hit,
                        "ir_compile_ms": ir_compile_ms,
                        "skim_cache_delta": cache_delta,
                        "fallback_args": args,
                        "fallback_kwargs": kwargs,
                    }
                    # Sharrow may lower ``dataframe`` to only referenced
                    # utility leaves. Scheduling identity remains on the
                    # outer frame, but ordinary destination rows do not have
                    # scheduling start/end fields and must never read them.
                    candidate_entry.update(
                        _candidate_sink_metadata(
                            choosers,
                            trace_label,
                            required=device_logsum_sink is not None,
                        )
                    )
                    candidate_queue.append(candidate_entry)
                    # eval_utilities needs only a correctly shaped host object;
                    # the reducer consumes the queued device matrix directly.
                    placeholder = np.zeros(
                        (len(dataframe), len(document["alternatives"])),
                        dtype=np.float32,
                    )
                    return placeholder, None, None
                except Exception as exc:
                    logger.warning(
                        "%s strict CUDA candidate generation fallback: %s",
                        trace_label,
                        exc,
                        exc_info=True,
                    )
                    write_phase15_report({
                        "phase": candidate_phase,
                        "trace_label": trace_label,
                        "rows": len(dataframe) if "dataframe" in locals() else None,
                        "candidate_used": False,
                        "fallback_used": True,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    })
                    return original_apply_flow(*args, **kwargs)
            result = original_apply_flow(*args, **kwargs)
            if result[0] is not None and result[1] is not None and result[2] is not None:
                captured_flow["flow"], captured_flow["tree"] = result[1], result[2]
            return result

        activitysim_flow.apply_flow = capture_apply_flow
    try:
        return simulate.simple_simulate_logsums(
            state,
            choosers,
            spec,
            nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=state.settings.chunk_size,
            trace_label=trace_label,
            chunk_tag="trip_destination.compute_logsums_tripnum_batched_cuda",
            explicit_chunk_size=explicit_chunk_size,
        )
    finally:
        simulate.compute_nested_exp_utilities = original_reducer
        if original_apply_flow is not None:
            activitysim_flow.apply_flow = original_apply_flow


def choose_trip_destinations_batched(
    state,
    nth_trips,
    alternatives,
    tours_merged,
    model_settings,
    want_logsums,
    want_sample_table,
    size_term_matrix,
    skim_hotel,
    estimator,
    chunk_size,
    trace_label,
):
    """Choose every purpose for one trip number with one preprocessor pass.

    Sampling and final simulation remain purpose-specific. The purpose groups
    are disjoint random-number channel rows, so preparing them together does
    not change any chooser's stream. OD and DP draws remain adjacent and in
    their original order for every chooser.
    """
    from activitysim.abm.models import trip_destination as td
    from activitysim.core import assign, chunk, los, tracing

    if estimator is not None:
        raise DestinationBatchUnsupported("batched trip destination estimation")
    network_los = state.get_injectable("network_los")
    # ActivitySim's current LOS API supports one- and two-zone systems and no
    # longer exports THREE_ZONE; older releases used the integer value 3.
    if network_los.zone_system == getattr(los, "THREE_ZONE", 3):
        raise DestinationBatchUnsupported("batched three-zone destination logsums")

    preflight_settings = state.filesystem.read_model_settings(
        model_settings.LOGSUM_SETTINGS
    )
    if _single_preprocessor_settings(preflight_settings) is None:
        raise DestinationBatchUnsupported(
            "destination logsum model does not have one supported preprocessor"
        )
    preprocessor = _single_preprocessor_settings(preflight_settings)
    spec_name = preprocessor["SPEC"]
    if not spec_name.endswith(".csv"):
        spec_name += ".csv"
    preprocessor_spec = assign.read_assignment_spec(
        state.filesystem.get_config_file_path(spec_name)
    )
    purposes = list(nth_trips["primary_purpose"].drop_duplicates())
    coefficient_sets = [
        state.filesystem.get_segment_coefficients(preflight_settings, purpose)
        for purpose in purposes
    ]
    if not _purpose_invariant_preprocessor(preprocessor_spec, coefficient_sets):
        raise DestinationBatchUnsupported(
            "destination preprocessor references purpose-varying coefficients"
        )

    started = time.perf_counter()
    sampling_seconds = 0.0
    preparation_seconds = 0.0
    bundles = []
    empty_results = []
    for purpose, trips_segment in nth_trips.groupby(
        "primary_purpose", observed=True
    ):
        purpose_trace = tracing.extend_trace_label(trace_label, purpose)
        stage_started = time.perf_counter()
        if os.environ.get("CHOICEFORGE_PHASE41_EXACT_TRIP_SAMPLING", "0") == "1":
            from choiceforge.trip_destination_resident import (
                sample_trip_destinations_resident_exact_abi,
            )

            destination_sample = sample_trip_destinations_resident_exact_abi(
                state,
                primary_purpose=purpose,
                trips=trips_segment,
                alternatives=alternatives,
                model_settings=model_settings,
                size_term_matrix=size_term_matrix,
                skim_hotel=skim_hotel,
                estimator=estimator,
                trace_label=purpose_trace,
            )
        elif os.environ.get("CHOICEFORGE_PHASE40_RESIDENT_TRIP_SAMPLING", "0") == "1":
            from choiceforge.trip_destination_resident import (
                sample_trip_destinations_resident,
            )

            destination_sample = sample_trip_destinations_resident(
                state,
                primary_purpose=purpose,
                trips=trips_segment,
                alternatives=alternatives,
                model_settings=model_settings,
                size_term_matrix=size_term_matrix,
                skim_hotel=skim_hotel,
                estimator=estimator,
                trace_label=purpose_trace,
            )
        elif os.environ.get("CHOICEFORGE_PHASE39_CUDA_TRIP_SAMPLING", "0") == "1":
            from choiceforge.trip_destination_sampling import (
                sample_trip_destinations_cuda,
            )

            destination_sample = sample_trip_destinations_cuda(
                state,
                primary_purpose=purpose,
                trips=trips_segment,
                alternatives=alternatives,
                model_settings=model_settings,
                size_term_matrix=size_term_matrix,
                skim_hotel=skim_hotel,
                estimator=estimator,
                trace_label=purpose_trace,
            )
        else:
            destination_sample = td.trip_destination_sample(
                state,
                primary_purpose=purpose,
                trips=trips_segment,
                alternatives=alternatives,
                model_settings=model_settings,
                size_term_matrix=size_term_matrix,
                skim_hotel=skim_hotel,
                estimator=estimator,
                chunk_size=chunk_size,
                trace_label=purpose_trace,
            )
        sampling_seconds += time.perf_counter() - stage_started
        viable = trips_segment.index.isin(destination_sample.index.unique())
        trips_viable = trips_segment[viable]
        if trips_viable.empty:
            sample = destination_sample
            if want_sample_table:
                sample.set_index(
                    model_settings.ALT_DEST_COL_NAME, append=True, inplace=True
                )
            else:
                sample = None
            empty_results.append(
                (purpose, pd.Series(index=trips_viable.index).to_frame("choice"), sample)
            )
            continue
        stage_started = time.perf_counter()
        bundles.append(
            _prepare_logsum_bundle(
                state,
                purpose,
                trips_viable,
                destination_sample,
                tours_merged,
                model_settings,
                skim_hotel,
                purpose_trace,
            )
        )
        preparation_seconds += time.perf_counter() - stage_started

    if not bundles:
        return empty_results

    combined_skims = _generic_logsum_skims(state)
    all_frames = []
    for bundle in bundles:
        all_frames.extend(bundle["directional_frames"])
    native_raw_capture = {}
    native_shadow = os.environ.get("CHOICEFORGE_PHASE35_NATIVE_TRIP_LOGSUM", "0") == "1"
    native_production = os.environ.get("CHOICEFORGE_PHASE35_NATIVE_TRIP_LOGSUM_PRODUCTION", "0") == "1"
    # Native production consumes each purpose's compact frame directly.  The
    # legacy preprocessor requires one stacked frame, but constructing it and
    # then copying thirty slices back out was pure memory traffic in Phases
    # 35-41.
    phase42_compact = (
        os.environ.get("CHOICEFORGE_PHASE42_NUMERIC_COMPILER", "0") == "1"
    )
    all_combined = None if native_production and phase42_compact else pd.concat(
        [bundle["combined"] for bundle in bundles], axis=0
    )
    preprocess_started = time.perf_counter()
    if native_production:
        # Preserve ActivitySim's exact keyed random-ledger sequence while
        # bypassing every dense assignment expression. The reviewed public
        # preprocessor contains exactly three broadcast lognormal draws.
        random_draw_count = _controlled_random_draw_count(
            state, bundles[0]["logsum_settings"]
        )
        if random_draw_count != 3:
            raise DestinationBatchUnsupported(
                "Phase 35 native trip logsum requires exactly three controlled "
                f"wait-time draws; configured preprocessor declares {random_draw_count}"
            )
        if os.environ.get("CHOICEFORGE_PHASE43_COMPACT_TRIP_STATE", "0") == "1":
            compact_draws = _phase43_compact_draws_for_bundles(
                state,
                bundles,
                random_draw_count,
                include_choice_draws=True,
            )
            for bundle, values in zip(bundles, compact_draws):
                bundle["compact_random_draws"] = values
        else:
            rng = state.get_rn_generator()
            directional_draws = [
                np.asarray(
                    rng.normal_for_df(
                        frame, broadcast=True, size=random_draw_count
                    ),
                    dtype=np.float64,
                )
                for frame in all_frames
            ]
            native_raw_capture["random_draws"] = np.concatenate(
                directional_draws, axis=0
            )
    else:
        with chunk.chunk_log(
            state,
            tracing.extend_trace_label(trace_label, "batched_preprocessor"),
            base=True,
        ):
            supported = _combined_preprocessor(
                state,
                all_frames,
                all_combined,
                bundles[0]["locals"],
                combined_skims,
                bundles[0]["logsum_settings"],
                tracing.extend_trace_label(trace_label, "batched"),
                raw_capture=native_raw_capture,
            )
        if not supported:
            raise AssertionError("preflight accepted a destination preprocessor that later failed")
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    logsums_started = time.perf_counter()
    cursor = 0
    draw_cursor = 0
    for bundle in bundles:
        count = len(bundle["combined"])
        if all_combined is not None:
            bundle["combined"] = all_combined.iloc[cursor : cursor + count].copy()
        cursor += count
        if "compact_random_draws" in bundle:
            bundle_draws = bundle["compact_random_draws"]
        else:
            draws = native_raw_capture.get("random_draws")
            bundle_draws = None if draws is None else draws[draw_cursor : draw_cursor + count]
            draw_cursor += count
        if native_production:
            values = _native_trip_logsum_values(
                state, bundle, combined_skims, bundle_draws
            )
            sample_count = len(bundle["destination_sample"])
            bundle["destination_sample"]["od_logsum"] = values[:sample_count]
            bundle["destination_sample"]["dp_logsum"] = values[sample_count:]
        else:
            _evaluate_logsum_bundle(state, bundle, combined_skims, model_settings)
            if native_shadow:
                candidate = _native_trip_logsum_values(
                    state, bundle, combined_skims, bundle_draws
                )
                sample_count = len(bundle["destination_sample"])
                reference = np.r_[
                    np.asarray(bundle["destination_sample"]["od_logsum"]),
                    np.asarray(bundle["destination_sample"]["dp_logsum"]),
                ]
                equal = np.equal(candidate, reference) | (
                    np.isnan(candidate) & np.isnan(reference)
                )
                mismatches = int(np.count_nonzero(~equal))
                max_abs = float(np.nanmax(np.abs(candidate - reference))) if len(candidate) else 0.0
                _TRIP_NATIVE_LOGSUM_TELEMETRY[-1].update(
                    {
                        "shadow_bit_mismatches": mismatches,
                        "shadow_max_abs_difference": max_abs,
                    }
                )
                # Nested logsum reduction is float64 and may differ in the
                # final representable bit when the same exact float32 utility
                # matrix is launched through a separately compiled native ABI.
                # This is far below the established 1e-5 diagnostic boundary;
                # keep a much tighter 1e-12 fail-closed native gate here.
                if max_abs > 1.0e-12:
                    first = int(np.flatnonzero(~equal)[0])
                    raise AssertionError(
                        "trip native logsum shadow mismatch "
                        f"row={first} actual={candidate[first]!r} expected={reference[first]!r} "
                        f"max_abs={max_abs:.3e}"
                    )
    logsums_seconds = time.perf_counter() - logsums_started

    results = []
    simulation_started = time.perf_counter()
    from activitysim.core import simulate as activitysim_simulate

    simulation_profile = {
        "interaction_seconds": 0.0,
        "utility_seconds": 0.0,
        "probability_seconds": 0.0,
        "choice_seconds": 0.0,
        "interaction_calls": 0,
        "compact_choice_calls": 0,
    }
    phase43_profile = (
        os.environ.get("CHOICEFORGE_PHASE43_COMPACT_TRIP_STATE", "0") == "1"
    )
    if phase43_profile:
        from activitysim.core import interaction_simulate as activitysim_interaction
        from activitysim.core import logit as activitysim_logit

        original_interaction = td.interaction_sample_simulate
        original_eval_interaction = activitysim_interaction.eval_interaction_utilities
        original_utils_to_probs = activitysim_logit.utils_to_probs
        original_make_choices = activitysim_logit.make_choices
        phase43_active_choice = {"index": None, "draws": None}

        def timed_eval_interaction(*args, **kwargs):
            started_profile = time.perf_counter()
            try:
                return original_eval_interaction(*args, **kwargs)
            finally:
                simulation_profile["utility_seconds"] += (
                    time.perf_counter() - started_profile
                )

        def timed_utils_to_probs(*args, **kwargs):
            started_profile = time.perf_counter()
            try:
                return original_utils_to_probs(*args, **kwargs)
            finally:
                simulation_profile["probability_seconds"] += (
                    time.perf_counter() - started_profile
                )

        def timed_make_choices(*args, **kwargs):
            started_profile = time.perf_counter()
            try:
                probs = args[1] if len(args) > 1 else kwargs.get("probs")
                active_index = phase43_active_choice["index"]
                active_draws = phase43_active_choice["draws"]
                if (
                    active_index is None
                    or active_draws is None
                    or not probs.index.equals(active_index)
                ):
                    return original_make_choices(*args, **kwargs)

                # Preserve ActivitySim's validation, choice_maker arithmetic,
                # pandas result types, and bad-choice behavior. Only the keyed
                # random lookup is replaced by the already-advanced compact
                # ledger values for exactly this purpose bundle.
                state_arg = args[0] if args else kwargs["state"]
                trace_label_arg = (
                    args[2] if len(args) > 2 else kwargs.get("trace_label")
                )
                trace_choosers = (
                    args[3] if len(args) > 3 else kwargs.get("trace_choosers")
                )
                allow_bad_probs = (
                    args[4] if len(args) > 4 else kwargs.get("allow_bad_probs", False)
                )
                bad_probs = probs.sum(axis=1).sub(
                    np.ones(len(probs.index))
                ).abs() > 0.001 * np.ones(len(probs.index))
                if bad_probs.any() and not allow_bad_probs:
                    activitysim_logit.report_bad_choices(
                        state_arg,
                        bad_probs,
                        probs,
                        state_arg.settings.skip_failed_choices,
                        trace_label=trace_label_arg,
                        msg="probabilities do not add up to 1",
                        trace_choosers=trace_choosers,
                    )
                choices = pd.Series(
                    activitysim_logit.choice_maker(probs.values, active_draws),
                    index=probs.index,
                )
                choices[bad_probs] = -99
                rands = pd.Series(active_draws, index=probs.index)
                global _PHASE43_CHOICE_DRAWS_CONSUMED
                _PHASE43_CHOICE_DRAWS_CONSUMED += len(active_draws)
                simulation_profile["compact_choice_calls"] += 1
                return choices, rands
            finally:
                simulation_profile["choice_seconds"] += (
                    time.perf_counter() - started_profile
                )

        def timed_interaction(*args, **kwargs):
            started_profile = time.perf_counter()
            simulation_profile["interaction_calls"] += 1
            try:
                return original_interaction(*args, **kwargs)
            finally:
                simulation_profile["interaction_seconds"] += (
                    time.perf_counter() - started_profile
                )

        activitysim_interaction.eval_interaction_utilities = timed_eval_interaction
        activitysim_logit.utils_to_probs = timed_utils_to_probs
        activitysim_logit.make_choices = timed_make_choices
        td.interaction_sample_simulate = timed_interaction

    original_spec_for_segment = activitysim_simulate.spec_for_segment
    if phase42_compact:
        activitysim_simulate.spec_for_segment = lambda *call_args, **call_kwargs: (
            _phase42_simulation_spec(
                original_spec_for_segment, *call_args, **call_kwargs
            )
        )
    try:
        for bundle in bundles:
            if phase43_profile:
                phase43_active_choice["index"] = bundle["trips"].index
                phase43_active_choice["draws"] = bundle.get(
                    "compact_choice_draws"
                )
            destinations = td.trip_destination_simulate(
                state,
                primary_purpose=bundle["purpose"],
                trips=bundle["trips"],
                destination_sample=bundle["destination_sample"],
                model_settings=model_settings,
                want_logsums=want_logsums,
                size_term_matrix=size_term_matrix,
                skim_hotel=skim_hotel,
                estimator=estimator,
                trace_label=bundle["trace_label"].replace(
                    ".compute_logsums_tripnum_batched", ""
                ),
            )
            sample = bundle["destination_sample"]
            if want_sample_table:
                sample.set_index(
                    model_settings.ALT_DEST_COL_NAME, append=True, inplace=True
                )
            else:
                sample = None
            results.append((bundle["purpose"], destinations, sample))
    finally:
        activitysim_simulate.spec_for_segment = original_spec_for_segment
        if phase43_profile:
            td.interaction_sample_simulate = original_interaction
            activitysim_interaction.eval_interaction_utilities = original_eval_interaction
            activitysim_logit.utils_to_probs = original_utils_to_probs
            activitysim_logit.make_choices = original_make_choices
    simulation_seconds = time.perf_counter() - simulation_started

    total_seconds = time.perf_counter() - started
    _TRIP_DESTINATION_STAGE_TELEMETRY.append(
        {
            "trace_label": str(trace_label),
            "purposes": len(bundles),
            "trip_rows": int(sum(len(bundle["trips"]) for bundle in bundles)),
            "sample_rows": int(
                sum(len(bundle["destination_sample"]) for bundle in bundles)
            ),
            "sampling_seconds": float(sampling_seconds),
            "preparation_seconds": float(preparation_seconds),
            "preprocessor_seconds": float(preprocess_ms / 1000),
            "logsums_seconds": float(logsums_seconds),
            "simulation_seconds": float(simulation_seconds),
            "simulation_profile": {
                **simulation_profile,
                "outer_seconds": float(
                    simulation_seconds - simulation_profile["interaction_seconds"]
                ),
            },
            "total_seconds": float(total_seconds),
        }
    )

    logger.info(
        "%s ChoiceForge trip-number batch purposes=%d rows=%d "
        "preprocessor=%.3fms total=%.3fms",
        trace_label,
        len(bundles),
        sum(len(bundle["combined"]) for bundle in bundles),
        preprocess_ms,
        total_seconds * 1000,
    )
    if os.environ.get("CHOICEFORGE_STRICT_CUDA_CANDIDATE", "0") == "1":
        trip_num = int(nth_trips["trip_num"].iloc[0])
        final_intermediate_trip_num = int(nth_trips["trip_count"].max()) - 1
        if trip_num >= final_intermediate_trip_num:
            if os.environ.get("CHOICEFORGE_STRICT_CUDA_MODE_CHOICE", "0") == "1":
                from choiceforge.activitysim_mode_choice import (
                    install_activitysim_trip_mode_candidate,
                )

                install_activitysim_trip_mode_candidate()
                logger.info(
                    "%s retained strict CUDA skim cache for trip mode choice",
                    trace_label,
                )
            else:
                from choiceforge.cuda_skims import clear_cuda_dataset_cache

                clear_cuda_dataset_cache()
                logger.info("%s released strict CUDA skim cache", trace_label)
    return results + empty_results
