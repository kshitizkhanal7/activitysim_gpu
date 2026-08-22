"""Optimized ActivitySim trip-destination helpers.

The public prototype computes two trip-mode-choice logsums for every sampled
destination. ActivitySim normally runs the identical model twice: origin to
sampled stop, then sampled stop to the half-tour destination. This module
stacks those two directions and evaluates the mode-choice model once.
"""

from __future__ import annotations

import logging
import os
import re
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


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


def _combined_preprocessor(
    state,
    frames,
    combined,
    locals_dict,
    skims,
    logsum_settings,
    trace_label,
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

    logsum_settings = state.filesystem.read_model_settings(
        model_settings.LOGSUM_SETTINGS
    )
    coefficients = state.filesystem.get_segment_coefficients(
        logsum_settings, primary_purpose
    )
    bundle_trace = tracing.extend_trace_label(
        trace_label, "compute_logsums_tripnum_batched"
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

    origin_column = "_choiceforge_origin"
    destination_column = "_choiceforge_destination"
    od = choosers.copy()
    dp = choosers.copy()
    od[origin_column] = od[model_settings.TRIP_ORIGIN].to_numpy()
    od[destination_column] = od[model_settings.ALT_DEST_COL_NAME].to_numpy()
    dp[origin_column] = dp[model_settings.ALT_DEST_COL_NAME].to_numpy()
    dp[destination_column] = dp[model_settings.PRIMARY_DEST].to_numpy()
    return {
        "purpose": primary_purpose,
        "trips": trips,
        "destination_sample": destination_sample,
        "od": od,
        "dp": dp,
        "combined": pd.concat((od, dp), axis=0),
        "logsum_settings": logsum_settings,
        "coefficients": coefficients,
        "nest_spec": nest_spec,
        "logsum_spec": logsum_spec,
        "locals": locals_dict,
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
):
    """Keep ActivitySim's evaluator intact and replace only its nest reduction."""
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
    phase15_report_dir = os.environ.get("CHOICEFORGE_PHASE15_REPORT_DIR")
    phase15_run_id = os.environ.get("CHOICEFORGE_PHASE15_RUN_ID", "")
    phase15_report_sequence = 0
    candidate_queue = []
    captured_flow = {}

    def write_phase15_report(payload):
        """Write one deterministic device-resident candidate record."""
        if not phase15_report_dir:
            return
        from pathlib import Path
        import json
        import re

        nonlocal phase15_report_sequence
        phase15_report_sequence += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", trace_label).strip("-")
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase15_run_id).strip("-")
        prefix = f"{safe_run_id}_" if safe_run_id else ""
        filename = (
            Path(phase15_report_dir)
            / f"{prefix}batch_{phase15_report_sequence:03d}_{safe_label}.json"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def strict_cuda_inputs(call_spec, dataframe, call_locals):
        """Build the shared strict document and typed real-batch environment."""
        from choiceforge.sharrow_ir import specification_ir
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
        environment = {"df": dataframe, **strict_locals}
        environment.update({
            name: cuda_wrapper_from_activitysim(value)
            for name, value in skims.items()
            if name in {"od_skims", "odt_skims", "dot_skims"}
        })
        for column in dataframe.columns:
            environment[column] = dataframe[column].to_numpy(copy=False)
        return specification_ir(spec_frame), environment

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
            specification_ir,
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
        document = specification_ir(spec_frame)
        cpu = evaluate_strict_cpu(document, environment, rows=len(dataframe))
        cuda = evaluate_strict_cuda(document, environment, rows=len(dataframe))
        report = compare_strict_cpu_cuda(
            cpu, cuda, row_labels=raw_utilities.index.to_numpy(copy=False)
        )
        report["trace_label"] = trace_label
        report["activitysim_authoritative"] = True
        report["comparison_mode"] = "require_exact"
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
            specification_ir,
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
        document = specification_ir(spec_frame)
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
                )
                telemetry = entry["telemetry"]
                write_phase15_report({
                    "phase": 15,
                    "trace_label": trace_label,
                    "rows": len(raw_utilities),
                    "terms": telemetry.terms,
                    "alternatives": telemetry.alternatives,
                    "candidate_used": True,
                    "fallback_used": False,
                    "device_resident_utility_handoff": True,
                    "utility_device_to_host_bytes": 0,
                    "nested_host_to_device_bytes": 0,
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
                })
                logger.info(
                    "%s ChoiceForge strict candidate rows=%d resolve=%.3fms pack=%.3fms "
                    "upload=%.3fms coefficient=%.3fms utility=%.3fms "
                    "nested=%.3fms download=%.3fms",
                    trace_label,
                    telemetry.rows,
                    telemetry.binding_resolve_ms,
                    telemetry.host_pack_ms,
                    telemetry.input_upload_ms,
                    telemetry.coefficient_upload_ms,
                    telemetry.kernel_ms,
                    nested.kernel_ms,
                    nested.device_to_host_ms,
                )
                return pd.DataFrame(
                    {"root": np.exp(logsums)}, index=raw_utilities.index
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
                        "phase": 15,
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
                            "phase": 15,
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
                    document, environment = strict_cuda_inputs(
                        call_spec, dataframe, call_locals
                    )
                    generated = evaluate_strict_cuda(
                        document,
                        environment,
                        rows=len(dataframe),
                        return_device=True,
                        capture_features=False,
                    )
                    candidate_queue.append({
                        "rows": len(dataframe),
                        "alternatives": tuple(document["alternatives"]),
                        "utilities": generated.utilities,
                        "telemetry": generated.telemetry,
                        "fallback_args": args,
                        "fallback_kwargs": kwargs,
                    })
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
                        "phase": 15,
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
    bundles = []
    empty_results = []
    for purpose, trips_segment in nth_trips.groupby(
        "primary_purpose", observed=True
    ):
        purpose_trace = tracing.extend_trace_label(trace_label, purpose)
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

    if not bundles:
        return empty_results

    combined_skims = _generic_logsum_skims(state)
    all_frames = []
    for bundle in bundles:
        all_frames.extend((bundle["od"], bundle["dp"]))
    all_combined = pd.concat(
        [bundle["combined"] for bundle in bundles], axis=0
    )
    preprocess_started = time.perf_counter()
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
        )
    if not supported:
        raise AssertionError("preflight accepted a destination preprocessor that later failed")
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    cursor = 0
    for bundle in bundles:
        count = len(bundle["combined"])
        bundle["combined"] = all_combined.iloc[cursor : cursor + count].copy()
        cursor += count
        _evaluate_logsum_bundle(state, bundle, combined_skims, model_settings)

    results = []
    for bundle in bundles:
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
            sample.set_index(model_settings.ALT_DEST_COL_NAME, append=True, inplace=True)
        else:
            sample = None
        results.append((bundle["purpose"], destinations, sample))

    logger.info(
        "%s ChoiceForge trip-number batch purposes=%d rows=%d "
        "preprocessor=%.3fms total=%.3fms",
        trace_label,
        len(bundles),
        len(all_combined),
        preprocess_ms,
        (time.perf_counter() - started) * 1000,
    )
    if os.environ.get("CHOICEFORGE_STRICT_CUDA_CANDIDATE", "0") == "1":
        trip_num = int(nth_trips["trip_num"].iloc[0])
        final_intermediate_trip_num = int(nth_trips["trip_count"].max()) - 1
        if trip_num >= final_intermediate_trip_num:
            from choiceforge.cuda_skims import clear_cuda_dataset_cache

            clear_cuda_dataset_cache()
            logger.info("%s released strict CUDA skim cache", trace_label)
    return results + empty_results
