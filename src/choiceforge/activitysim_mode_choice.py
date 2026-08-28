"""Opt-in generated-CUDA utility bridge for ActivitySim trip mode choice.

Phase 17 discovered that bypassing Sharrow during destination logsums displaced
Sharrow's first-use compilation into the later trip-mode-choice component.
This bridge reuses the same strict IR/CUDA backend for that consumer while
leaving ActivitySim's nested-logit probabilities, random draws, and choice
assembly authoritative.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import time

from .activitysim_destination import _cached_strict_ir


logger = logging.getLogger(__name__)
_ORIGINAL_TRIP_MODE_CHOICE_SIMULATE = None
_ORIGINAL_TOUR_MODE_CHOICE_SIMULATE = None
_REPORT_SEQUENCE = 0


def install_activitysim_trip_mode_candidate() -> None:
    """Install the bridge after destination logsums have established the plans."""
    global _ORIGINAL_TRIP_MODE_CHOICE_SIMULATE
    from activitysim.abm.models import trip_mode_choice

    if _ORIGINAL_TRIP_MODE_CHOICE_SIMULATE is not None:
        return
    _ORIGINAL_TRIP_MODE_CHOICE_SIMULATE = trip_mode_choice.mode_choice_simulate
    trip_mode_choice.mode_choice_simulate = _mode_choice_simulate_cuda
    logger.info("installed ChoiceForge strict-CUDA trip-mode utility bridge")


def install_activitysim_tour_mode_candidate() -> None:
    """Install the same generated-CUDA evaluator for primary tour mode choice.

    ActivitySim imports ``run_tour_mode_choice_simulate`` into the component
    module, so both the defining utility module and the already-bound component
    symbol are replaced. At-work mode choice keeps its original binding and is
    outside this qualification boundary.
    """
    global _ORIGINAL_TOUR_MODE_CHOICE_SIMULATE
    from activitysim.abm.models import tour_mode_choice
    from activitysim.abm.models.util import mode

    if _ORIGINAL_TOUR_MODE_CHOICE_SIMULATE is not None:
        return
    _ORIGINAL_TOUR_MODE_CHOICE_SIMULATE = mode.run_tour_mode_choice_simulate
    mode.run_tour_mode_choice_simulate = _tour_mode_choice_simulate_cuda
    tour_mode_choice.run_tour_mode_choice_simulate = _tour_mode_choice_simulate_cuda
    logger.info("installed ChoiceForge strict-CUDA tour-mode utility bridge")


def _mode_choice_simulate_cuda(*args, **kwargs):
    """Replace only Sharrow utility evaluation for one mode-choice segment."""
    return _generated_mode_choice_simulate(
        _ORIGINAL_TRIP_MODE_CHOICE_SIMULATE, "trip_mode_choice", *args, **kwargs
    )


def _tour_mode_choice_simulate_cuda(*args, **kwargs):
    """Replace only Sharrow utility evaluation for one tour-mode segment."""
    return _generated_mode_choice_simulate(
        _ORIGINAL_TOUR_MODE_CHOICE_SIMULATE, "tour_mode_choice", *args, **kwargs
    )


def _generated_mode_choice_simulate(original, component, *args, **kwargs):
    if original is None:
        raise RuntimeError(f"ChoiceForge {component} bridge was not installed")
    from activitysim.core import flow as activitysim_flow

    original_apply_flow = activitysim_flow.apply_flow

    def generated_apply_flow(*flow_args, **flow_kwargs):
        state = flow_args[0] if flow_args else flow_kwargs["state"]
        spec = flow_args[1] if len(flow_args) > 1 else flow_kwargs["spec"]
        choosers = flow_args[2] if len(flow_args) > 2 else flow_kwargs["choosers"]
        locals_d = flow_args[3] if len(flow_args) > 3 else flow_kwargs.get("locals_d", {})
        trace_label = (
            flow_args[4] if len(flow_args) > 4 else flow_kwargs.get("trace_label", "")
        )
        started = time.perf_counter()
        try:
            document, environment, ir_cache_hit, ir_compile_ms = _strict_inputs(
                state, spec, choosers, locals_d
            )
            from .cuda_backend import _cupy
            from .sharrow_cuda import evaluate_strict_cuda

            generated = evaluate_strict_cuda(
                document,
                environment,
                rows=len(choosers),
                return_device=True,
                capture_features=False,
                locality_tile_rows=1,
                locality_optimized=False,
                compact_inputs=True,
                group_skim_indices=True,
                sparse_zero_coefficients=False,
                expression_float32=True,
                persistent_plan=True,
                reuse_buffers=(
                    os.environ.get("CHOICEFORGE_STRICT_CUDA_REUSE_BUFFERS", "0")
                    == "1"
                ),
            )
            download_started = time.perf_counter()
            utilities = _cupy().asnumpy(generated.utilities)
            download_ms = (time.perf_counter() - download_started) * 1000
            telemetry = generated.telemetry
            _write_report({
                "phase": 33 if component == "tour_mode_choice" else 17,
                "component": component,
                "trace_label": trace_label,
                "rows": len(choosers),
                "terms": telemetry.terms,
                "alternatives": telemetry.alternatives,
                "candidate_used": True,
                "fallback_used": False,
                "expression_dtype": telemetry.expression_dtype,
                "persistent_plan": telemetry.persistent_plan,
                "plan_cache_hit": telemetry.plan_cache_hit,
                "plan_build_ms": telemetry.plan_build_ms,
                "reusable_workspace": telemetry.reusable_workspace,
                "workspace_cache_hit": telemetry.workspace_cache_hit,
                "ir_cache_hit": ir_cache_hit,
                "ir_compile_ms": ir_compile_ms,
                "binding_resolve_ms": telemetry.binding_resolve_ms,
                "host_pack_ms": telemetry.host_pack_ms,
                "input_upload_ms": telemetry.input_upload_ms,
                "kernel_ms": telemetry.kernel_ms,
                "utility_download_ms": download_ms,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "cache_key": telemetry.cache_key,
                "source_sha256": telemetry.source_sha256,
            })
            logger.info(
                "%s ChoiceForge %s utilities rows=%d plan_hit=%s "
                "pack=%.3fms upload=%.3fms kernel=%.3fms download=%.3fms",
                trace_label,
                component,
                len(choosers),
                telemetry.plan_cache_hit,
                telemetry.host_pack_ms,
                telemetry.input_upload_ms,
                telemetry.kernel_ms,
                download_ms,
            )
            return utilities, None, None
        except Exception as exc:
            _write_report({
                "phase": 33 if component == "tour_mode_choice" else 17,
                "component": component,
                "trace_label": trace_label,
                "rows": len(choosers),
                "candidate_used": False,
                "fallback_used": True,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            })
            logger.warning(
                "%s strict-CUDA %s utility fallback: %s",
                trace_label,
                component,
                exc,
                exc_info=True,
            )
            return original_apply_flow(*flow_args, **flow_kwargs)

    activitysim_flow.apply_flow = generated_apply_flow
    try:
        return original(*args, **kwargs)
    finally:
        activitysim_flow.apply_flow = original_apply_flow


def _strict_inputs(state, spec, dataframe, locals_d):
    """Build the same zero-copy typed environment used by destination batches."""
    from .cuda_skims import cuda_wrapper_from_activitysim

    spec_frame = spec.reset_index()
    if "Expression" not in spec_frame:
        raise ValueError("trip-mode spec reset did not expose Expression")
    column_arrays = {
        column: dataframe[column].to_numpy(copy=False)
        for column in dataframe.columns
    }
    environment = {"df": column_arrays}
    environment.update(state.get_global_constants())
    environment.update(locals_d or {})
    targeted = 0
    for name in (
        "od_skims",
        "do_skims",
        "odt_skims",
        "dot_skims",
        "odr_skims",
        "dor_skims",
    ):
        value = (locals_d or {}).get(name)
        if value is not None and getattr(value, "df", None) is not None:
            environment[name] = cuda_wrapper_from_activitysim(value)
            targeted += 1
    if (locals_d or {}).get("od_skims") is not None:
        environment["od_skims_reverse"] = cuda_wrapper_from_activitysim(
            locals_d["od_skims"], reverse=True
        )
    if not targeted:
        raise ValueError("trip-mode candidate found no targeted skim wrapper")
    environment.update(column_arrays)
    document, ir_cache_hit, ir_compile_ms = _cached_strict_ir(spec_frame)
    return document, environment, ir_cache_hit, ir_compile_ms


def _write_report(payload):
    report_dir = os.environ.get("CHOICEFORGE_PHASE17_MODE_REPORT_DIR")
    if not report_dir:
        return
    global _REPORT_SEQUENCE
    _REPORT_SEQUENCE += 1
    run_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        os.environ.get("CHOICEFORGE_PHASE17_RUN_ID", ""),
    ).strip("-")
    prefix = f"{run_id}_" if run_id else ""
    path = Path(report_dir) / f"{prefix}mode_{_REPORT_SEQUENCE:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
