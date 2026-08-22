"""CUDA generator for ChoiceForge strict IR version 3.

Both this backend and :func:`choiceforge.sharrow_ir.evaluate_strict_cpu` consume
the same hashed document. The generated kernel evaluates every expression in
source order, stores one float32 feature value, then performs separate ordered
float32 multiply and add operations for every alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping

import numpy as np

from .sharrow_ir import _resolved_coefficients, _validate_document


_KERNEL_CACHE: dict[str, Any] = {}
_COEFFICIENT_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class InputBinding:
    source: tuple[str, ...]
    value_kind: str
    storage_kind: str
    slot: int
    skim_rank: int = 0


@dataclass(frozen=True)
class StrictCudaTelemetry:
    rows: int
    terms: int
    alternatives: int
    input_bytes: int
    binding_resolve_ms: float
    host_pack_ms: float
    input_upload_ms: float
    coefficient_upload_ms: float
    host_to_device_ms: float
    kernel_ms: float
    device_to_host_ms: float
    cache_key: str
    source_sha256: str
    compiled_this_call: bool
    coefficient_cache_hit: bool
    feature_threads: int


@dataclass(frozen=True)
class StrictCudaResult:
    features: Any
    utilities: Any
    telemetry: StrictCudaTelemetry


@dataclass(frozen=True)
class StrictIrLogsumTelemetry:
    utility: StrictCudaTelemetry
    nested_logsum: Any


def evaluate_strict_cuda(
    document: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    rows: int | None = None,
    coefficient_environment: Mapping[str, Any] | None = None,
    return_device: bool = False,
    capture_features: bool = True,
) -> StrictCudaResult:
    """Generate, cache, and execute a strict CUDA evaluator.

    Input leaves are packed by semantic type. Floating leaves become float64;
    integer and Boolean leaves remain int64 so mask operations retain their
    strict CPU meaning. The cache key contains the IR hash, input schema, and
    generated-source hash; CuPy additionally maintains its normal disk cache.
    """
    from .cuda_backend import _cupy

    _validate_document(document)
    cp = _cupy()
    rows = _infer_rows(environment) if rows is None else int(rows)
    if rows < 0:
        raise ValueError("rows must be nonnegative")
    binding_started = time.perf_counter()
    bindings, values = _bindings(document, environment)
    binding_resolve_ms = (time.perf_counter() - binding_started) * 1000
    if not capture_features and not return_device:
        raise ValueError("capture_features=False is only valid with return_device=True")
    source, source_sha256 = generate_cuda_source(
        document, bindings, capture_features=capture_features
    )
    schema = [
        {"source": binding.source, "kind": binding.value_kind,
         "storage": binding.storage_kind, "slot": binding.slot,
         "skim_rank": binding.skim_rank}
        for binding in bindings
    ]
    cache_payload = json.dumps(
        {"ir": document["sha256"], "schema": schema, "source": source_sha256},
        sort_keys=True, separators=(",", ":"),
    )
    cache_key = f"{document['sha256']}:{hashlib.sha256(cache_payload.encode()).hexdigest()}"
    compiled_this_call = cache_key not in _KERNEL_CACHE
    if compiled_this_call:
        kernel = cp.RawKernel(
            source,
            "choiceforge_strict_ir_v3",
            options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=true"),
        )
        kernel.compile()
        _KERNEL_CACHE[cache_key] = kernel
    kernel = _KERNEL_CACHE[cache_key]

    dense_bindings = [binding for binding in bindings if binding.storage_kind != "skim"]
    float_inputs, int_inputs, host_pack_ms, input_upload_ms = _pack_inputs(
        cp, dense_bindings, values, rows
    )
    skim_arguments = _skim_kernel_arguments(bindings, values)
    coefficient_values = np.ascontiguousarray(
        _resolved_coefficients(
            document, coefficient_environment or {}, dtype=np.float32
        )
    )
    coefficient_digest = hashlib.sha256(coefficient_values.tobytes()).hexdigest()
    coefficient_key = f"{cp.cuda.Device().id}:{document['sha256']}:{coefficient_digest}"
    coefficient_cache_hit = coefficient_key in _COEFFICIENT_CACHE
    coefficient_started = time.perf_counter()
    if coefficient_cache_hit:
        coefficients = _COEFFICIENT_CACHE[coefficient_key]
    else:
        coefficients = cp.asarray(coefficient_values)
        cp.cuda.Stream.null.synchronize()
        _COEFFICIENT_CACHE[coefficient_key] = coefficients
    coefficient_upload_ms = (
        0.0 if coefficient_cache_hit
        else (time.perf_counter() - coefficient_started) * 1000
    )
    features = (
        cp.empty((rows, len(document["terms"])), dtype=cp.float32)
        if capture_features else cp.empty((1,), dtype=cp.float32)
    )
    utilities = cp.empty((rows, len(document["alternatives"])), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    uploaded = time.perf_counter()
    if rows:
        feature_threads = min(256, max(32, _round_up_warp(len(document["terms"]))))
        threads = max(
            feature_threads,
            min(1024, max(32, _round_up_warp(len(document["alternatives"])))),
        )
        kernel(
            (rows,),
            (threads,),
            (
                float_inputs,
                int_inputs,
                coefficients,
                features,
                utilities,
                np.int64(rows),
            ) + skim_arguments,
            shared_mem=len(document["terms"]) * np.dtype(np.float32).itemsize,
        )
        cp.cuda.Stream.null.synchronize()
    calculated = time.perf_counter()
    if return_device:
        result_features = features
        result_utilities = utilities
    else:
        result_features = cp.asnumpy(features)
        result_utilities = cp.asnumpy(utilities)
    downloaded = time.perf_counter()
    return StrictCudaResult(
        features=result_features,
        utilities=result_utilities,
        telemetry=StrictCudaTelemetry(
            rows=rows,
            terms=len(document["terms"]),
            alternatives=len(document["alternatives"]),
            input_bytes=int(
                float_inputs.nbytes + int_inputs.nbytes + coefficients.nbytes
                + sum(
                    values[binding.source].orig.nbytes
                    + values[binding.source].dest.nbytes
                    + (values[binding.source].time.nbytes
                       if values[binding.source].time is not None else 0)
                    for binding in bindings if binding.storage_kind == "skim"
                )
            ),
            binding_resolve_ms=binding_resolve_ms,
            host_pack_ms=host_pack_ms,
            input_upload_ms=input_upload_ms,
            coefficient_upload_ms=coefficient_upload_ms,
            host_to_device_ms=input_upload_ms + coefficient_upload_ms,
            kernel_ms=(calculated - uploaded) * 1000,
            device_to_host_ms=0.0 if return_device else (downloaded - calculated) * 1000,
            cache_key=cache_key,
            source_sha256=source_sha256,
            compiled_this_call=compiled_this_call,
            coefficient_cache_hit=coefficient_cache_hit,
            feature_threads=min(
                256, max(32, _round_up_warp(len(document["terms"])))
            ),
        ),
    )


def generate_cuda_source(
    document: Mapping[str, Any], bindings: list[InputBinding], *, capture_features: bool = True
) -> tuple[str, str]:
    """Emit inspectable CUDA C++ from a strict IR document and typed schema."""
    _validate_document(document)
    refs = {binding.source: _binding_reference(binding) for binding in bindings}
    types = {binding.source: binding.value_kind for binding in bindings}
    feature_threads = min(256, max(32, _round_up_warp(len(document["terms"]))))
    terms_by_thread = [[] for _ in range(feature_threads)]
    float_input_count = sum(binding.storage_kind == "float64" for binding in bindings)
    int_input_count = sum(binding.storage_kind == "int64" for binding in bindings)
    skim_parameters = []
    for binding in bindings:
        if binding.storage_kind != "skim":
            continue
        prefix = f"skim_{binding.slot}"
        skim_parameters.extend((
            f"    const float* {prefix}_data",
            f"    const long long* {prefix}_orig",
            f"    const long long* {prefix}_dest",
        ))
        if binding.skim_rank == 3:
            skim_parameters.append(f"    const long long* {prefix}_time")
        skim_parameters.append(f"    long long {prefix}_dest_count")
        if binding.skim_rank == 3:
            skim_parameters.append(f"    long long {prefix}_time_count")
    skim_signature = (",\n" + ",\n".join(skim_parameters)) if skim_parameters else ""
    for index, term in enumerate(document["terms"]):
        code, _ = _emit_node(term["tree"], refs, types)
        output_line = (
            f"        output_features[row * TERM_COUNT + {index}] = term_{index}_f32;"
            if capture_features else ""
        )
        terms_by_thread[index % feature_threads].append(
            f"            const double term_{index}_f64 = (double)({code});\n"
            f"            const float term_{index}_f32 = __double2float_rn(term_{index}_f64);\n"
            f"            shared_features[{index}] = term_{index}_f32;\n"
            f"{output_line}"
        )
    cases = []
    for thread, thread_terms in enumerate(terms_by_thread):
        if thread_terms:
            cases.append(
                f"        case {thread}: {{\n"
                f"{chr(10).join(thread_terms)}\n"
                f"            break;\n"
                f"        }}"
            )
    source = f'''extern "C" __global__ void choiceforge_strict_ir_v3(
    const double* float_inputs,
    const long long* int_inputs,
    const float* coefficients,
    float* output_features,
    float* output_utilities,
    long long rows{skim_signature}) {{
    constexpr int TERM_COUNT = {len(document["terms"])};
    constexpr int ALTERNATIVE_COUNT = {len(document["alternatives"])};
    constexpr int FLOAT_INPUT_COUNT = {float_input_count};
    constexpr int INT_INPUT_COUNT = {int_input_count};
    const long long row = (long long)blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared_features[];
    if (threadIdx.x < {feature_threads}) {{
        switch ((int)threadIdx.x) {{
{chr(10).join(cases)}
        }}
    }}
    __syncthreads();
    const int alternative = (int)threadIdx.x;
    if (alternative < ALTERNATIVE_COUNT) {{
        float utility = 0.0f;
        #pragma unroll 1
        for (int term = 0; term < TERM_COUNT; ++term) {{
            const float product = __fmul_rn(
                shared_features[term],
                coefficients[term * ALTERNATIVE_COUNT + alternative]
            );
            utility = __fadd_rn(utility, product);
        }}
        output_utilities[row * ALTERNATIVE_COUNT + alternative] = utility;
    }}
}}
'''
    return source, hashlib.sha256(source.encode()).hexdigest()


def compare_strict_cpu_cuda(strict, cuda: StrictCudaResult, *, row_labels=None) -> dict:
    """Exact qualification report for the shared-IR CPU and CUDA targets."""
    if strict.features.shape != cuda.features.shape:
        raise ValueError("strict CPU/CUDA feature shapes differ")
    if strict.utilities.shape != cuda.utilities.shape:
        raise ValueError("strict CPU/CUDA utility shapes differ")
    feature_equal = _equal(strict.features, cuda.features)
    utility_equal = _equal(strict.utilities, cuda.utilities)
    return {
        "schema_version": 1,
        "ir_sha256": strict.ir_sha256,
        "rows": int(strict.features.shape[0]),
        "terms": int(strict.features.shape[1]),
        "alternatives": int(strict.utilities.shape[1]),
        "exact_gate_passed": bool(feature_equal.all() and utility_equal.all()),
        "feature_comparison": {
            "exact_cells": int(feature_equal.sum()),
            "total_cells": int(feature_equal.size),
            "max_abs": _max_abs(strict.features, cuda.features),
            "first_divergence": _first_detail(
                feature_equal, strict.features, cuda.features, row_labels,
                "term", strict.term_labels,
            ),
        },
        "utility_comparison": {
            "exact_cells": int(utility_equal.sum()),
            "total_cells": int(utility_equal.size),
            "max_abs": _max_abs(strict.utilities, cuda.utilities),
            "first_divergence": _first_detail(
                utility_equal, strict.utilities, cuda.utilities, row_labels,
                "alternative", strict.alternative_names,
            ),
        },
        "kernel": {
            "cache_key": cuda.telemetry.cache_key,
            "source_sha256": cuda.telemetry.source_sha256,
            "compiled_this_call": cuda.telemetry.compiled_this_call,
            "input_bytes": cuda.telemetry.input_bytes,
            "binding_resolve_ms": cuda.telemetry.binding_resolve_ms,
            "host_pack_ms": cuda.telemetry.host_pack_ms,
            "input_upload_ms": cuda.telemetry.input_upload_ms,
            "coefficient_upload_ms": cuda.telemetry.coefficient_upload_ms,
            "host_to_device_ms": cuda.telemetry.host_to_device_ms,
            "kernel_ms": cuda.telemetry.kernel_ms,
            "device_to_host_ms": cuda.telemetry.device_to_host_ms,
            "coefficient_cache_hit": cuda.telemetry.coefficient_cache_hit,
            "feature_threads": cuda.telemetry.feature_threads,
        },
    }


def mtc21_logsums_from_strict_ir_cuda(
    document,
    environment,
    nest_spec,
    *,
    rows=None,
    coefficient_environment=None,
    return_telemetry=False,
):
    """Keep generated strict utilities on-device through MTC-21 reduction."""
    from .nested_logit import mtc21_nested_logsums_cuda

    generated = evaluate_strict_cuda(
        document,
        environment,
        rows=rows,
        coefficient_environment=coefficient_environment,
        return_device=True,
        capture_features=False,
    )
    logsums, nested = mtc21_nested_logsums_cuda(
        generated.utilities,
        nest_spec,
        document["alternatives"],
        return_telemetry=True,
    )
    if not return_telemetry:
        return logsums
    return logsums, StrictIrLogsumTelemetry(generated.telemetry, nested)


def clear_strict_cuda_cache() -> None:
    """Clear only ChoiceForge's in-process RawKernel handle cache (for tests)."""
    _KERNEL_CACHE.clear()
    _COEFFICIENT_CACHE.clear()


def _bindings(document, environment):
    sources = []
    seen = set()
    for term in document["terms"]:
        for source in _node_sources(term["tree"]):
            if source not in seen:
                seen.add(source)
                sources.append(source)
    float_slot = 0
    int_slot = 0
    skim_slot = 0
    bindings = []
    values = {}
    for source in sources:
        value = _source_value(source, environment)
        if getattr(value, "choiceforge_device_skim_binding", False):
            binding = InputBinding(
                source, "float", "skim", skim_slot,
                3 if value.time is not None else 2,
            )
            skim_slot += 1
        else:
            kind = _value_kind(value)
            if kind == "float":
                binding = InputBinding(source, kind, "float64", float_slot)
                float_slot += 1
            elif kind in {"int", "bool"}:
                binding = InputBinding(source, kind, "int64", int_slot)
                int_slot += 1
            else:
                raise ValueError(f"strict CUDA input {source!r} has unsupported kind {kind!r}")
        bindings.append(binding)
        values[source] = value
    return bindings, values


def _node_sources(tree):
    op = tree["op"]
    if op == "name":
        yield ("name", tree["name"])
    elif op == "column":
        yield ("column", tree["name"])
    elif op == "skim":
        yield ("skim", tree["direction"], tree["key"]["value"])
    for key in ("arg", "left", "right", "value"):
        child = tree.get(key)
        if isinstance(child, Mapping):
            yield from _node_sources(child)
    for key in ("args", "rights"):
        for child in tree.get(key, []):
            yield from _node_sources(child)
    for child in tree.get("keywords", {}).values():
        yield from _node_sources(child)


def _source_value(source, environment):
    if source[0] == "name":
        return environment[source[1]]
    if source[0] == "column":
        return environment["df"][source[1]]
    wrapper = environment[source[1]]
    if hasattr(wrapper, "strict_binding"):
        return wrapper.strict_binding(source[2])
    return wrapper[source[2]]


def _value_kind(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy(copy=False)
    dtype_value = getattr(value, "dtype", None)
    dtype = np.dtype(dtype_value if dtype_value is not None else np.asarray(value).dtype)
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in "iu":
        return "int"
    if dtype.kind in "fc":
        if dtype.kind == "c":
            raise ValueError("complex strict CUDA inputs are unsupported")
        return "float"
    return "unsupported"


def _pack_inputs(cp, bindings, values, rows):
    """Pack on the host and perform one upload per semantic storage type.

    Phase 14 uploaded every leaf separately and then launched device-side
    column stacking. Real batches have many leaves, so launch/allocation
    overhead dominated the generated kernel. Two contiguous transfers retain
    the identical row-major ABI without changing arithmetic semantics.
    """
    if any(hasattr(values[binding.source], "__cuda_array_interface__") for binding in bindings):
        return _pack_mixed_device_inputs(cp, bindings, values, rows)
    pack_started = time.perf_counter()
    floats = []
    ints = []
    for binding in bindings:
        dtype = np.float64 if binding.storage_kind == "float64" else np.int64
        value = values[binding.source]
        if hasattr(value, "to_numpy"):
            value = value.to_numpy(copy=False)
        if hasattr(value, "__cuda_array_interface__"):
            value = cp.asnumpy(value)
        array = np.asarray(value, dtype=dtype)
        if array.ndim == 0:
            array = np.full(rows, array, dtype=dtype)
        if array.ndim != 1 or int(array.shape[0]) != rows:
            raise ValueError(
                f"strict CUDA input {binding.source!r} produced shape {array.shape}, expected ({rows},)"
            )
        (floats if binding.storage_kind == "float64" else ints).append(array)
    host_float_matrix = (
        np.ascontiguousarray(np.column_stack(floats), dtype=np.float64)
        if floats else np.empty((rows, 0), dtype=np.float64)
    )
    host_int_matrix = (
        np.ascontiguousarray(np.column_stack(ints), dtype=np.int64)
        if ints else np.empty((rows, 0), dtype=np.int64)
    )
    host_pack_ms = (time.perf_counter() - pack_started) * 1000
    upload_started = time.perf_counter()
    float_matrix = cp.asarray(host_float_matrix)
    int_matrix = cp.asarray(host_int_matrix)
    cp.cuda.Stream.null.synchronize()
    input_upload_ms = (time.perf_counter() - upload_started) * 1000
    return float_matrix, int_matrix, host_pack_ms, input_upload_ms


def _pack_mixed_device_inputs(cp, bindings, values, rows):
    """Pack host columns and device skim gathers without a device-to-host copy."""
    pack_started = time.perf_counter()
    groups = {}
    for storage_kind, dtype in (("float64", np.float64), ("int64", np.int64)):
        selected = [binding for binding in bindings if binding.storage_kind == storage_kind]
        host_values, host_slots, device_values, device_slots = [], [], [], []
        for binding in selected:
            value = values[binding.source]
            if hasattr(value, "__cuda_array_interface__"):
                array = value
                if array.ndim != 1 or int(array.shape[0]) != rows:
                    raise ValueError(
                        f"strict CUDA input {binding.source!r} produced shape {array.shape}, expected ({rows},)"
                    )
                device_values.append(array)
                device_slots.append(binding.slot)
            else:
                if hasattr(value, "to_numpy"):
                    value = value.to_numpy(copy=False)
                array = np.asarray(value, dtype=dtype)
                if array.ndim == 0:
                    array = np.full(rows, array, dtype=dtype)
                if array.ndim != 1 or int(array.shape[0]) != rows:
                    raise ValueError(
                        f"strict CUDA input {binding.source!r} produced shape {array.shape}, expected ({rows},)"
                    )
                host_values.append(array)
                host_slots.append(binding.slot)
        host_matrix = (
            np.ascontiguousarray(np.column_stack(host_values), dtype=dtype)
            if host_values else None
        )
        groups[storage_kind] = (
            len(selected), dtype, host_matrix, host_slots, device_values, device_slots
        )
    host_pack_ms = (time.perf_counter() - pack_started) * 1000
    upload_started = time.perf_counter()
    matrices = {}
    for storage_kind, (count, dtype, host, host_slots, device, device_slots) in groups.items():
        matrix = cp.empty((rows, count), dtype=dtype)
        if host is not None:
            matrix[:, host_slots] = cp.asarray(host)
        if device:
            matrix[:, device_slots] = cp.column_stack(
                [cp.asarray(value, dtype=dtype) for value in device]
            )
        matrices[storage_kind] = matrix
    cp.cuda.Stream.null.synchronize()
    input_upload_ms = (time.perf_counter() - upload_started) * 1000
    return matrices["float64"], matrices["int64"], host_pack_ms, input_upload_ms


def _binding_reference(binding):
    if binding.storage_kind == "skim":
        prefix = f"skim_{binding.slot}"
        if binding.skim_rank == 3:
            index = (
                f"(({prefix}_orig[row] * {prefix}_dest_count + {prefix}_dest[row]) "
                f"* {prefix}_time_count + {prefix}_time[row])"
            )
        else:
            index = f"({prefix}_orig[row] * {prefix}_dest_count + {prefix}_dest[row])"
        return f"{prefix}_data[{index}]"
    matrix = "float_inputs" if binding.storage_kind == "float64" else "int_inputs"
    count_name = "FLOAT_INPUT_COUNT" if binding.storage_kind == "float64" else "INT_INPUT_COUNT"
    return f"{matrix}[row * {count_name} + {binding.slot}]"


def _skim_kernel_arguments(bindings, values):
    arguments = []
    for binding in bindings:
        if binding.storage_kind != "skim":
            continue
        value = values[binding.source]
        arguments.extend((value.data, value.orig, value.dest))
        if binding.skim_rank == 3:
            arguments.append(value.time)
        arguments.append(np.int64(value.dest_count))
        if binding.skim_rank == 3:
            arguments.append(np.int64(value.time_count))
    return tuple(arguments)


def _emit_node(tree, refs, types):
    op = tree["op"]
    if op == "const":
        value = tree["value"]
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return f"{value}LL", "int"
        if isinstance(value, float):
            return _float_literal(value), "float"
        raise ValueError(f"strict CUDA cannot emit constant {value!r}")
    if op == "name":
        source = ("name", tree["name"])
        return refs[source], types[source]
    if op == "column":
        source = ("column", tree["name"])
        return refs[source], types[source]
    if op == "skim":
        source = ("skim", tree["direction"], tree["key"]["value"])
        return refs[source], types[source]
    if op in {"neg", "pos", "not"}:
        value, kind = _emit_node(tree["arg"], refs, types)
        if op == "neg":
            return f"(-((double)({value})))", "float"
        if op == "pos":
            return f"(+((double)({value})))", "float"
        if kind == "bool":
            return f"(!((bool)({value})))", "bool"
        if kind == "int":
            return f"(~((long long)({value})))", "int"
        raise ValueError("strict CUDA bitwise invert requires Boolean or integer input")
    if op in {"add", "sub", "mul", "div"}:
        left, _ = _emit_node(tree["left"], refs, types)
        right, _ = _emit_node(tree["right"], refs, types)
        token = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        return f"(((double)({left})) {token} ((double)({right})))", "float"
    if op in {"and", "or"}:
        children = tree.get("args") or [tree["left"], tree["right"]]
        emitted = [_emit_node(child, refs, types) for child in children]
        token = "&" if op == "and" else "|"
        kind = "bool" if all(child_kind == "bool" for _, child_kind in emitted) else "int"
        cast = "bool" if kind == "bool" else "long long"
        code = f" {token} ".join(f"(({cast})({value}))" for value, _ in emitted)
        return f"({code})", kind
    if op == "compare":
        left, _ = _emit_node(tree["left"], refs, types)
        pieces = []
        for operator, right_tree in zip(tree["operators"], tree["rights"]):
            right, _ = _emit_node(right_tree, refs, types)
            token = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[operator]
            pieces.append(f"(((double)({left})) {token} ((double)({right})))")
            left = right
        return "(" + " && ".join(pieces) + ")", "bool"
    if op == "clip":
        value, _ = _emit_node(tree["value"], refs, types)
        code = f"((double)({value}))"
        lower = tree["keywords"].get("lower")
        upper = tree["keywords"].get("upper")
        if tree["args"]:
            lower = tree["args"][0]
        if lower is not None:
            bound, _ = _emit_node(lower, refs, types)
            code = f"(({code}) < ((double)({bound})) ? ((double)({bound})) : ({code}))"
        if upper is not None:
            bound, _ = _emit_node(upper, refs, types)
            code = f"(({code}) > ((double)({bound})) ? ((double)({bound})) : ({code}))"
        return code, "float"
    if op == "maximum":
        emitted = [_emit_node(child, refs, types)[0] for child in tree["args"]]
        code = f"((double)({emitted[0]}))"
        for value in emitted[1:]:
            other = f"((double)({value}))"
            code = f"(((({code}) != ({code})) || (({other}) != ({other}))) ? __longlong_as_double(0x7ff8000000000000ULL) : (({code}) > ({other}) ? ({code}) : ({other})))"
        return code, "float"
    raise ValueError(f"strict CUDA emission for {op!r} is not implemented")


def _float_literal(value):
    if math.isnan(value):
        return "__longlong_as_double(0x7ff8000000000000ULL)"
    if math.isinf(value):
        literal = "__longlong_as_double(0x7ff0000000000000ULL)"
        return literal if value > 0 else f"(-{literal})"
    return f"{value:.17g}"


def _infer_rows(environment):
    for value in environment.values():
        candidates = value.values() if isinstance(value, Mapping) else (value,)
        for candidate in candidates:
            shape = getattr(candidate, "shape", ())
            if shape:
                return int(shape[0])
    raise ValueError("environment needs at least one row-array value")


def _round_up_warp(value):
    return ((value + 31) // 32) * 32


def _equal(left, right):
    return np.equal(left, right) | (np.isnan(left) & np.isnan(right))


def _max_abs(left, right):
    with np.errstate(invalid="ignore"):
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    finite = delta[np.isfinite(delta)]
    return 0.0 if not len(finite) else float(finite.max())


def _first_detail(mask, cpu, cuda, row_labels, dimension_name, dimension_labels):
    positions = np.argwhere(~mask)
    if not len(positions):
        return None
    row, column = (int(x) for x in positions[0])
    label = row if row_labels is None else np.asarray(row_labels)[row]
    if hasattr(label, "item"):
        label = label.item()
    return {
        "row_position": row,
        "row_label": label,
        f"{dimension_name}_position": column,
        dimension_name: dimension_labels[column],
        "cpu": float(cpu[row, column]),
        "cuda": float(cuda[row, column]),
        "abs_difference": float(abs(cpu[row, column] - cuda[row, column])),
    }
