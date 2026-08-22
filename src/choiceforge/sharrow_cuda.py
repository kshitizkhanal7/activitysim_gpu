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
_SOURCE_BINDING_CACHE: dict[str, tuple[tuple[str, ...], ...]] = {}


@dataclass(frozen=True)
class InputBinding:
    source: tuple[str, ...]
    value_kind: str
    storage_kind: str
    slot: int
    skim_rank: int = 0
    skim_group: int = -1


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
    tile_rows: int
    dense_row_inputs: int
    scalar_inputs: int
    unique_skim_bindings: int
    skim_reference_uses: int
    skim_loads_avoided_per_row: int
    skim_index_groups: int
    grouped_skim_indices: bool
    active_coefficients: int
    zero_coefficient_ops_skipped_per_row: int
    sparse_zero_coefficients: bool
    expression_dtype: str


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
    locality_tile_rows: int = 1,
    locality_optimized: bool = False,
    compact_inputs: bool = False,
    group_skim_indices: bool = False,
    sparse_zero_coefficients: bool = False,
    expression_float32: bool = False,
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
    locality_tile_rows = int(locality_tile_rows)
    if locality_tile_rows not in {1, 2, 4, 8}:
        raise ValueError("locality_tile_rows must be one of 1, 2, 4, or 8")
    locality_optimized = bool(locality_optimized or locality_tile_rows > 1)
    compact_inputs = bool(compact_inputs or locality_optimized)
    group_skim_indices = bool(group_skim_indices or locality_optimized)
    binding_started = time.perf_counter()
    bindings, values = _bindings(
        document, environment, scalar_inputs=compact_inputs
    )
    binding_resolve_ms = (time.perf_counter() - binding_started) * 1000
    coefficient_values = np.ascontiguousarray(
        _resolved_coefficients(
            document, coefficient_environment or {}, dtype=np.float32
        )
    )
    if not capture_features and not return_device:
        raise ValueError("capture_features=False is only valid with return_device=True")
    source, source_sha256 = generate_cuda_source(
        document,
        bindings,
        capture_features=capture_features,
        locality_tile_rows=locality_tile_rows,
        locality_optimized=locality_optimized,
        group_skim_indices=group_skim_indices,
        coefficient_values=coefficient_values,
        sparse_zero_coefficients=sparse_zero_coefficients,
        expression_float32=expression_float32,
    )
    schema = [
        {"source": binding.source, "kind": binding.value_kind,
         "storage": binding.storage_kind, "slot": binding.slot,
         "skim_rank": binding.skim_rank, "skim_group": binding.skim_group}
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
    (
        float_inputs,
        int_inputs,
        float_scalars,
        int_scalars,
        host_pack_ms,
        input_upload_ms,
    ) = _pack_inputs(
        cp,
        dense_bindings,
        values,
        rows,
        float_dtype=np.float32 if expression_float32 else np.float64,
    )
    skim_arguments = _skim_kernel_arguments(
        bindings, values, grouped_indices=group_skim_indices and not locality_optimized
    )
    coefficient_digest = hashlib.sha256(coefficient_values.tobytes()).hexdigest()
    coefficient_key = f"{cp.cuda.Device().id}:{document['sha256']}:{coefficient_digest}"
    coefficient_cache_hit = coefficient_key in _COEFFICIENT_CACHE
    coefficient_started = time.perf_counter()
    if coefficient_cache_hit:
        coefficients = _COEFFICIENT_CACHE[coefficient_key]
    else:
        coefficients = cp.asarray(coefficient_values)
        _COEFFICIENT_CACHE[coefficient_key] = coefficients
    if not coefficient_cache_hit:
        cp.cuda.Stream.null.synchronize()
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
        feature_threads = (
            256 // locality_tile_rows
            if locality_optimized
            else min(256, max(32, _round_up_warp(len(document["terms"]))))
        )
        threads = (
            256
            if locality_optimized
            else max(
                feature_threads,
                min(1024, max(32, _round_up_warp(len(document["alternatives"])))),
            )
        )
        kernel(
            ((rows + locality_tile_rows - 1) // locality_tile_rows,),
            (threads,),
            (
                float_inputs,
                int_inputs,
                float_scalars,
                int_scalars,
                coefficients,
            ) + (
                features,
                utilities,
                np.int64(rows),
            ) + skim_arguments,
            shared_mem=_shared_memory_bytes(
                len(document["terms"]),
                sum(b.storage_kind == "skim" for b in bindings),
                _skim_group_count(bindings),
                locality_tile_rows,
                locality_optimized,
                group_skim_indices,
                sparse_zero_coefficients,
            ),
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
                float_inputs.nbytes + int_inputs.nbytes
                + float_scalars.nbytes + int_scalars.nbytes
                + coefficients.nbytes
                + _skim_position_bytes(
                    bindings,
                    values,
                    grouped=group_skim_indices and not locality_optimized,
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
                (256 // locality_tile_rows) if locality_optimized
                else 256,
                max(32, _round_up_warp(len(document["terms"])))
            ),
            tile_rows=locality_tile_rows,
            dense_row_inputs=_storage_slot_count(
                bindings, {"float64", "int64"}
            ),
            scalar_inputs=_storage_slot_count(
                bindings, {"scalar_float64", "scalar_int64"}
            ),
            unique_skim_bindings=sum(b.storage_kind == "skim" for b in bindings),
            skim_reference_uses=_skim_reference_uses(document),
            skim_loads_avoided_per_row=(
                max(
                    0,
                    _skim_reference_uses(document)
                    - sum(b.storage_kind == "skim" for b in bindings),
                )
                if locality_optimized else 0
            ),
            skim_index_groups=_skim_group_count(bindings),
            grouped_skim_indices=bool(group_skim_indices),
            active_coefficients=int(np.count_nonzero(coefficient_values)),
            zero_coefficient_ops_skipped_per_row=(
                int(coefficient_values.size - np.count_nonzero(coefficient_values))
                if sparse_zero_coefficients and not locality_optimized else 0
            ),
            sparse_zero_coefficients=bool(
                sparse_zero_coefficients and not locality_optimized
            ),
            expression_dtype="float32" if expression_float32 else "float64",
        ),
    )


def generate_cuda_source(
    document: Mapping[str, Any],
    bindings: list[InputBinding],
    *,
    capture_features: bool = True,
    locality_tile_rows: int = 1,
    locality_optimized: bool = False,
    group_skim_indices: bool = False,
    coefficient_values=None,
    sparse_zero_coefficients: bool = False,
    expression_float32: bool = False,
) -> tuple[str, str]:
    """Emit inspectable CUDA C++ from a strict IR document and typed schema."""
    _validate_document(document)
    if locality_tile_rows not in {1, 2, 4, 8}:
        raise ValueError("locality_tile_rows must be one of 1, 2, 4, or 8")
    tiled = bool(locality_optimized or locality_tile_rows > 1)
    grouped_direct = bool(group_skim_indices and not tiled)
    sparse_direct = bool(sparse_zero_coefficients and not tiled)
    if sparse_direct:
        if coefficient_values is None:
            raise ValueError("sparse coefficient generation requires resolved values")
        coefficient_values = np.asarray(coefficient_values, dtype=np.float32)
        expected = (len(document["terms"]), len(document["alternatives"]))
        if coefficient_values.shape != expected:
            raise ValueError(
                f"coefficient matrix shape {coefficient_values.shape} != {expected}"
            )
    refs = {
        binding.source: _binding_reference(
            binding, tiled=tiled, grouped_indices=grouped_direct
        )
        for binding in bindings
    }
    types = {binding.source: binding.value_kind for binding in bindings}
    feature_threads = (
        256 // locality_tile_rows if tiled
        else min(256, max(32, _round_up_warp(len(document["terms"]))))
    )
    terms_by_thread = [[] for _ in range(feature_threads)]
    float_input_count = _storage_slot_count(bindings, {"float64"})
    int_input_count = _storage_slot_count(bindings, {"int64"})
    float_scalar_count = _storage_slot_count(bindings, {"scalar_float64"})
    int_scalar_count = _storage_slot_count(bindings, {"scalar_int64"})
    skim_bindings = [binding for binding in bindings if binding.storage_kind == "skim"]
    skim_groups = _skim_groups(bindings)
    skim_parameters = []
    if grouped_direct:
        skim_parameters.extend(
            f"    const float* skim_{binding.slot}_data"
            for binding in skim_bindings
        )
        for group, binding in skim_groups:
            prefix = f"skim_group_{group}"
            skim_parameters.extend((
                f"    const long long* {prefix}_orig",
                f"    const long long* {prefix}_dest",
            ))
            if binding.skim_rank == 3:
                skim_parameters.append(f"    const long long* {prefix}_time")
            skim_parameters.append(f"    long long {prefix}_dest_count")
            if binding.skim_rank == 3:
                skim_parameters.append(f"    long long {prefix}_time_count")
    else:
        for binding in skim_bindings:
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
        code, _ = _emit_node(
            term["tree"], refs, types,
            numeric_type="float" if expression_float32 else "double",
        )
        shared_target = (
            f"shared_features[tile_row * TERM_COUNT + {index}]"
            if tiled else f"shared_features[{index}]"
        )
        output_line = (
            f"        output_features[row * TERM_COUNT + {index}] = term_{index}_f32;"
            if capture_features else ""
        )
        finite_guard = (
            f"            thread_nonfinite = thread_nonfinite || "
            f"!isfinite(term_{index}_f32);"
            if sparse_direct else ""
        )
        term_conversion = (
            f"            const float term_{index}_f32 = (float)({code});\n"
            if expression_float32
            else (
                f"            const double term_{index}_f64 = (double)({code});\n"
                f"            const float term_{index}_f32 = "
                f"__double2float_rn(term_{index}_f64);\n"
            )
        )
        terms_by_thread[index % feature_threads].append(
            term_conversion
            +
            f"            {shared_target} = term_{index}_f32;\n"
            f"{finite_guard}\n"
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
    if tiled:
        skim_cases = []
        for binding in skim_bindings:
            skim_cases.append(
                f"                case {binding.slot}: "
                f"shared_skims[gather_row * SKIM_COUNT + {binding.slot}] = "
                f"{_skim_global_reference(binding, 'skim_row')}; break;"
            )
        if locality_tile_rows == 1:
            gather_code = f'''    for (int skim = (int)threadIdx.x; skim < SKIM_COUNT; skim += 256) {{
        const int gather_row = 0;
        const long long skim_row = tile_base;
        if (skim_row < rows) {{
            switch (skim) {{
{chr(10).join(skim_cases)}
            }}
        }}
    }}'''
        else:
            gather_code = f'''    for (int skim = warp; skim < SKIM_COUNT; skim += 8) {{
        if (lane < TILE_ROWS) {{
            const int gather_row = lane;
            const long long skim_row = tile_base + gather_row;
            if (skim_row < rows) {{
                switch (skim) {{
{chr(10).join(skim_cases)}
                }}
            }}
        }}
    }}'''
        source = f'''extern "C" __global__ void choiceforge_strict_ir_v3(
    const {"float" if expression_float32 else "double"}* float_inputs,
    const long long* int_inputs,
    const {"float" if expression_float32 else "double"}* float_scalars,
    const long long* int_scalars,
    const float* coefficients,
    float* output_features,
    float* output_utilities,
    long long rows{skim_signature}) {{
    constexpr int TERM_COUNT = {len(document["terms"])};
    constexpr int ALTERNATIVE_COUNT = {len(document["alternatives"])};
    constexpr int FLOAT_INPUT_COUNT = {float_input_count};
    constexpr int INT_INPUT_COUNT = {int_input_count};
    constexpr int FLOAT_SCALAR_COUNT = {float_scalar_count};
    constexpr int INT_SCALAR_COUNT = {int_scalar_count};
    constexpr int SKIM_COUNT = {len(skim_bindings)};
    constexpr int TILE_ROWS = {locality_tile_rows};
    constexpr int THREADS_PER_ROW = 256 / TILE_ROWS;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int tile_row = (int)threadIdx.x / THREADS_PER_ROW;
    const int row_thread = (int)threadIdx.x - tile_row * THREADS_PER_ROW;
    const long long tile_base = (long long)blockIdx.x * TILE_ROWS;
    const long long row = tile_base + tile_row;
    extern __shared__ float shared_values[];
    float* shared_features = shared_values;
    float* shared_skims = shared_values + TILE_ROWS * TERM_COUNT;

    // Each warp owns a subset of skim cubes while its first TILE_ROWS lanes
    // gather adjacent model rows. This turns repeated per-term coordinate
    // lookups into one cooperative load per unique skim and row.
{gather_code}
    __syncthreads();
    if (row >= rows) return;

    // One warp evaluates one row. Feature expressions remain in source order
    // by index, and each lane owns every 32nd feature without changing the
    // ordered utility accumulation below.
    switch (row_thread) {{
{chr(10).join(cases)}
    }}
    __syncthreads();
    if (row_thread < ALTERNATIVE_COUNT) {{
        float utility = 0.0f;
        #pragma unroll 1
        for (int term = 0; term < TERM_COUNT; ++term) {{
            const float product = __fmul_rn(
                shared_features[tile_row * TERM_COUNT + term],
                coefficients[term * ALTERNATIVE_COUNT + row_thread]
            );
            utility = __fadd_rn(utility, product);
        }}
        output_utilities[row * ALTERNATIVE_COUNT + row_thread] = utility;
    }}
}}
'''
    else:
        if grouped_direct:
            group_cases = []
            for group, binding in skim_groups:
                prefix = f"skim_group_{group}"
                if binding.skim_rank == 3:
                    index = (
                        f"(({prefix}_orig[row] * {prefix}_dest_count + "
                        f"{prefix}_dest[row]) * {prefix}_time_count + "
                        f"{prefix}_time[row])"
                    )
                else:
                    index = (
                        f"({prefix}_orig[row] * {prefix}_dest_count + "
                        f"{prefix}_dest[row])"
                    )
                group_cases.append(
                    f"        case {group}: shared_skim_indices[{group}] = {index}; break;"
                )
        if grouped_direct or sparse_direct:
            feature_bytes_aligned = ((len(document["terms"]) * 4 + 7) // 8) * 8
            skim_index_bytes = len(skim_groups) * 8 if grouped_direct else 0
            shared_declaration = f'''    extern __shared__ unsigned char shared_bytes[];
    float* shared_features = reinterpret_cast<float*>(shared_bytes);
    constexpr int FEATURE_BYTES_ALIGNED = {feature_bytes_aligned};'''
            if grouped_direct:
                shared_declaration += '''
    long long* shared_skim_indices = reinterpret_cast<long long*>(shared_bytes + FEATURE_BYTES_ALIGNED);'''
            if sparse_direct:
                shared_declaration += f'''
    int* shared_nonfinite = reinterpret_cast<int*>(shared_bytes + FEATURE_BYTES_ALIGNED + {skim_index_bytes});'''
        else:
            shared_declaration = "    extern __shared__ float shared_features[];"
        initialization = []
        if sparse_direct:
            initialization.append(
                "    if ((int)threadIdx.x == 0) *shared_nonfinite = 0;"
            )
        if grouped_direct:
            initialization.append(f'''    constexpr int SKIM_GROUP_COUNT = {len(skim_groups)};
    if ((int)threadIdx.x < SKIM_GROUP_COUNT) {{
        switch ((int)threadIdx.x) {{
{chr(10).join(group_cases)}
        }}
    }}''')
        group_load = "\n".join(initialization)
        if initialization:
            group_load += "\n    __syncthreads();"
        if sparse_direct:
            sparse_cases = []
            for alternative in range(len(document["alternatives"])):
                active_terms = np.flatnonzero(
                    coefficient_values[:, alternative] != np.float32(0.0)
                )
                operations = "\n".join(
                    "            utility = __fadd_rn(utility, __fmul_rn("
                    f"shared_features[{int(term)}], "
                    f"coefficients[{int(term)} * ALTERNATIVE_COUNT + {alternative}]));"
                    for term in active_terms
                )
                sparse_cases.append(
                    f"        case {alternative}: {{\n{operations}\n"
                    "            break;\n        }"
                )
            utility_body = f'''        if (*shared_nonfinite) {{
            #pragma unroll 1
            for (int term = 0; term < TERM_COUNT; ++term) {{
                utility = __fadd_rn(utility, __fmul_rn(
                    shared_features[term],
                    coefficients[term * ALTERNATIVE_COUNT + alternative]
                ));
            }}
        }} else {{
            switch (alternative) {{
{chr(10).join(sparse_cases)}
            }}
        }}'''
        else:
            utility_body = '''        #pragma unroll 1
        for (int term = 0; term < TERM_COUNT; ++term) {
            const float product = __fmul_rn(
                shared_features[term],
                coefficients[term * ALTERNATIVE_COUNT + alternative]
            );
            utility = __fadd_rn(utility, product);
        }'''
        thread_nonfinite_declaration = (
            "    bool thread_nonfinite = false;" if sparse_direct else ""
        )
        nonfinite_vote = (
            '''    if (__any_sync(0xffffffff, thread_nonfinite) &&
        (((int)threadIdx.x & 31) == 0)) {
        atomicExch(shared_nonfinite, 1);
    }'''
            if sparse_direct else ""
        )
        source = f'''extern "C" __global__ void choiceforge_strict_ir_v3(
    const {"float" if expression_float32 else "double"}* float_inputs,
    const long long* int_inputs,
    const {"float" if expression_float32 else "double"}* float_scalars,
    const long long* int_scalars,
    const float* coefficients,
    float* output_features,
    float* output_utilities,
    long long rows{skim_signature}) {{
    constexpr int TERM_COUNT = {len(document["terms"])};
    constexpr int ALTERNATIVE_COUNT = {len(document["alternatives"])};
    constexpr int FLOAT_INPUT_COUNT = {float_input_count};
    constexpr int INT_INPUT_COUNT = {int_input_count};
    constexpr int FLOAT_SCALAR_COUNT = {float_scalar_count};
    constexpr int INT_SCALAR_COUNT = {int_scalar_count};
    const long long row = (long long)blockIdx.x;
    if (row >= rows) return;
{shared_declaration}
{group_load}
{thread_nonfinite_declaration}
    if (threadIdx.x < {feature_threads}) {{
        switch ((int)threadIdx.x) {{
{chr(10).join(cases)}
        }}
    }}
{nonfinite_vote}
    __syncthreads();
    const int alternative = (int)threadIdx.x;
    if (alternative < ALTERNATIVE_COUNT) {{
        float utility = 0.0f;
{utility_body}
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
            "tile_rows": cuda.telemetry.tile_rows,
            "dense_row_inputs": cuda.telemetry.dense_row_inputs,
            "scalar_inputs": cuda.telemetry.scalar_inputs,
            "unique_skim_bindings": cuda.telemetry.unique_skim_bindings,
            "skim_reference_uses": cuda.telemetry.skim_reference_uses,
            "skim_loads_avoided_per_row": cuda.telemetry.skim_loads_avoided_per_row,
            "skim_index_groups": cuda.telemetry.skim_index_groups,
            "grouped_skim_indices": cuda.telemetry.grouped_skim_indices,
            "active_coefficients": cuda.telemetry.active_coefficients,
            "zero_coefficient_ops_skipped_per_row": (
                cuda.telemetry.zero_coefficient_ops_skipped_per_row
            ),
            "sparse_zero_coefficients": cuda.telemetry.sparse_zero_coefficients,
            "expression_dtype": cuda.telemetry.expression_dtype,
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
    locality_tile_rows=1,
    locality_optimized=False,
    compact_inputs=False,
    group_skim_indices=False,
    sparse_zero_coefficients=False,
    expression_float32=False,
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
        locality_tile_rows=locality_tile_rows,
        locality_optimized=locality_optimized,
        compact_inputs=compact_inputs,
        group_skim_indices=group_skim_indices,
        sparse_zero_coefficients=sparse_zero_coefficients,
        expression_float32=expression_float32,
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


def _bindings(document, environment, *, scalar_inputs=False):
    source_key = str(document["sha256"])
    if source_key in _SOURCE_BINDING_CACHE:
        sources = _SOURCE_BINDING_CACHE[source_key]
    else:
        ordered_sources = []
        seen = set()
        for term in document["terms"]:
            for source in _node_sources(term["tree"]):
                if source not in seen:
                    seen.add(source)
                    ordered_sources.append(source)
        sources = tuple(ordered_sources)
        _SOURCE_BINDING_CACHE[source_key] = sources
    float_slot = 0
    int_slot = 0
    scalar_float_slot = 0
    scalar_int_slot = 0
    skim_slot = 0
    skim_groups = {}
    compact_slot_maps = {
        "float64": {}, "int64": {}, "scalar_float64": {}, "scalar_int64": {}
    }
    bindings = []
    values = {}
    for source in sources:
        value = _source_value(source, environment)
        if getattr(value, "choiceforge_device_skim_binding", False):
            direction = source[1]
            if direction not in skim_groups:
                skim_groups[direction] = len(skim_groups)
            binding = InputBinding(
                source, "float", "skim", skim_slot,
                3 if value.time is not None else 2,
                skim_groups[direction],
            )
            skim_slot += 1
        else:
            kind = _value_kind(value)
            scalar = bool(scalar_inputs and _is_scalar_value(value))
            if kind == "float" and scalar:
                storage = "scalar_float64"
                identity = _input_storage_identity(value, storage)
                slot = compact_slot_maps[storage].get(identity)
                if slot is None:
                    slot = scalar_float_slot
                    compact_slot_maps[storage][identity] = slot
                    scalar_float_slot += 1
                binding = InputBinding(
                    source, kind, storage, slot
                )
            elif kind == "float":
                storage = "float64"
                if scalar_inputs:
                    identity = _input_storage_identity(value, storage)
                    slot = compact_slot_maps[storage].get(identity)
                else:
                    slot = None
                if slot is None:
                    slot = float_slot
                    float_slot += 1
                    if scalar_inputs:
                        compact_slot_maps[storage][identity] = slot
                binding = InputBinding(source, kind, storage, slot)
            elif kind in {"int", "bool"} and scalar:
                storage = "scalar_int64"
                identity = _input_storage_identity(value, storage)
                slot = compact_slot_maps[storage].get(identity)
                if slot is None:
                    slot = scalar_int_slot
                    compact_slot_maps[storage][identity] = slot
                    scalar_int_slot += 1
                binding = InputBinding(
                    source, kind, storage, slot
                )
            elif kind in {"int", "bool"}:
                storage = "int64"
                if scalar_inputs:
                    identity = _input_storage_identity(value, storage)
                    slot = compact_slot_maps[storage].get(identity)
                else:
                    slot = None
                if slot is None:
                    slot = int_slot
                    int_slot += 1
                    if scalar_inputs:
                        compact_slot_maps[storage][identity] = slot
                binding = InputBinding(source, kind, storage, slot)
            else:
                raise ValueError(f"strict CUDA input {source!r} has unsupported kind {kind!r}")
        bindings.append(binding)
        values[source] = value
    return bindings, values


def _is_scalar_value(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape) == ()
    return np.isscalar(value)


def _input_storage_identity(value, storage_kind):
    target_dtype = (
        np.float64 if storage_kind in {"float64", "scalar_float64"} else np.int64
    )
    if storage_kind.startswith("scalar_"):
        if hasattr(value, "__cuda_array_interface__"):
            from .cuda_backend import _cupy

            value = _cupy().asnumpy(value)
        return np.asarray(value, dtype=target_dtype).reshape(()).tobytes()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy(copy=False)
    array = value
    interface = getattr(array, "__cuda_array_interface__", None)
    if interface is None:
        array = np.asarray(array)
        interface = array.__array_interface__
    return (
        int(interface["data"][0]),
        tuple(array.shape),
        tuple(array.strides) if array.strides is not None else None,
        np.dtype(array.dtype).str,
        np.dtype(target_dtype).str,
    )


def _storage_slot_count(bindings, storage_kinds):
    return len({
        (binding.storage_kind, binding.slot)
        for binding in bindings
        if binding.storage_kind in storage_kinds
    })


def _unique_storage_bindings(bindings, storage_kind):
    """Return one binding for every packed slot, in ABI slot order."""
    unique = {}
    for binding in bindings:
        if binding.storage_kind == storage_kind:
            unique.setdefault(binding.slot, binding)
    return [unique[slot] for slot in sorted(unique)]


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


def _pack_inputs(cp, bindings, values, rows, *, float_dtype=np.float64):
    """Pack on the host and perform one upload per semantic storage type.

    Phase 14 uploaded every leaf separately and then launched device-side
    column stacking. Real batches have many leaves, so launch/allocation
    overhead dominated the generated kernel. Two contiguous transfers retain
    the identical row-major ABI without changing arithmetic semantics.
    """
    if any(
        hasattr(values[binding.source], "__cuda_array_interface__")
        for binding in bindings
    ):
        return _pack_mixed_device_inputs(
            cp, bindings, values, rows, float_dtype=float_dtype
        )
    pack_started = time.perf_counter()
    floats = []
    ints = []
    float_scalars = []
    int_scalars = []
    ordered = []
    for storage_kind in (
        "float64", "int64", "scalar_float64", "scalar_int64"
    ):
        ordered.extend(_unique_storage_bindings(bindings, storage_kind))
    for binding in ordered:
        is_float = binding.storage_kind in {"float64", "scalar_float64"}
        is_scalar = binding.storage_kind.startswith("scalar_")
        dtype = float_dtype if is_float else np.int64
        value = values[binding.source]
        if hasattr(value, "to_numpy"):
            value = value.to_numpy(copy=False)
        array = np.asarray(value, dtype=dtype)
        if is_scalar:
            if array.ndim != 0:
                raise ValueError(
                    f"strict CUDA scalar input {binding.source!r} produced shape {array.shape}"
                )
            (float_scalars if is_float else int_scalars).append(array.item())
            continue
        if array.ndim == 0:
            array = np.full(rows, array, dtype=dtype)
        if array.ndim != 1 or int(array.shape[0]) != rows:
            raise ValueError(
                f"strict CUDA input {binding.source!r} produced shape {array.shape}, expected ({rows},)"
            )
        (floats if is_float else ints).append(array)
    host_float_matrix = (
        np.ascontiguousarray(np.column_stack(floats), dtype=float_dtype)
        if floats else np.empty((rows, 0), dtype=float_dtype)
    )
    host_int_matrix = (
        np.ascontiguousarray(np.column_stack(ints), dtype=np.int64)
        if ints else np.empty((rows, 0), dtype=np.int64)
    )
    host_float_scalars = np.ascontiguousarray(float_scalars, dtype=float_dtype)
    host_int_scalars = np.ascontiguousarray(int_scalars, dtype=np.int64)
    host_pack_ms = (time.perf_counter() - pack_started) * 1000
    upload_started = time.perf_counter()
    float_matrix = cp.asarray(host_float_matrix)
    int_matrix = cp.asarray(host_int_matrix)
    device_float_scalars = cp.asarray(host_float_scalars)
    device_int_scalars = cp.asarray(host_int_scalars)
    cp.cuda.Stream.null.synchronize()
    input_upload_ms = (time.perf_counter() - upload_started) * 1000
    return (
        float_matrix,
        int_matrix,
        device_float_scalars,
        device_int_scalars,
        host_pack_ms,
        input_upload_ms,
    )


def _pack_mixed_device_inputs(
    cp, bindings, values, rows, *, float_dtype=np.float64
):
    """Pack host columns and device skim gathers without a device-to-host copy."""
    pack_started = time.perf_counter()
    groups = {}
    for storage_kind, dtype in (("float64", float_dtype), ("int64", np.int64)):
        selected = _unique_storage_bindings(bindings, storage_kind)
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
    scalar_groups = {}
    for storage_kind, dtype in (
        ("scalar_float64", float_dtype), ("scalar_int64", np.int64)
    ):
        selected = _unique_storage_bindings(bindings, storage_kind)
        host_values, device_values, device_slots = [], [], []
        for binding in selected:
            value = values[binding.source]
            if hasattr(value, "__cuda_array_interface__"):
                if tuple(value.shape) != ():
                    raise ValueError(
                        f"strict CUDA scalar input {binding.source!r} produced shape {value.shape}"
                    )
                device_values.append(value)
                device_slots.append(binding.slot)
            else:
                array = np.asarray(value, dtype=dtype)
                if array.ndim != 0:
                    raise ValueError(
                        f"strict CUDA scalar input {binding.source!r} produced shape {array.shape}"
                    )
                host_values.append((binding.slot, array.item()))
        scalar_groups[storage_kind] = (len(selected), dtype, host_values, device_values, device_slots)
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
    scalar_arrays = {}
    for storage_kind, (count, dtype, host, device, device_slots) in scalar_groups.items():
        array = cp.empty(count, dtype=dtype)
        if host:
            slots, host_values = zip(*host)
            array[list(slots)] = cp.asarray(host_values, dtype=dtype)
        if device:
            array[device_slots] = cp.asarray(device, dtype=dtype).reshape(-1)
        scalar_arrays[storage_kind] = array
    cp.cuda.Stream.null.synchronize()
    input_upload_ms = (time.perf_counter() - upload_started) * 1000
    return (
        matrices["float64"],
        matrices["int64"],
        scalar_arrays["scalar_float64"],
        scalar_arrays["scalar_int64"],
        host_pack_ms,
        input_upload_ms,
    )


def _binding_reference(binding, *, tiled=False, grouped_indices=False):
    if binding.storage_kind == "skim":
        if grouped_indices:
            return (
                f"skim_{binding.slot}_data["
                f"shared_skim_indices[{binding.skim_group}]]"
            )
        return (
            f"shared_skims[tile_row * SKIM_COUNT + {binding.slot}]"
            if tiled else _skim_global_reference(binding, "row")
        )
    if binding.storage_kind == "scalar_float64":
        return f"float_scalars[{binding.slot}]"
    if binding.storage_kind == "scalar_int64":
        return f"int_scalars[{binding.slot}]"
    matrix = "float_inputs" if binding.storage_kind == "float64" else "int_inputs"
    count_name = "FLOAT_INPUT_COUNT" if binding.storage_kind == "float64" else "INT_INPUT_COUNT"
    return f"{matrix}[row * {count_name} + {binding.slot}]"


def _skim_global_reference(binding, row_name):
    prefix = f"skim_{binding.slot}"
    if binding.skim_rank == 3:
        index = (
            f"(({prefix}_orig[{row_name}] * {prefix}_dest_count + "
            f"{prefix}_dest[{row_name}]) * {prefix}_time_count + "
            f"{prefix}_time[{row_name}])"
        )
    else:
        index = (
            f"({prefix}_orig[{row_name}] * {prefix}_dest_count + "
            f"{prefix}_dest[{row_name}])"
        )
    return f"{prefix}_data[{index}]"


def _skim_reference_uses(document):
    return sum(
        source[0] == "skim"
        for term in document["terms"]
        for source in _node_sources(term["tree"])
    )


def _skim_kernel_arguments(bindings, values, *, grouped_indices=False):
    arguments = []
    if grouped_indices:
        skim_bindings = [b for b in bindings if b.storage_kind == "skim"]
        arguments.extend(values[binding.source].data for binding in skim_bindings)
        for group, representative in _skim_groups(bindings):
            value = values[representative.source]
            members = [b for b in skim_bindings if b.skim_group == group]
            for member in members:
                other = values[member.source]
                if (
                    member.skim_rank != representative.skim_rank
                    or other.dest_count != value.dest_count
                    or other.time_count != value.time_count
                    or _device_pointer(other.orig) != _device_pointer(value.orig)
                    or _device_pointer(other.dest) != _device_pointer(value.dest)
                    or (
                        value.time is not None
                        and _device_pointer(other.time) != _device_pointer(value.time)
                    )
                ):
                    raise ValueError("grouped skim bindings do not share row coordinates")
            arguments.extend((value.orig, value.dest))
            if representative.skim_rank == 3:
                arguments.append(value.time)
            arguments.append(np.int64(value.dest_count))
            if representative.skim_rank == 3:
                arguments.append(np.int64(value.time_count))
        return tuple(arguments)
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


def _skim_groups(bindings):
    representatives = {}
    for binding in bindings:
        if binding.storage_kind == "skim":
            representatives.setdefault(binding.skim_group, binding)
    return sorted(representatives.items())


def _skim_group_count(bindings):
    return len(_skim_groups(bindings))


def _device_pointer(value):
    interface = value.__cuda_array_interface__
    return int(interface["data"][0])


def _skim_position_bytes(bindings, values, *, grouped):
    selected = (
        [representative for _, representative in _skim_groups(bindings)]
        if grouped else [b for b in bindings if b.storage_kind == "skim"]
    )
    return sum(
        values[binding.source].orig.nbytes
        + values[binding.source].dest.nbytes
        + (
            values[binding.source].time.nbytes
            if values[binding.source].time is not None else 0
        )
        for binding in selected
    )


def _shared_memory_bytes(
    terms, skims, skim_groups, tile_rows, locality_optimized,
    grouped_skim_indices, sparse_zero_coefficients=False,
):
    if locality_optimized:
        return tile_rows * (terms + skims) * np.dtype(np.float32).itemsize
    feature_bytes = terms * np.dtype(np.float32).itemsize
    if not grouped_skim_indices and not sparse_zero_coefficients:
        return feature_bytes
    aligned = ((feature_bytes + 7) // 8) * 8
    return (
        aligned
        + (skim_groups * np.dtype(np.int64).itemsize if grouped_skim_indices else 0)
        + (np.dtype(np.int32).itemsize if sparse_zero_coefficients else 0)
    )


def _emit_node(tree, refs, types, *, numeric_type="double"):
    op = tree["op"]
    if op == "const":
        value = tree["value"]
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return f"{value}LL", "int"
        if isinstance(value, float):
            return _float_literal(value, numeric_type=numeric_type), "float"
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
        value, kind = _emit_node(
            tree["arg"], refs, types, numeric_type=numeric_type
        )
        if op == "neg":
            return f"(-(({numeric_type})({value})))", "float"
        if op == "pos":
            return f"(+(({numeric_type})({value})))", "float"
        if kind == "bool":
            return f"(!((bool)({value})))", "bool"
        if kind == "int":
            return f"(~((long long)({value})))", "int"
        raise ValueError("strict CUDA bitwise invert requires Boolean or integer input")
    if op in {"add", "sub", "mul", "div"}:
        left, _ = _emit_node(
            tree["left"], refs, types, numeric_type=numeric_type
        )
        right, _ = _emit_node(
            tree["right"], refs, types, numeric_type=numeric_type
        )
        token = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        return (
            f"((({numeric_type})({left})) {token} "
            f"(({numeric_type})({right})))", "float"
        )
    if op in {"and", "or"}:
        children = tree.get("args") or [tree["left"], tree["right"]]
        emitted = [
            _emit_node(child, refs, types, numeric_type=numeric_type)
            for child in children
        ]
        token = "&" if op == "and" else "|"
        kind = "bool" if all(child_kind == "bool" for _, child_kind in emitted) else "int"
        cast = "bool" if kind == "bool" else "long long"
        code = f" {token} ".join(f"(({cast})({value}))" for value, _ in emitted)
        return f"({code})", kind
    if op == "compare":
        left, _ = _emit_node(
            tree["left"], refs, types, numeric_type=numeric_type
        )
        pieces = []
        for operator, right_tree in zip(tree["operators"], tree["rights"]):
            right, _ = _emit_node(
                right_tree, refs, types, numeric_type=numeric_type
            )
            token = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[operator]
            pieces.append(
                f"((({numeric_type})({left})) {token} "
                f"(({numeric_type})({right})))"
            )
            left = right
        return "(" + " && ".join(pieces) + ")", "bool"
    if op == "clip":
        value, _ = _emit_node(
            tree["value"], refs, types, numeric_type=numeric_type
        )
        code = f"(({numeric_type})({value}))"
        lower = tree["keywords"].get("lower")
        upper = tree["keywords"].get("upper")
        if tree["args"]:
            lower = tree["args"][0]
        if lower is not None:
            bound, _ = _emit_node(
                lower, refs, types, numeric_type=numeric_type
            )
            code = (
                f"(({code}) < (({numeric_type})({bound})) ? "
                f"(({numeric_type})({bound})) : ({code}))"
            )
        if upper is not None:
            bound, _ = _emit_node(
                upper, refs, types, numeric_type=numeric_type
            )
            code = (
                f"(({code}) > (({numeric_type})({bound})) ? "
                f"(({numeric_type})({bound})) : ({code}))"
            )
        return code, "float"
    if op == "maximum":
        emitted = [
            _emit_node(child, refs, types, numeric_type=numeric_type)[0]
            for child in tree["args"]
        ]
        code = f"(({numeric_type})({emitted[0]}))"
        nan_literal = _float_literal(float("nan"), numeric_type=numeric_type)
        for value in emitted[1:]:
            other = f"(({numeric_type})({value}))"
            code = f"(((({code}) != ({code})) || (({other}) != ({other}))) ? {nan_literal} : (({code}) > ({other}) ? ({code}) : ({other})))"
        return code, "float"
    raise ValueError(f"strict CUDA emission for {op!r} is not implemented")


def _float_literal(value, *, numeric_type="double"):
    if numeric_type == "float":
        if math.isnan(value):
            return "__int_as_float(0x7fc00000)"
        if math.isinf(value):
            literal = "__int_as_float(0x7f800000)"
            return literal if value > 0 else f"(-{literal})"
        literal = format(float(np.float32(value)), ".9g")
        if not any(marker in literal for marker in (".", "e", "E")):
            literal += ".0"
        return f"{literal}f"
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
