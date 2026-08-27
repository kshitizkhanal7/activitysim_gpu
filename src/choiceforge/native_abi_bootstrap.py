"""Native strict-CUDA ABI bootstrap from reviewed IR and raw skim metadata.

This module is the Phase 30 production boundary.  It never accepts a dense
preprocessor dataframe or a captured strict invocation.  Instead it derives
the complete kernel ABI from the hashed utility IR, the named Phase 29 source
typing contract, scalar model constants, and immutable raw skim cubes.
Unknown or mistyped sources fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np

from .cuda_backend import _cupy
from .raw_table_input_generation import (
    RAW_FLOAT_SOURCES,
    RAW_INT_SOURCES,
    RAW_SOURCE_ALIASES,
)
from .sharrow_cuda import (
    InputBinding,
    ResidentStrictCudaInvocation,
    _binding_schema,
    _host_coefficients,
    _node_sources,
    _shared_memory_bytes,
    _validate_document,
    _value_kind,
    generate_cuda_source,
)


_NATIVE_KERNEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class NativeSkimCube:
    """One immutable device cube with its declared logical dimensions."""

    data: Any
    dest_count: int
    time_count: int
    rank: int


@dataclass(frozen=True)
class NativeStrictAbiPlan:
    """A strict invocation skeleton plus its auditable data-only ABI."""

    invocation: ResidentStrictCudaInvocation
    bindings: tuple[InputBinding, ...]
    manifest: Mapping[str, Any]


def _source_label(source) -> str:
    return ":".join(str(part) for part in source)


def _ordered_sources(document):
    seen = set()
    result = []
    for term in document["terms"]:
        for source in _node_sources(term["tree"]):
            if source not in seen:
                seen.add(source)
                result.append(tuple(source))
    return tuple(result)


def _scalar_value(source, environment):
    if source[0] != "name" or len(source) != 2:
        raise ValueError(
            f"native ABI source {_source_label(source)!r} is neither declared "
            "row state nor a named scalar"
        )
    name = source[1]
    if name not in environment:
        raise ValueError(f"native ABI scalar {name!r} is absent")
    value = environment[name]
    shape = getattr(value, "shape", None)
    if (shape is not None and tuple(shape) != ()) or (
        shape is None and not np.isscalar(value)
    ):
        raise ValueError(f"native ABI scalar {name!r} is not scalar")
    return value


def _compile_bindings(document, scalar_environment, cube_loader):
    float_slot = int_slot = scalar_float_slot = scalar_int_slot = skim_slot = 0
    skim_groups = {}
    float_slots = {}
    int_slots = {}
    cubes = {}
    scalar_values = {}
    bindings = []
    for source in _ordered_sources(document):
        label = _source_label(source)
        if source[0] == "skim":
            cube = cube_loader(source)
            if not isinstance(cube, NativeSkimCube):
                raise TypeError(f"native skim loader returned no cube for {label!r}")
            direction = source[1]
            expected_rank = 3 if direction in {
                "odt_skims", "dot_skims", "odr_skims", "dor_skims"
            } else 2 if direction in {"od_skims", "od_skims_reverse"} else None
            if expected_rank is None or cube.rank != expected_rank:
                raise ValueError(
                    f"native skim {label!r} rank {cube.rank} violates direction contract"
                )
            group = skim_groups.setdefault(direction, len(skim_groups))
            binding = InputBinding(source, "float", "skim", skim_slot, cube.rank, group)
            cubes[source] = cube
            skim_slot += 1
        elif label in RAW_FLOAT_SOURCES:
            canonical = RAW_SOURCE_ALIASES.get(label, label)
            slot = float_slots.get(canonical)
            if slot is None:
                slot = float_slot
                float_slots[canonical] = slot
                float_slot += 1
            binding = InputBinding(source, "float", "float64", slot)
        elif label in RAW_INT_SOURCES:
            canonical = RAW_SOURCE_ALIASES.get(label, label)
            slot = int_slots.get(canonical)
            if slot is None:
                slot = int_slot
                int_slots[canonical] = slot
                int_slot += 1
            value_kind = "bool" if label not in {
                "name:auto_ownership", "name:age", "name:number_of_participants",
                "column:auto_ownership", "column:age", "column:hhsize",
                "column:dest_topology", "column:num_workers",
            } else "int"
            binding = InputBinding(source, value_kind, "int64", slot)
        else:
            value = _scalar_value(source, scalar_environment)
            kind = _value_kind(value)
            if kind == "float":
                binding = InputBinding(
                    source, kind, "scalar_float64", scalar_float_slot
                )
                scalar_float_slot += 1
            elif kind in {"int", "bool"}:
                binding = InputBinding(
                    source, kind, "scalar_int64", scalar_int_slot
                )
                scalar_int_slot += 1
            else:
                raise ValueError(
                    f"native scalar {label!r} has unsupported kind {kind!r}"
                )
            scalar_values[source] = value
        bindings.append(binding)
    return tuple(bindings), cubes, scalar_values


def _scalar_arrays(cp, bindings, scalar_values):
    float_bindings = sorted(
        (item for item in bindings if item.storage_kind == "scalar_float64"),
        key=lambda item: item.slot,
    )
    int_bindings = sorted(
        (item for item in bindings if item.storage_kind == "scalar_int64"),
        key=lambda item: item.slot,
    )
    return (
        cp.asarray(
            [scalar_values[item.source] for item in float_bindings], dtype=cp.float32
        ),
        cp.asarray(
            [scalar_values[item.source] for item in int_bindings], dtype=cp.int64
        ),
    )


def _skim_arguments(cp, bindings, cubes, rows):
    skim_bindings = [item for item in bindings if item.storage_kind == "skim"]
    arguments = [cubes[item.source].data for item in skim_bindings]
    representatives = {}
    for item in skim_bindings:
        representatives.setdefault(item.skim_group, item)
    coordinate_bytes = 0
    for group in sorted(representatives):
        item = representatives[group]
        cube = cubes[item.source]
        origin = cp.empty(rows, dtype=cp.int64)
        destination = cp.empty(rows, dtype=cp.int64)
        arguments.extend((origin, destination))
        coordinate_bytes += int(origin.nbytes + destination.nbytes)
        if item.skim_rank == 3:
            period = cp.empty(rows, dtype=cp.int64)
            arguments.append(period)
            coordinate_bytes += int(period.nbytes)
        arguments.append(np.int64(cube.dest_count))
        if item.skim_rank == 3:
            arguments.append(np.int64(cube.time_count))
    return tuple(arguments), coordinate_bytes


def compile_native_strict_abi(
    document: Mapping[str, Any],
    scalar_environment: Mapping[str, Any],
    cube_loader: Callable[[tuple[str, ...]], NativeSkimCube],
    *,
    rows: int,
) -> NativeStrictAbiPlan:
    """Compile a strict resident invocation without dense preprocessor values."""
    _validate_document(document)
    rows = int(rows)
    if rows <= 0:
        raise ValueError("native strict ABI requires at least one row")
    cp = _cupy()
    bindings, cubes, scalar_values = _compile_bindings(
        document, scalar_environment, cube_loader
    )
    coefficients_host = _host_coefficients(document, {})
    source, source_sha256 = generate_cuda_source(
        document,
        bindings,
        capture_features=False,
        locality_tile_rows=1,
        locality_optimized=False,
        group_skim_indices=True,
        coefficient_values=coefficients_host,
        sparse_zero_coefficients=False,
        expression_float32=True,
        fused_utility_accumulation=True,
    )
    kernel_key = f"{document['sha256']}:{source_sha256}"
    kernel = _NATIVE_KERNEL_CACHE.get(kernel_key)
    compiled = kernel is None
    if kernel is None:
        kernel = cp.RawKernel(
            source,
            "choiceforge_strict_ir_v3",
            options=("--std=c++11", "--fmad=true", "--prec-div=true", "--ftz=true"),
        )
        kernel.compile()
        _NATIVE_KERNEL_CACHE[kernel_key] = kernel
    coefficients = cp.asarray(coefficients_host)
    float_scalars, int_scalars = _scalar_arrays(cp, bindings, scalar_values)
    skim_arguments, coordinate_bytes = _skim_arguments(cp, bindings, cubes, rows)
    def unique_sources(storage):
        by_slot = {}
        for item in bindings:
            if item.storage_kind == storage:
                by_slot.setdefault(item.slot, item.source)
        return tuple(by_slot[slot] for slot in sorted(by_slot))

    float_sources = unique_sources("float64")
    int_sources = unique_sources("int64")
    skim_bindings = tuple(item for item in bindings if item.storage_kind == "skim")
    grid = (rows,)
    block = (256,)
    shared_mem = _shared_memory_bytes(
        len(document["terms"]), len(skim_bindings),
        len({item.skim_group for item in skim_bindings}), 1, False, True,
    )
    float_inputs = cp.empty((rows, len(float_sources)), dtype=cp.float32)
    int_inputs = cp.empty((rows, len(int_sources)), dtype=cp.int64)
    features = cp.empty((1,), dtype=cp.float32)
    utilities = cp.empty((rows, len(document["alternatives"])), dtype=cp.float32)
    unique_cubes = {
        int(cube.data.__cuda_array_interface__["data"][0]): cube.data
        for cube in cubes.values()
    }
    invocation = ResidentStrictCudaInvocation(
        kernel=kernel,
        float_inputs=float_inputs,
        int_inputs=int_inputs,
        float_scalars=float_scalars,
        int_scalars=int_scalars,
        coefficients=coefficients,
        features=features,
        utilities=utilities,
        skim_arguments=skim_arguments,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        rows=rows,
        terms=len(document["terms"]),
        alternatives=len(document["alternatives"]),
        dense_input_bytes=int(float_inputs.nbytes + int_inputs.nbytes),
        skim_coordinate_bytes=coordinate_bytes,
        logical_skim_bindings=len(skim_bindings),
        unique_skim_arrays=len(unique_cubes),
        shared_skim_data_bytes=sum(int(item.nbytes) for item in unique_cubes.values()),
        float_input_sources=float_sources,
        int_input_sources=int_sources,
        skim_input_sources=tuple(item.source for item in skim_bindings),
        skim_input_ranks=tuple(item.skim_rank for item in skim_bindings),
        skim_input_groups=tuple(item.skim_group for item in skim_bindings),
    )
    binding_document = _binding_schema(bindings)
    schema_payload = {
        "contract": "hashed_ir_plus_named_raw_sources_no_dense_preprocessor",
        "ir_sha256": document["sha256"],
        "terms": len(document["terms"]),
        "alternatives": list(document["alternatives"]),
        "bindings": binding_document,
        "codegen": {
            "expression_dtype": "float32",
            "feature_storage_dtype": "float32",
            "utility_accumulation": "sharrow_fused_float32",
            "grouped_skim_indices": True,
            "capture_features": False,
        },
    }
    schema_sha256 = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        **schema_payload,
        "schema_sha256": schema_sha256,
        "generated_source_sha256": source_sha256,
        "compiled_this_call": compiled,
        "dense_preprocessor_rows_read": 0,
        "dense_preprocessor_values_read": 0,
        "float_row_sources": len(float_sources),
        "int_row_sources": len(int_sources),
        "scalar_sources": len(float_scalars) + len(int_scalars),
        "skim_sources": len(skim_bindings),
        "skim_coordinate_groups": len({item.skim_group for item in skim_bindings}),
    }
    return NativeStrictAbiPlan(invocation, bindings, manifest)


def clear_native_abi_cache() -> None:
    _NATIVE_KERNEL_CACHE.clear()
