"""Rebuild strict-CUDA row inputs from compact chooser and slot state.

Phase 27 removes captured row-dense chooser leaves and skim coordinates from
the sealed execution graph.  Qualification factors every value bit-for-bit as
one constant, one value per chooser, or one value per observed exact start/end
time slot. A CUDA replay first expands CSR row ownership and then reconstructs the exact
row arrays without touching the host or the original captured arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping

import numpy as np

from .cuda_backend import _cupy


_FILL_OWNER_KERNEL = None
_EXPAND_BITS_KERNEL = None


def _kernels(cp):
    global _FILL_OWNER_KERNEL, _EXPAND_BITS_KERNEL
    if _FILL_OWNER_KERNEL is None:
        _FILL_OWNER_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__ void fill_csr_owner(
                const long long* offsets, int* owners, long long chooser_count
            ) {
                long long chooser = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (chooser >= chooser_count) return;
                for (long long row = offsets[chooser]; row < offsets[chooser + 1]; ++row)
                    owners[row] = (int)chooser;
            }
            """,
            "fill_csr_owner",
        )
        _EXPAND_BITS_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__ void expand_compact_bits(
                unsigned char* output,
                const signed char* kind,
                const int* position,
                const int* owner,
                const long long* offsets,
                const int* slot,
                const unsigned char* constants,
                const unsigned char* owner_values,
                const unsigned char* slot_values,
                const int* pattern_ids,
                const long long* pattern_offsets,
                const unsigned char* pattern_values,
                long long rows,
                int columns,
                int owner_columns,
                int slot_columns,
                int pattern_columns,
                int slot_count,
                int pattern_width,
                int item_size
            ) {
                long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                long long total = rows * (long long)columns;
                if (element >= total) return;
                long long row = element / columns;
                int column = (int)(element - row * columns);
                int compact_column = position[column];
                const unsigned char* source;
                if (kind[column] == 0) {
                    source = constants + (long long)compact_column * item_size;
                } else if (kind[column] == 1) {
                    source = owner_values
                        + ((long long)owner[row] * owner_columns + compact_column) * item_size;
                } else if (kind[column] == 2) {
                    source = slot_values
                        + ((long long)slot[row] * slot_columns + compact_column) * item_size;
                } else if (kind[column] == 3) {
                    int pattern = pattern_ids[
                        (long long)owner[row] * pattern_columns + compact_column
                    ];
                    long long value = pattern_offsets[compact_column]
                        + (long long)pattern * pattern_width
                        + (row - offsets[owner[row]]);
                    source = pattern_values + value * item_size;
                } else {
                    unsigned char* target = output + element * item_size;
                    for (int byte = 0; byte < item_size; ++byte) target[byte] = 0;
                    return;
                }
                unsigned char* target = output + element * item_size;
                if (item_size == 8) {
                    *((unsigned long long*)target) = *((const unsigned long long*)source);
                } else if (item_size == 4) {
                    *((unsigned int*)target) = *((const unsigned int*)source);
                } else if (item_size == 2) {
                    *((unsigned short*)target) = *((const unsigned short*)source);
                } else if (item_size == 1) {
                    *target = *source;
                } else {
                    for (int byte = 0; byte < item_size; ++byte) target[byte] = source[byte];
                }
            }
            """,
            "expand_compact_bits",
        )
    return _FILL_OWNER_KERNEL, _EXPAND_BITS_KERNEL


@dataclass(frozen=True)
class CompactArrayFactor:
    """A bit-exact constant/chooser/slot factorization of one row array."""

    target: Any
    original_shape: tuple[int, ...]
    kind: Any
    position: Any
    constants: Any
    owner_values: Any
    slot_values: Any
    pattern_ids: Any
    pattern_offsets: Any
    pattern_values: Any
    constant_columns: int
    owner_columns: int
    slot_columns: int
    pattern_columns: int
    slot_count: int
    pattern_width: int
    column_labels: tuple[str, ...]
    semantic_generated_columns: int

    @property
    def compact_bytes(self):
        return int(
            self.kind.nbytes
            + self.position.nbytes
            + self.constants.nbytes
            + self.owner_values.nbytes
            + self.slot_values.nbytes
            + self.pattern_ids.nbytes
            + self.pattern_offsets.nbytes
            + self.pattern_values.nbytes
        )

    @property
    def target_bytes(self):
        return int(self.target.nbytes)

    def execute(self, owners, offsets, slots):
        cp = _cupy()
        _, expand = _kernels(cp)
        rows = int(self.original_shape[0])
        columns = (
            int(np.prod(self.original_shape[1:]))
            if len(self.original_shape) > 1 else 1
        )
        total = rows * columns
        if total:
            expand(
                ((total + 255) // 256,),
                (256,),
                (
                    self.target,
                    self.kind,
                    self.position,
                    owners,
                    offsets,
                    slots,
                    self.constants,
                    self.owner_values,
                    self.slot_values,
                    self.pattern_ids,
                    self.pattern_offsets,
                    self.pattern_values,
                    np.int64(rows),
                    np.int32(columns),
                    np.int32(self.owner_columns),
                    np.int32(self.slot_columns),
                    np.int32(self.pattern_columns),
                    np.int32(self.slot_count),
                    np.int32(self.pattern_width),
                    np.int32(self.target.dtype.itemsize),
                ),
            )
        return self.target

    def cpu_snapshot(self):
        cp = _cupy()
        return {
            "shape": self.original_shape,
            "dtype": self.target.dtype,
            "kind": cp.asnumpy(self.kind),
            "position": cp.asnumpy(self.position),
            "constants": cp.asnumpy(self.constants),
            "owner_values": cp.asnumpy(self.owner_values),
            "slot_values": cp.asnumpy(self.slot_values),
            "pattern_ids": cp.asnumpy(self.pattern_ids),
            "pattern_offsets": cp.asnumpy(self.pattern_offsets),
            "pattern_values": cp.asnumpy(self.pattern_values),
            "slot_count": self.slot_count,
            "pattern_width": self.pattern_width,
        }


def _factor_array(
    source, owner_index, slot_index, owner_starts, *, label, column_labels=None
):
    cp = _cupy()
    source = cp.asarray(source)
    if source.ndim < 1 or int(source.shape[0]) != int(owner_index.size):
        raise ValueError(f"{label} is not a row array")
    rows = int(source.shape[0])
    columns = int(np.prod(source.shape[1:])) if source.ndim > 1 else 1
    matrix = source.reshape(rows, columns)
    kinds = []
    positions = []
    constant_columns = []
    owner_columns = []
    slot_columns = []
    pattern_columns = []
    pattern_id_columns = []
    pattern_value_blocks = []
    pattern_offsets = []
    owner_device = cp.asarray(owner_index, dtype=cp.int32)
    slot_device = cp.asarray(slot_index, dtype=cp.int32)
    owner_starts_device = cp.asarray(owner_starts, dtype=cp.int64)

    slot_count = int(np.max(slot_index)) + 1
    first_for_slot = np.full(slot_count, -1, dtype=np.int64)
    for row, slot in enumerate(slot_index):
        if first_for_slot[int(slot)] < 0:
            first_for_slot[int(slot)] = row
    if np.any(first_for_slot < 0):
        raise ValueError(f"{label} received a non-contiguous compact slot index")
    first_for_slot_device = cp.asarray(first_for_slot, dtype=cp.int64)

    def bytes_of(values):
        return cp.ascontiguousarray(values).view(cp.uint8).reshape(rows, -1)

    for column in range(columns):
        values = matrix[:, column]
        raw = bytes_of(values)
        if bool(cp.all(raw == raw[0]).item()):
            positions.append(len(constant_columns))
            kinds.append(0)
            constant_columns.append(column)
            continue
        owner_sample = values[owner_starts_device]
        owner_candidate = owner_sample[owner_device]
        if bool(cp.all(raw == bytes_of(owner_candidate)).item()):
            positions.append(len(owner_columns))
            kinds.append(1)
            owner_columns.append(column)
            continue
        slot_sample = values[first_for_slot_device]
        slot_candidate = slot_sample[slot_device]
        if bool(cp.all(raw == bytes_of(slot_candidate)).item()):
            positions.append(len(slot_columns))
            kinds.append(2)
            slot_columns.append(column)
            continue
        # Some preprocessor leaves are genuine chooser-by-alternative
        # functions (MTC daily parking cost is rate times duration). Preserve
        # their exact preprocessor semantics with a dictionary of unique slot
        # response patterns plus one compact pattern id per chooser.
        host_values = cp.asnumpy(values)
        owner_count = int(owner_starts.size)
        offsets_host = np.r_[owner_starts, rows]
        pattern_width = int(np.max(np.diff(offsets_host)))
        pattern = np.zeros((owner_count, pattern_width), dtype=source.dtype)
        for owner in range(owner_count):
            begin, finish = offsets_host[owner : owner + 2]
            pattern[owner, : finish - begin] = host_values[begin:finish]
        raw_pattern = (
            np.ascontiguousarray(pattern).view(np.uint8).reshape(owner_count, -1)
        )
        _, unique_indices, inverse = np.unique(
            raw_pattern, axis=0, return_index=True, return_inverse=True
        )
        unique_patterns = pattern[unique_indices]
        encoded_bytes = int(inverse.nbytes + unique_patterns.nbytes)
        if encoded_bytes < int(values.nbytes):
            positions.append(len(pattern_columns))
            kinds.append(3)
            pattern_columns.append(column)
            pattern_id_columns.append(inverse.astype(np.int32))
            pattern_offsets.append(
                sum(block.size for block in pattern_value_blocks)
            )
            pattern_value_blocks.append(unique_patterns.reshape(-1))
            continue
        semantic = (
            f" ({column_labels[column]!r})"
            if column_labels is not None and column < len(column_labels)
            else ""
        )
        raise ValueError(
            f"{label} column {column}{semantic} is neither constant, "
            "chooser-factored, nor slot-factored"
        )

    dtype = source.dtype
    owner_count = int(owner_starts.size)
    constants = (
        matrix[0, constant_columns].copy()
        if constant_columns else cp.empty((0,), dtype=dtype)
    )
    owner_values = (
        matrix[owner_starts_device][:, owner_columns].copy()
        if owner_columns else cp.empty((owner_count, 0), dtype=dtype)
    )
    slot_values = (
        matrix[first_for_slot_device][:, slot_columns].copy()
        if slot_columns else cp.empty((slot_count, 0), dtype=dtype)
    )
    pattern_ids = (
        cp.asarray(np.column_stack(pattern_id_columns), dtype=cp.int32)
        if pattern_id_columns else cp.empty((owner_count, 0), dtype=cp.int32)
    )
    pattern_values = (
        cp.asarray(np.concatenate(pattern_value_blocks), dtype=dtype)
        if pattern_value_blocks else cp.empty((0,), dtype=dtype)
    )
    return CompactArrayFactor(
        target=cp.empty_like(source),
        original_shape=tuple(int(x) for x in source.shape),
        kind=cp.asarray(kinds, dtype=cp.int8),
        position=cp.asarray(positions, dtype=cp.int32),
        constants=constants,
        owner_values=owner_values,
        slot_values=slot_values,
        pattern_ids=pattern_ids,
        pattern_offsets=cp.asarray(pattern_offsets, dtype=cp.int64),
        pattern_values=pattern_values,
        constant_columns=len(constant_columns),
        owner_columns=len(owner_columns),
        slot_columns=len(slot_columns),
        pattern_columns=len(pattern_columns),
        slot_count=slot_count,
        pattern_width=int(np.max(np.diff(np.r_[owner_starts, rows]))),
        column_labels=tuple(
            _format_source_label(item)
            for item in (
                column_labels
                if column_labels is not None
                else (f"column_{index}" for index in range(columns))
            )
        ),
        semantic_generated_columns=0,
    )


@dataclass(frozen=True)
class ResidentInputExpansionPlan:
    """Sealed expansion state plus an invocation wired only to rebuilt arrays."""

    invocation: Any
    offsets: Any
    slots: Any
    owners: Any
    factors: tuple[CompactArrayFactor, ...]
    labels: tuple[str, ...]
    original_dense_bytes: int
    original_coordinate_bytes: int
    semantic_program: Any | None = None

    @classmethod
    def compile(cls, invocation, metadata: Mapping[str, Any]):
        return cls._compile(invocation, metadata, semantic=False)

    @classmethod
    def _compile(cls, invocation, metadata: Mapping[str, Any], *, semantic):
        cp = _cupy()
        chooser_ids = np.asarray(metadata["chooser_ids"], dtype=np.int64)
        rows = int(invocation.rows)
        if chooser_ids.ndim != 1 or chooser_ids.size != rows or rows == 0:
            raise ValueError("Phase 27 requires one nonempty chooser id per utility row")
        first = np.r_[True, chooser_ids[1:] != chooser_ids[:-1]]
        owner_starts = np.flatnonzero(first).astype(np.int64)
        owner_index = np.cumsum(first, dtype=np.int32) - 1
        if not np.array_equal(chooser_ids[owner_starts][owner_index], chooser_ids):
            raise ValueError("chooser rows are not contiguous")
        offsets = np.r_[owner_starts, rows].astype(np.int64)
        start = np.asarray(metadata["start"])
        end = np.asarray(metadata["end"])
        if start.shape != chooser_ids.shape or end.shape != chooser_ids.shape:
            raise ValueError("time metadata does not match utility rows")
        # Keep the representation independent of a model's clock encoding
        # (hour labels, half-hour labels, or arbitrary alternative ids).
        _, slots = np.unique(
            np.column_stack((start, end)), axis=0, return_inverse=True
        )
        slots = slots.astype(np.int32)

        factors = []
        labels = []
        float_factor = _factor_array(
            invocation.float_inputs,
            owner_index,
            slots,
            owner_starts,
            label="float_inputs",
            column_labels=getattr(invocation, "float_input_sources", None),
        )
        int_factor = _factor_array(
            invocation.int_inputs,
            owner_index,
            slots,
            owner_starts,
            label="int_inputs",
            column_labels=getattr(invocation, "int_input_sources", None),
        )
        factors.extend((float_factor, int_factor))
        labels.extend(("float_inputs", "int_inputs"))

        rebuilt_args = list(invocation.skim_arguments)
        coordinate_bytes = 0
        for position in range(invocation.logical_skim_bindings, len(rebuilt_args)):
            value = rebuilt_args[position]
            if not hasattr(value, "__cuda_array_interface__"):
                continue
            if value.ndim < 1 or int(value.shape[0]) != rows:
                continue
            label = f"skim_coordinate_{position}"
            factor = _factor_array(value, owner_index, slots, owner_starts, label=label)
            rebuilt_args[position] = factor.target
            coordinate_bytes += int(value.nbytes)
            factors.append(factor)
            labels.append(label)

        if coordinate_bytes != int(invocation.skim_coordinate_bytes):
            raise ValueError(
                "Phase 27 did not identify every captured skim coordinate array: "
                f"found {coordinate_bytes}, expected {invocation.skim_coordinate_bytes}"
            )
        semantic_program = None
        if semantic:
            from .semantic_input_generation import compile_semantic_input_program

            semantic_program, float_factor, int_factor = compile_semantic_input_program(
                invocation=invocation,
                rebuilt_skim_arguments=tuple(rebuilt_args),
                metadata=metadata,
                owner_index=owner_index,
                owner_starts=owner_starts,
                slots=slots,
                float_factor=float_factor,
                int_factor=int_factor,
            )
            factors[0] = float_factor
            factors[1] = int_factor

        rebuilt = replace(
            invocation,
            float_inputs=float_factor.target,
            int_inputs=int_factor.target,
            skim_arguments=tuple(rebuilt_args),
        )
        plan = cls(
            invocation=rebuilt,
            offsets=cp.asarray(offsets),
            slots=cp.asarray(slots),
            owners=cp.empty(rows, dtype=cp.int32),
            factors=tuple(factors),
            labels=tuple(labels),
            original_dense_bytes=int(
                invocation.float_inputs.nbytes + invocation.int_inputs.nbytes
            ),
            original_coordinate_bytes=int(invocation.skim_coordinate_bytes),
            semantic_program=semantic_program,
        )
        plan.execute()
        cp.cuda.Stream.null.synchronize()
        for label, factor, original in (
            [("float_inputs", float_factor, invocation.float_inputs),
             ("int_inputs", int_factor, invocation.int_inputs)]
            + [
                (labels[index], factors[index], invocation.skim_arguments[position])
                for index, position in _coordinate_factor_positions(invocation)
            ]
        ):
            if not bool(
                cp.array_equal(
                    cp.ascontiguousarray(factor.target).view(cp.uint8),
                    cp.ascontiguousarray(original).view(cp.uint8),
                )
            ):
                raise ValueError(f"Phase 27 reconstructed {label} incorrectly")
        return plan

    @property
    def chooser_count(self):
        return int(self.offsets.size - 1)

    @property
    def compact_bytes(self):
        return int(
            self.offsets.nbytes
            + self.slots.nbytes
            + sum(x.compact_bytes for x in self.factors)
            + (
                self.semantic_program.compact_bytes
                if self.semantic_program is not None else 0
            )
        )

    @property
    def workspace_bytes(self):
        return int(self.owners.nbytes + sum(x.target_bytes for x in self.factors))

    def execute(self):
        cp = _cupy()
        fill_owner, _ = _kernels(cp)
        if self.chooser_count:
            fill_owner(
                ((self.chooser_count + 255) // 256,),
                (256,),
                (self.offsets, self.owners, np.int64(self.chooser_count)),
            )
        for factor in self.factors:
            factor.execute(self.owners, self.offsets, self.slots)
        if self.semantic_program is not None:
            self.semantic_program.execute(
                self.invocation.float_inputs,
                self.invocation.int_inputs,
                self.owners,
                self.slots,
            )
        return self.invocation

    def classification(self):
        return {
            label: {
                "constant_columns": factor.constant_columns,
                "chooser_columns": factor.owner_columns,
                "slot_columns": factor.slot_columns,
                "chooser_slot_pattern_columns": factor.pattern_columns,
                "semantic_generated_columns": factor.semantic_generated_columns,
                "target_bytes": factor.target_bytes,
                "compact_bytes": factor.compact_bytes,
                "columns": [
                    {
                        "column": index,
                        "source": factor.column_labels[index],
                        "factor": (
                            "constant" if int(kind) == 0 else
                            "chooser" if int(kind) == 1 else
                            "slot" if int(kind) == 2 else
                            "chooser_slot_pattern" if int(kind) == 3 else
                            "semantic_generated"
                        ),
                    }
                    for index, kind in enumerate(_cupy().asnumpy(factor.kind))
                ],
            }
            for label, factor in zip(self.labels, self.factors)
        } | ({"semantic_program": self.semantic_program.manifest()}
             if self.semantic_program is not None else {})

    def cpu_benchmark(self, runs=5):
        offsets = _cupy().asnumpy(self.offsets)
        slots = _cupy().asnumpy(self.slots)
        snapshots = [factor.cpu_snapshot() for factor in self.factors]

        def once(compare=False):
            started = time.perf_counter()
            owners = np.repeat(
                np.arange(offsets.size - 1, dtype=np.int32), np.diff(offsets)
            )
            local = np.arange(owners.size, dtype=np.int64) - offsets[owners]
            outputs = []
            for item in snapshots:
                columns = (
                    int(np.prod(item["shape"][1:]))
                    if len(item["shape"]) > 1 else 1
                )
                output = np.empty((owners.size, columns), dtype=item["dtype"])
                for column, kind in enumerate(item["kind"]):
                    position = item["position"][column]
                    if kind == 0:
                        output[:, column] = item["constants"][position]
                    elif kind == 1:
                        output[:, column] = item["owner_values"][owners, position]
                    elif kind == 2:
                        output[:, column] = item["slot_values"][slots, position]
                    else:
                        patterns = item["pattern_ids"][owners, position]
                        indices = (
                            item["pattern_offsets"][position]
                            + patterns * item["pattern_width"]
                            + local
                        )
                        output[:, column] = item["pattern_values"][indices]
                outputs.append(output.reshape(item["shape"]))
            seconds = time.perf_counter() - started
            if compare:
                cp = _cupy()
                for output, factor in zip(outputs, self.factors):
                    expected = cp.asnumpy(factor.target)
                    if not np.array_equal(
                        np.ascontiguousarray(output).view(np.uint8),
                        np.ascontiguousarray(expected).view(np.uint8),
                    ):
                        raise ValueError("CPU and CUDA compact expansion differ")
            return seconds

        once(compare=True)
        return [once(compare=False) for _ in range(max(1, int(runs)))]


def _coordinate_factor_positions(invocation):
    """Map factor-list positions to row-coordinate argument positions."""
    factor_index = 2
    result = []
    for position in range(
        invocation.logical_skim_bindings, len(invocation.skim_arguments)
    ):
        value = invocation.skim_arguments[position]
        if (
            hasattr(value, "__cuda_array_interface__")
            and value.ndim >= 1
            and int(value.shape[0]) == int(invocation.rows)
        ):
            result.append((factor_index, position))
            factor_index += 1
    return result


def _format_source_label(source):
    if isinstance(source, (tuple, list)):
        return ":".join(str(part) for part in source)
    return str(source)


class ResidentSemanticInputPlan(ResidentInputExpansionPlan):
    """Phase 28 plan that replaces response dictionaries with named formulas."""

    @classmethod
    def compile(cls, invocation, metadata: Mapping[str, Any]):
        return cls._compile(invocation, metadata, semantic=True)
