"""Named CUDA generators for Phase 28 mandatory-tour input semantics.

Phase 27 accepted anonymous, deduplicated chooser-response dictionaries.
This module accepts only the public MTC mode-choice preprocessing expressions
listed below, regenerates them from compact chooser/slot state and resident raw
skim cubes, and removes the dictionaries from the replay state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .cuda_backend import _cupy


_PARKING = "column:daily_parking_cost"


def _availability_expression(label, gather, auto_column):
    def g(direction, key):
        return gather(("skim", direction, key))

    def scaled(expression):
        # Public MTC constants.yaml declares TRANSIT_SCALE_FACTOR: 100.
        return f"(({expression}) / 100.0f)"

    walk_base = "true"
    drive_base = f"(int_inputs[row * int_columns + {auto_column}] > 0LL)"
    if label == "name:sov_available":
        return f"(({g('odt_skims', 'SOV_TIME')} > 0.0f) && ({g('dot_skims', 'SOV_TIME')} > 0.0f))"
    if label == "name:sovtoll_available":
        return f"(({g('odt_skims', 'SOVTOLL_VTOLL')} > 0.0f) || ({g('dot_skims', 'SOVTOLL_VTOLL')} > 0.0f))"
    if label == "name:hov2_available":
        return f"(({g('odt_skims', 'HOV2_TIME')} + {g('dot_skims', 'HOV2_TIME')}) > 0.0f)"
    if label == "name:hov2toll_available":
        return f"(({g('odt_skims', 'HOV2TOLL_VTOLL')} + {g('dot_skims', 'HOV2TOLL_VTOLL')}) > 0.0f)"
    if label == "name:hov3_available":
        return f"(({g('odt_skims', 'HOV3_TIME')} > 0.0f) && ({g('dot_skims', 'HOV3_TIME')} > 0.0f))"
    if label == "name:hov3toll_available":
        return f"(({g('odt_skims', 'HOV3TOLL_VTOLL')} + {g('dot_skims', 'HOV3TOLL_VTOLL')}) > 0.0f)"

    walk = {
        "name:walk_local_available": ("LOC", False),
        "name:walk_commuter_available": ("COM", True),
        "name:walk_express_available": ("EXP", True),
        "name:walk_heavyrail_available": ("HVY", True),
        "name:walk_lrf_available": ("LRF", True),
        "column:walk_ferry_available": ("LRF", True),
    }
    if label in walk:
        mode, keyed = walk[label]
        conditions = [
            walk_base,
            f"({scaled(g('odt_skims', f'WLK_{mode}_WLK_TOTIVT'))} > 0.0f)",
            f"({scaled(g('dot_skims', f'WLK_{mode}_WLK_TOTIVT'))} > 0.0f)",
        ]
        if keyed:
            conditions.append(
                f"(({scaled(g('odt_skims', f'WLK_{mode}_WLK_KEYIVT'))} + "
                f"{scaled(g('dot_skims', f'WLK_{mode}_WLK_KEYIVT'))}) > 0.0f)"
            )
        if label == "column:walk_ferry_available":
            conditions.append(
                f"(({scaled(g('odt_skims', 'WLK_LRF_WLK_FERRYIVT'))} + "
                f"{scaled(g('dot_skims', 'WLK_LRF_WLK_FERRYIVT'))}) > 0.0f)"
            )
        return "(" + " && ".join(conditions) + ")"

    drive = {
        "name:drive_local_available": ("LOC", False),
        "name:drive_commuter_available": ("COM", True),
        "name:drive_express_available": ("EXP", True),
        "name:drive_heavyrail_available": ("HVY", True),
        "name:drive_lrf_available": ("LRF", True),
        "column:drive_ferry_available": ("LRF", True),
    }
    if label in drive:
        mode, keyed = drive[label]
        conditions = [
            drive_base,
            f"({scaled(g('odt_skims', f'DRV_{mode}_WLK_TOTIVT'))} > 0.0f)",
            f"({scaled(g('dot_skims', f'WLK_{mode}_DRV_TOTIVT'))} > 0.0f)",
        ]
        if keyed:
            conditions.append(
                f"(({scaled(g('odt_skims', f'DRV_{mode}_WLK_KEYIVT'))} + "
                f"{scaled(g('dot_skims', f'WLK_{mode}_DRV_KEYIVT'))}) > 0.0f)"
            )
        if label == "column:drive_ferry_available":
            conditions.append(
                f"(({scaled(g('odt_skims', 'DRV_LRF_WLK_FERRYIVT'))} + "
                f"{scaled(g('dot_skims', 'WLK_LRF_WLK_FERRYIVT'))}) > 0.0f)"
            )
        return "(" + " && ".join(conditions) + ")"
    raise KeyError(label)


_AVAILABILITY_LABELS = {
    "name:sov_available",
    "name:sovtoll_available",
    "name:hov2_available",
    "name:hov2toll_available",
    "name:hov3_available",
    "name:hov3toll_available",
    "name:walk_local_available",
    "name:walk_commuter_available",
    "name:walk_express_available",
    "name:walk_heavyrail_available",
    "name:walk_lrf_available",
    "column:walk_ferry_available",
    "name:drive_local_available",
    "name:drive_commuter_available",
    "name:drive_express_available",
    "name:drive_heavyrail_available",
    "name:drive_lrf_available",
    "column:drive_ferry_available",
}


@dataclass(frozen=True)
class SemanticInputProgram:
    kernel: Any
    kernel_arguments: tuple[Any, ...]
    slot_start: Any
    slot_end: Any
    parking_rates: Any
    rows: int
    float_columns: int
    int_columns: int
    generated_float_columns: tuple[int, ...]
    generated_int_columns: tuple[int, ...]
    expressions: tuple[tuple[str, str], ...]
    removed_dictionary_bytes: int

    @property
    def compact_bytes(self):
        return int(self.slot_start.nbytes + self.slot_end.nbytes + self.parking_rates.nbytes)

    def execute(self, float_inputs, int_inputs, owners, slots):
        if self.rows:
            self.kernel(
                ((self.rows + 255) // 256,),
                (256,),
                (
                    float_inputs,
                    int_inputs,
                    owners,
                    slots,
                    self.slot_start,
                    self.slot_end,
                    self.parking_rates,
                    np.int64(self.rows),
                    np.int32(self.float_columns),
                    np.int32(self.int_columns),
                ) + self.kernel_arguments,
            )

    def manifest(self):
        return {
            "contract": "named_cuda_formulas_from_compact_owner_slot_state_and_raw_skims",
            "generated_float_columns": len(self.generated_float_columns),
            "generated_int_columns": len(self.generated_int_columns),
            "anonymous_response_pattern_columns": 0,
            "removed_response_dictionary_bytes": self.removed_dictionary_bytes,
            "semantic_compact_bytes": self.compact_bytes,
            "expressions": [
                {"source": source, "expression": expression}
                for source, expression in self.expressions
            ],
        }


def compile_semantic_input_program(
    *, invocation, rebuilt_skim_arguments, metadata: Mapping[str, Any],
    owner_index, owner_starts, slots, float_factor, int_factor,
    parking_rates_override=None,
):
    """Compile and fail-closed qualify all response-pattern columns."""
    cp = _cupy()
    float_kinds = cp.asnumpy(float_factor.kind)
    int_kinds = cp.asnumpy(int_factor.kind)
    float_pattern = np.flatnonzero((float_kinds == 3) | (float_kinds == 4))
    int_pattern = np.flatnonzero((int_kinds == 3) | (int_kinds == 4))
    float_labels = [float_factor.column_labels[index] for index in float_pattern]
    int_labels = [int_factor.column_labels[index] for index in int_pattern]
    unsupported = sorted(
        (set(float_labels) - {_PARKING})
        | (set(int_labels) - _AVAILABILITY_LABELS)
    )
    if unsupported:
        raise ValueError(
            "Phase 28 has no declared upstream expression for response columns: "
            + ", ".join(unsupported)
        )
    if set(float_labels) != {_PARKING}:
        raise ValueError("Phase 28 expected exactly one generated daily parking column")

    int_source_columns = {
        label: index for index, label in enumerate(int_factor.column_labels)
    }
    if "name:auto_ownership" not in int_source_columns:
        raise ValueError("Phase 28 drive availability requires compact auto_ownership")
    auto_column = int_source_columns["name:auto_ownership"]

    chooser_ids = np.asarray(metadata["chooser_ids"])
    start = np.asarray(metadata["start"], dtype=np.int16)
    end = np.asarray(metadata["end"], dtype=np.int16)
    slot_count = int(np.max(slots)) + 1
    slot_start = np.empty(slot_count, dtype=np.int16)
    slot_end = np.empty(slot_count, dtype=np.int16)
    for row, slot in enumerate(slots):
        slot_start[int(slot)] = start[row]
        slot_end[int(slot)] = end[row]
    if not (
        np.array_equal(slot_start[slots], start)
        and np.array_equal(slot_end[slots], end)
    ):
        raise ValueError("Phase 28 slot metadata is not an exact start/end mapping")

    parking_column = int(float_pattern[0])
    parking = cp.asnumpy(invocation.float_inputs[:, parking_column])
    durations = end.astype(np.int64) - start.astype(np.int64)
    owner_count = int(owner_starts.size)
    offsets = np.r_[owner_starts, chooser_ids.size]
    if parking_rates_override is None:
        rates = np.empty(owner_count, dtype=np.float64)
        for owner in range(owner_count):
            begin, finish = offsets[owner : owner + 2]
            local_duration = durations[begin:finish]
            rates[owner] = _infer_float32_multiplier(
                parking[begin:finish], local_duration
            )
    else:
        rates = np.asarray(parking_rates_override, dtype=np.float64)
        if rates.shape != (owner_count,):
            raise ValueError(
                "Phase 29 direct parking rates do not align with compact tours"
            )
    predicted = (rates[owner_index] * durations).astype(parking.dtype)
    if not np.array_equal(
        np.ascontiguousarray(predicted).view(np.uint8),
        np.ascontiguousarray(parking).view(np.uint8),
    ):
        raise ValueError(
            "Phase 28 daily_parking_cost does not satisfy rate * (end - start) "
            "with identical output bits"
        )

    sources = tuple(getattr(invocation, "skim_input_sources", ()))
    ranks = tuple(getattr(invocation, "skim_input_ranks", ()))
    groups = tuple(getattr(invocation, "skim_input_groups", ()))
    count = int(invocation.logical_skim_bindings)
    if not (len(sources) == len(ranks) == len(groups) == count):
        raise ValueError("Phase 28 requires semantic metadata for every resident skim")
    binding_by_source = {source: index for index, source in enumerate(sources)}

    group_arguments = {}
    position = count
    for group in sorted(set(groups)):
        representative = groups.index(group)
        rank = int(ranks[representative])
        origin = rebuilt_skim_arguments[position]
        destination = rebuilt_skim_arguments[position + 1]
        position += 2
        time_index = rebuilt_skim_arguments[position] if rank == 3 else None
        position += int(rank == 3)
        dest_count = rebuilt_skim_arguments[position]
        position += 1
        time_count = rebuilt_skim_arguments[position] if rank == 3 else None
        position += int(rank == 3)
        group_arguments[group] = (
            origin, destination, time_index, dest_count, time_count, rank
        )
    if position != len(rebuilt_skim_arguments):
        raise ValueError("Phase 28 could not parse the grouped resident skim ABI")

    selected_sources = []

    def gather(source):
        if source not in binding_by_source:
            raise ValueError(f"Phase 28 required resident skim {source!r} is absent")
        if source not in selected_sources:
            selected_sources.append(source)
        slot = selected_sources.index(source)
        binding = binding_by_source[source]
        group = groups[binding]
        rank = ranks[binding]
        if rank == 3:
            index = (
                f"((g{group}_orig[row] * g{group}_dest_count + g{group}_dest[row]) "
                f"* g{group}_time_count + g{group}_time[row])"
            )
        else:
            index = f"(g{group}_orig[row] * g{group}_dest_count + g{group}_dest[row])"
        return f"skim_{slot}[{index}]"

    int_assignments = []
    manifest = [
        (_PARKING, "daily_parking_rate[chooser] * (end[slot] - start[slot])")
    ]
    for column in int_pattern:
        label = int_factor.column_labels[int(column)]
        expression = _availability_expression(label, gather, auto_column)
        int_assignments.append(
            f"    int_inputs[row * int_columns + {int(column)}] = "
            f"({expression}) ? 1LL : 0LL;"
        )
        manifest.append((label, expression))

    used_groups = sorted({groups[binding_by_source[source]] for source in selected_sources})
    parameters = []
    arguments = []
    for slot, source in enumerate(selected_sources):
        parameters.append(f"    const float* skim_{slot}")
        arguments.append(rebuilt_skim_arguments[binding_by_source[source]])
    for group in used_groups:
        origin, destination, time_index, dest_count, time_count, rank = group_arguments[group]
        parameters.extend((
            f"    const long long* g{group}_orig",
            f"    const long long* g{group}_dest",
        ))
        arguments.extend((origin, destination))
        if rank == 3:
            parameters.append(f"    const long long* g{group}_time")
            arguments.append(time_index)
        parameters.append(f"    long long g{group}_dest_count")
        arguments.append(np.int64(dest_count))
        if rank == 3:
            parameters.append(f"    long long g{group}_time_count")
            arguments.append(np.int64(time_count))

    float_type = "float" if np.dtype(invocation.float_inputs.dtype) == np.float32 else "double"
    source = f'''extern "C" __global__ void choiceforge_semantic_inputs(
    {float_type}* float_inputs,
    long long* int_inputs,
    const int* owners,
    const int* slots,
    const short* slot_start,
    const short* slot_end,
    const double* parking_rates,
    long long rows,
    int float_columns,
    int int_columns{"," if parameters else ""}
{",".join(chr(10) + item for item in parameters)}) {{
    const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    const int owner = owners[row];
    const int slot = slots[row];
    float_inputs[row * float_columns + {parking_column}] = ({float_type})(
        parking_rates[owner] * (double)(slot_end[slot] - slot_start[slot]));
{chr(10).join(int_assignments)}
}}
'''
    # The existing CSR-local row slot indexes this tiny exact-alternative table;
    # no row-aligned copy of start/end is retained.
    compact_start = cp.asarray(slot_start, dtype=cp.int16)
    compact_end = cp.asarray(slot_end, dtype=cp.int16)
    kernel = cp.RawKernel(
        source, "choiceforge_semantic_inputs",
        options=("--std=c++11", "--fmad=false"),
    )
    kernel.compile()

    removed = int(
        float_factor.pattern_ids.nbytes + float_factor.pattern_offsets.nbytes
        + float_factor.pattern_values.nbytes + int_factor.pattern_ids.nbytes
        + int_factor.pattern_offsets.nbytes + int_factor.pattern_values.nbytes
    )
    program = SemanticInputProgram(
        kernel=kernel,
        kernel_arguments=tuple(arguments),
        slot_start=compact_start,
        slot_end=compact_end,
        parking_rates=cp.asarray(rates),
        rows=int(chooser_ids.size),
        float_columns=int(invocation.float_inputs.shape[1]),
        int_columns=int(invocation.int_inputs.shape[1]),
        generated_float_columns=tuple(int(x) for x in float_pattern),
        generated_int_columns=tuple(int(x) for x in int_pattern),
        expressions=tuple(manifest),
        removed_dictionary_bytes=removed,
    )
    return program, _remove_patterns(float_factor, float_pattern), _remove_patterns(
        int_factor, int_pattern
    )


def _remove_patterns(factor, columns):
    cp = _cupy()
    kinds = cp.asnumpy(factor.kind)
    positions = cp.asnumpy(factor.position)
    kinds[columns] = 4
    positions[columns] = 0
    return replace(
        factor,
        kind=cp.asarray(kinds, dtype=cp.int8),
        position=cp.asarray(positions, dtype=cp.int32),
        pattern_ids=cp.empty((factor.owner_values.shape[0], 0), dtype=cp.int32),
        pattern_offsets=cp.empty((0,), dtype=cp.int64),
        pattern_values=cp.empty((0,), dtype=factor.target.dtype),
        pattern_columns=0,
        semantic_generated_columns=int(len(columns)),
    )


def _infer_float32_multiplier(values, integer_multipliers):
    """Find one float64 rate whose correctly-rounded products match all bits."""
    values = np.asarray(values)
    multipliers = np.asarray(integer_multipliers, dtype=np.int64)
    if values.dtype != np.float32:
        usable = np.flatnonzero(multipliers != 0)
        if not usable.size:
            if np.any(values != 0):
                raise ValueError("nonzero value has a zero semantic multiplier")
            return 0.0
        candidate = float(values[usable[0]]) / float(multipliers[usable[0]])
        if not np.array_equal(
            (candidate * multipliers).astype(values.dtype), values
        ):
            raise ValueError("semantic multiplier is not exact for the target dtype")
        return candidate
    if np.any((multipliers == 0) & (values.view(np.uint32) != np.float32(0).view(np.uint32))):
        raise ValueError("nonzero parking cost has zero duration")
    lower = -np.inf
    upper = np.inf
    for value, multiplier in zip(values, multipliers):
        if multiplier == 0:
            continue
        previous = np.nextafter(value, np.float32(-np.inf), dtype=np.float32)
        following = np.nextafter(value, np.float32(np.inf), dtype=np.float32)
        lo = (float(previous) + float(value)) * 0.5 / float(multiplier)
        hi = (float(value) + float(following)) * 0.5 / float(multiplier)
        lower = max(lower, min(lo, hi))
        upper = min(upper, max(lo, hi))
    if not lower <= upper:
        raise ValueError("rounded products have no common float64 multiplier")
    candidates = [
        (lower + upper) * 0.5,
        lower,
        upper,
        np.nextafter(lower, np.inf),
        np.nextafter(upper, -np.inf),
    ]
    for candidate in candidates:
        predicted = (candidate * multipliers).astype(np.float32)
        if np.array_equal(predicted.view(np.uint32), values.view(np.uint32)):
            return float(candidate)
    raise ValueError("could not choose a rate inside the exact float32 rounding interval")
