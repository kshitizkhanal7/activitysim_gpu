"""Compile strict CUDA inputs from one-row-per-tour ActivitySim source tables.

Phase 29 removes dense preprocessor rows from input-plan construction.  The
compiler accepts the joined tour source table before ActivitySim's logsum
preprocessor, the land-use table, exact alternative metadata, model constants,
and the controlled standard-normal draws.  Every strict input slot and skim
coordinate is declared up front; unknown sources fail closed.

The captured dense invocation remains useful as an independent qualification
oracle and as the already-compiled utility ABI, but none of its row input
values or coordinate values are read while constructing the new plan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cuda_backend import _cupy
from .device_input_expansion import (
    CompactArrayFactor,
    ResidentInputExpansionPlan,
    _format_source_label,
)
from .semantic_input_generation import compile_semantic_input_program


_SEMANTIC_FLOAT = {"column:daily_parking_cost"}
_SEMANTIC_INT = {
    "name:sov_available",
    "name:sovtoll_available",
    "name:hov2_available",
    "name:hov2toll_available",
    "name:hov3_available",
    "name:hov3toll_available",
    "name:walk_local_available",
    "name:walk_lrf_available",
    "name:walk_express_available",
    "name:walk_heavyrail_available",
    "name:walk_commuter_available",
    "name:drive_local_available",
    "name:drive_lrf_available",
    "name:drive_express_available",
    "name:drive_heavyrail_available",
    "name:drive_commuter_available",
    "column:walk_ferry_available",
    "column:drive_ferry_available",
}

# Public input typing contract used by the Phase 30 native ABI bootstrap.  A
# source not listed here is not silently guessed from a sample population.
# The sets include both direct raw-table values and CUDA-generated values.
RAW_FLOAT_SOURCES = frozenset({
    "column:terminal_time",
    "column:ivot",
    "column:daily_parking_cost",
    "column:density_index",
    "column:origin_walk_time",
    "column:destination_walk_time",
    "column:dest_density_index",
    "column:totalWaitTaxi",
    "column:totalWaitSingleTNC",
    "column:totalWaitSharedTNC",
    # Trip-mode logsum ABI (Phase 35). These names are generated directly
    # from raw trip rows by TripLogsumNativePlan.
    "column:total_terminal_time",
    "column:total_parking_cost",
    "column:origin_density_index",
    "column:origTaxiWaitTime",
    "column:origSingleTNCWaitTime",
    "column:origSharedTNCWaitTime",
    "column:i_tour_mode",
})
RAW_INT_SOURCES = frozenset({
    "name:sov_available",
    "name:auto_ownership",
    "name:age",
    "name:is_joint",
    "name:is_atwork_subtour",
    "name:work_tour_is_SOV",
    "name:sovtoll_available",
    "name:hov2_available",
    "name:number_of_participants",
    "column:hhsize",
    "name:hov2toll_available",
    "name:hov3_available",
    "name:hov3toll_available",
    "column:dest_topology",
    "name:work_tour_is_bike",
    "name:walk_local_available",
    "name:walk_lrf_available",
    "name:walk_express_available",
    "name:walk_heavyrail_available",
    "name:walk_commuter_available",
    "name:drive_local_available",
    "name:drive_lrf_available",
    "name:drive_express_available",
    "name:drive_heavyrail_available",
    "name:drive_commuter_available",
    "column:is_indiv",
    "column:num_workers",
    "column:walk_ferry_available",
    "column:drive_ferry_available",
    "column:destination_in_cbd",
    "name:is_escort",
    # The reviewed MTC utility spec addresses these same preprocessor arrays
    # through both bare-name and dataframe-column syntax.
    "column:age",
    "column:auto_ownership",
    "column:is_joint",
    # Trip-mode logsum ABI (Phase 35).
    "column:trip_topology",
    "name:outbound",
    "name:inbound",
    "name:drive_local_available_outbound",
    "name:drive_local_available_inbound",
    "name:drive_lrf_available_outbound",
    "name:drive_lrf_available_inbound",
    "name:drive_express_available_outbound",
    "name:drive_express_available_inbound",
    "name:drive_heavyrail_available_outbound",
    "name:drive_heavyrail_available_inbound",
    "name:drive_commuter_available_outbound",
    "name:drive_commuter_available_inbound",
    "name:tour_mode_is_auto",
    "name:tour_mode_is_walk",
    "name:tour_mode_is_bike",
    "name:tour_mode_is_walk_transit",
    "name:tour_mode_is_drive_transit",
    "name:tour_mode_is_ride_hail",
    "column:tour_mode_is_SOV",
    "column:tour_mode_is_SR2",
    "column:tour_mode_is_SR3P",
    "column:first_trip",
    "column:outbound",
    "column:inbound",
    "column:tour_mode_is_auto",
    "column:tour_mode_is_walk",
    "column:tour_mode_is_walk_transit",
    "column:tour_mode_is_drive_transit",
    "column:tour_mode_is_ride_hail",
})

RAW_SOURCE_ALIASES = {
    "column:age": "name:age",
    "column:auto_ownership": "name:auto_ownership",
    "column:is_joint": "name:is_joint",
    "column:outbound": "name:outbound",
    "column:inbound": "name:inbound",
    "column:tour_mode_is_auto": "name:tour_mode_is_auto",
    "column:tour_mode_is_walk": "name:tour_mode_is_walk",
    "column:tour_mode_is_walk_transit": "name:tour_mode_is_walk_transit",
    "column:tour_mode_is_drive_transit": "name:tour_mode_is_drive_transit",
    "column:tour_mode_is_ride_hail": "name:tour_mode_is_ride_hail",
}


def _row_topology(metadata: Mapping[str, Any], rows: int):
    chooser_ids = np.asarray(metadata["chooser_ids"], dtype=np.int64)
    if chooser_ids.ndim != 1 or chooser_ids.size != rows or rows == 0:
        raise ValueError("Phase 29 requires one nonempty chooser id per utility row")
    first = np.r_[True, chooser_ids[1:] != chooser_ids[:-1]]
    owner_starts = np.flatnonzero(first).astype(np.int64)
    owners = np.cumsum(first, dtype=np.int32) - 1
    if not np.array_equal(chooser_ids[owner_starts][owners], chooser_ids):
        raise ValueError("Phase 29 chooser rows are not contiguous")
    offsets = np.r_[owner_starts, rows].astype(np.int64)
    start = np.asarray(metadata["start"], dtype=np.int16)
    end = np.asarray(metadata["end"], dtype=np.int16)
    if start.shape != chooser_ids.shape or end.shape != chooser_ids.shape:
        raise ValueError("Phase 29 time metadata does not match utility rows")
    pairs, slots = np.unique(
        np.column_stack((start, end)), axis=0, return_inverse=True
    )
    return chooser_ids, owner_starts, owners, offsets, pairs, slots.astype(np.int32)


def _declared_factor(target, labels, declarations, *, owner_count, slot_count, width):
    """Build a compact factor without inspecting any value in ``target``."""
    cp = _cupy()
    labels = tuple(labels)
    if len(labels) != len(declarations):
        raise ValueError("declared factor label/value count mismatch")
    kinds = []
    positions = []
    constants = []
    owner_columns = []
    slot_columns = []
    for label, (kind, value) in zip(labels, declarations):
        if kind == 0:
            positions.append(len(constants))
            constants.append(value)
        elif kind == 1:
            values = np.asarray(value)
            if values.shape != (owner_count,):
                raise ValueError(f"{label} owner source has shape {values.shape}")
            positions.append(len(owner_columns))
            owner_columns.append(values)
        elif kind == 2:
            values = np.asarray(value)
            if values.shape != (slot_count,):
                raise ValueError(f"{label} slot source has shape {values.shape}")
            positions.append(len(slot_columns))
            slot_columns.append(values)
        elif kind == 4:
            positions.append(0)
        else:
            raise ValueError(f"{label} has unsupported Phase 29 factor kind {kind}")
        kinds.append(kind)
    dtype = np.dtype(target.dtype)
    constants_array = np.asarray(constants, dtype=dtype)
    owner_array = (
        np.column_stack(owner_columns).astype(dtype, copy=False)
        if owner_columns else np.empty((owner_count, 0), dtype=dtype)
    )
    slot_array = (
        np.column_stack(slot_columns).astype(dtype, copy=False)
        if slot_columns else np.empty((slot_count, 0), dtype=dtype)
    )
    return CompactArrayFactor(
        target=cp.empty_like(target),
        original_shape=tuple(int(x) for x in target.shape),
        kind=cp.asarray(kinds, dtype=cp.int8),
        position=cp.asarray(positions, dtype=cp.int32),
        constants=cp.asarray(constants_array),
        owner_values=cp.asarray(owner_array),
        slot_values=cp.asarray(slot_array),
        pattern_ids=cp.empty((owner_count, 0), dtype=cp.int32),
        pattern_offsets=cp.empty((0,), dtype=cp.int64),
        pattern_values=cp.empty((0,), dtype=dtype),
        constant_columns=len(constants),
        owner_columns=len(owner_columns),
        slot_columns=len(slot_columns),
        pattern_columns=0,
        slot_count=slot_count,
        pattern_width=width,
        column_labels=labels,
        semantic_generated_columns=sum(kind == 4 for kind in kinds),
    )


def _land_values(land_use, zones, column):
    try:
        values = land_use[column].reindex(np.asarray(zones)).to_numpy()
    except KeyError as exc:
        raise ValueError(f"Phase 29 land-use column {column!r} is absent") from exc
    if pd.isna(values).any():
        raise ValueError(f"Phase 29 land-use lookup {column!r} contains missing zones")
    return values


def _density_band(land_use, zones):
    population = _land_values(land_use, zones, "TOTPOP")
    employment = _land_values(land_use, zones, "TOTEMP")
    acres = _land_values(land_use, zones, "TOTACRE")
    measure = (population + employment) / (acres / 640.0)
    return np.asarray(
        pd.cut(
            measure,
            bins=[-np.inf, 500, 2000, 5000, 15000, np.inf],
            labels=[5, 4, 3, 2, 1],
        ).astype(int)
    )


def _mapped(mapping, bands):
    return np.asarray([mapping[int(value)] for value in bands], dtype=np.float64)


def _scaled_lognormal(z, mean, sd, lower, upper):
    x = 1.0 + ((sd * sd) / (mean * mean))
    mu = np.log(mean / np.sqrt(x))
    sigma = np.sqrt(np.log(x))
    return np.exp(np.asarray(z, dtype=np.float64) * sigma + mu).clip(lower, upper)


def _wait_totals(raw, origin, destination, constants, owner_count):
    draws = np.asarray(raw.get("standard_normal_draws"))
    if draws.shape != (owner_count, 6):
        raise ValueError(
            "Phase 29 requires six compact controlled normal draws per tour; "
            f"received {draws.shape}"
        )
    origin_band = _density_band(raw["land_use"], origin)
    destination_band = _density_band(raw["land_use"], destination)
    lower = float(constants["min_waitTime"])
    upper = float(constants["max_waitTime"])
    families = (
        ("Taxi_waitTime_mean", "Taxi_waitTime_sd"),
        ("TNC_single_waitTime_mean", "TNC_single_waitTime_sd"),
        ("TNC_shared_waitTime_mean", "TNC_shared_waitTime_sd"),
    )
    totals = []
    for family, (mean_key, sd_key) in enumerate(families):
        origin_mean = _mapped(constants[mean_key], origin_band)
        origin_sd = _mapped(constants[sd_key], origin_band)
        destination_mean = _mapped(constants[mean_key], destination_band)
        destination_sd = _mapped(constants[sd_key], destination_band)
        origin_wait = _scaled_lognormal(
            draws[:, family * 2], origin_mean, origin_sd, lower, upper
        )
        destination_wait = _scaled_lognormal(
            draws[:, family * 2 + 1], destination_mean, destination_sd, lower, upper
        )
        totals.append(origin_wait + destination_wait)
    return tuple(totals)


def _owner_sources(raw, owner_ids, constants):
    tours = raw["tours"].reindex(owner_ids)
    if len(tours) != len(owner_ids) or tours.index.has_duplicates:
        raise ValueError("Phase 29 raw tour source is not unique and complete")
    purpose = str(raw["tour_purpose"])
    destination_column = (
        "workplace_zone_id" if purpose == "work" else "school_zone_id"
    )
    required = {
        "home_zone_id", destination_column, "value_of_time", "tour_type",
        "tour_category", "number_of_participants", "free_parking_at_work",
        "auto_ownership", "age", "hhsize", "num_workers",
    }
    missing = sorted(required - set(tours.columns))
    if missing:
        raise ValueError("Phase 29 raw tour columns are absent: " + ", ".join(missing))
    origin = np.asarray(tours["home_zone_id"])
    destination = np.asarray(tours[destination_column])
    land_use = raw["land_use"]
    waits = _wait_totals(raw, origin, destination, constants, len(owner_ids))
    density_index = (
        np.asarray(tours["density_index"])
        if "density_index" in tours
        else _land_values(land_use, origin, "density_index")
    )
    free_parking = (
        np.asarray(tours["tour_type"].astype(str) == "work")
        & np.asarray(tours["free_parking_at_work"], dtype=bool)
    )
    parking_rate = np.where(
        free_parking, 0.0, _land_values(land_use, destination, "PRKCST")
    ).astype(np.float64)
    values = {
        "column:terminal_time": _land_values(land_use, destination, "TERMINAL"),
        "column:ivot": 1.0 / np.asarray(tours["value_of_time"], dtype=np.float64),
        "column:density_index": density_index,
        "column:origin_walk_time": float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]),
        "column:destination_walk_time": float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]),
        "column:dest_density_index": _land_values(land_use, destination, "density_index"),
        "column:totalWaitTaxi": waits[0],
        "column:totalWaitSingleTNC": waits[1],
        "column:totalWaitSharedTNC": waits[2],
        "name:auto_ownership": np.asarray(tours["auto_ownership"]),
        "name:age": np.asarray(tours["age"]),
        "name:is_joint": np.asarray(tours["tour_category"].astype(str) == "joint"),
        "name:is_atwork_subtour": np.asarray(tours["tour_category"].astype(str) == "atwork"),
        "name:work_tour_is_SOV": np.zeros(len(tours), dtype=bool),
        "name:number_of_participants": np.asarray(tours["number_of_participants"]),
        "column:hhsize": np.asarray(tours["hhsize"]),
        "column:dest_topology": _land_values(land_use, destination, "TOPOLOGY"),
        "name:work_tour_is_bike": np.zeros(len(tours), dtype=bool),
        "column:is_indiv": np.asarray(tours["tour_category"].astype(str) != "joint"),
        "column:num_workers": np.asarray(tours["num_workers"]),
        "column:destination_in_cbd": (
            _land_values(land_use, destination, "area_type")
            < int(raw.get("cbd_threshold", 2))
        ),
        "name:is_escort": np.asarray(tours["tour_type"].astype(str) == "escort"),
    }
    return values, origin, destination, parking_rate


def _input_declarations(labels, values, semantic, owner_count):
    declarations = []
    manifest = []
    for label in labels:
        if label in semantic:
            declarations.append((4, None))
            manifest.append({"source": label, "provenance": "declared_cuda_formula"})
        elif label in values:
            value = values[label]
            if np.isscalar(value):
                declarations.append((0, value))
                provenance = "model_constant_formula"
            else:
                array = np.asarray(value)
                if array.shape != (owner_count,):
                    raise ValueError(f"Phase 29 source {label!r} is not per-tour")
                declarations.append((1, array))
                provenance = "raw_table_or_compact_stochastic_formula"
            manifest.append({"source": label, "provenance": provenance})
        else:
            raise ValueError(f"Phase 29 has no raw-table formula for {label!r}")
    return declarations, manifest


def _zone_positions(land_use, zones):
    lookup = pd.Series(np.arange(len(land_use), dtype=np.int64), index=land_use.index)
    values = lookup.reindex(np.asarray(zones)).to_numpy()
    if pd.isna(values).any():
        raise ValueError("Phase 29 skim coordinate refers to an absent land-use zone")
    return values.astype(np.int64)


def _period_positions(values):
    lookup = {"EA": 0, "AM": 1, "MD": 2, "PM": 3, "EV": 4}
    try:
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Phase 29 has no skim period coordinate for {exc.args[0]!r}") from exc


def _slot_values(row_values, slots, slot_count, label):
    result = np.empty(slot_count, dtype=np.asarray(row_values).dtype)
    seen = np.zeros(slot_count, dtype=bool)
    for value, slot in zip(row_values, slots):
        slot = int(slot)
        if seen[slot] and result[slot] != value:
            raise ValueError(f"Phase 29 slot source {label} is not slot-stable")
        result[slot] = value
        seen[slot] = True
    if not seen.all():
        raise ValueError(f"Phase 29 slot source {label} is incomplete")
    return result


def _coordinate_plan(invocation, metadata, raw, origin, destination, slots, width):
    sources = tuple(invocation.skim_input_sources)
    ranks = tuple(invocation.skim_input_ranks)
    groups = tuple(invocation.skim_input_groups)
    count = int(invocation.logical_skim_bindings)
    if not (len(sources) == len(ranks) == len(groups) == count):
        raise ValueError("Phase 29 requires semantic metadata for every skim binding")
    land_use = raw["land_use"]
    origin_position = _zone_positions(land_use, origin)
    destination_position = _zone_positions(land_use, destination)
    out_slot = _slot_values(
        _period_positions(metadata["out_period"]), slots, int(np.max(slots)) + 1,
        "out_period",
    )
    in_slot = _slot_values(
        _period_positions(metadata["in_period"]), slots, int(np.max(slots)) + 1,
        "in_period",
    )
    direction_coordinates = {
        "odt_skims": (origin_position, destination_position, out_slot),
        "dot_skims": (destination_position, origin_position, in_slot),
        "odr_skims": (origin_position, destination_position, in_slot),
        "dor_skims": (destination_position, origin_position, out_slot),
        "od_skims": (origin_position, destination_position),
        "od_skims_reverse": (destination_position, origin_position),
    }
    rebuilt = list(invocation.skim_arguments)
    factors = []
    labels = []
    oracle_coordinates = []
    manifest = []
    position = count
    for group in sorted(set(groups)):
        representative = groups.index(group)
        direction = str(sources[representative][1])
        rank = int(ranks[representative])
        expected = direction_coordinates.get(direction)
        if expected is None or len(expected) != rank:
            raise ValueError(
                f"Phase 29 has no coordinate contract for {direction!r} rank {rank}"
            )
        for axis in range(2):
            label = f"skim_group_{group}_{direction}_{'origin' if axis == 0 else 'destination'}"
            oracle_coordinates.append(rebuilt[position])
            factor = _declared_factor(
                rebuilt[position], (label,), ((1, expected[axis]),),
                owner_count=len(origin), slot_count=int(np.max(slots)) + 1,
                width=width,
            )
            rebuilt[position] = factor.target
            factors.append(factor)
            labels.append(label)
            manifest.append({"source": label, "provenance": "raw_zone_table_index"})
            position += 1
        if rank == 3:
            label = f"skim_group_{group}_{direction}_time"
            oracle_coordinates.append(rebuilt[position])
            factor = _declared_factor(
                rebuilt[position], (label,), ((2, expected[2]),),
                owner_count=len(origin), slot_count=int(np.max(slots)) + 1,
                width=width,
            )
            rebuilt[position] = factor.target
            factors.append(factor)
            labels.append(label)
            manifest.append({"source": label, "provenance": "alternative_period"})
            position += 1
        position += 1  # destination dimension
        if rank == 3:
            position += 1  # time dimension
    if position != len(rebuilt):
        raise ValueError("Phase 29 could not parse the grouped skim ABI")
    return (
        tuple(rebuilt), tuple(factors), tuple(labels), tuple(oracle_coordinates),
        manifest,
    )


@dataclass(frozen=True)
class ResidentRawTableInputPlan(ResidentInputExpansionPlan):
    """A sealed raw-table-to-strict-input CUDA plan with an oracle-only audit."""

    raw_source_manifest: tuple[Mapping[str, Any], ...] = ()
    oracle_dense_bytes_read_for_compile: int = 0
    oracle_comparison_performed: bool = True

    @classmethod
    def compile(
        cls, invocation, metadata: Mapping[str, Any], raw_source,
        *, validate_oracle: bool = True,
    ):
        cp = _cupy()
        (
            chooser_ids, owner_starts, owner_index, offsets, _pairs, slots,
        ) = _row_topology(metadata, int(invocation.rows))
        owner_ids = chooser_ids[owner_starts]
        owner_count = int(owner_ids.size)
        slot_count = int(np.max(slots)) + 1
        width = int(np.max(np.diff(offsets)))
        constants = raw_source["constants"]
        values, origin, destination, parking_rates = _owner_sources(
            raw_source, owner_ids, constants
        )
        float_labels = tuple(
            _format_source_label(item) for item in invocation.float_input_sources
        )
        int_labels = tuple(
            _format_source_label(item) for item in invocation.int_input_sources
        )
        float_declarations, float_manifest = _input_declarations(
            float_labels, values, _SEMANTIC_FLOAT, owner_count
        )
        int_declarations, int_manifest = _input_declarations(
            int_labels, values, _SEMANTIC_INT, owner_count
        )
        float_factor = _declared_factor(
            invocation.float_inputs, float_labels, float_declarations,
            owner_count=owner_count, slot_count=slot_count, width=width,
        )
        int_factor = _declared_factor(
            invocation.int_inputs, int_labels, int_declarations,
            owner_count=owner_count, slot_count=slot_count, width=width,
        )
        (
            rebuilt_args, coordinate_factors, coordinate_labels,
            oracle_coordinates, coordinate_manifest,
        ) = (
            _coordinate_plan(
                invocation, metadata, raw_source, origin, destination, slots, width
            )
        )
        semantic_program, float_factor, int_factor = compile_semantic_input_program(
            invocation=invocation,
            rebuilt_skim_arguments=rebuilt_args,
            metadata=metadata,
            owner_index=owner_index,
            owner_starts=owner_starts,
            slots=slots,
            float_factor=float_factor,
            int_factor=int_factor,
            parking_rates_override=parking_rates,
        )
        factors = (float_factor, int_factor) + coordinate_factors
        labels = ("float_inputs", "int_inputs") + coordinate_labels
        rebuilt = replace(
            invocation,
            float_inputs=float_factor.target,
            int_inputs=int_factor.target,
            skim_arguments=rebuilt_args,
        )
        plan = cls(
            invocation=rebuilt,
            offsets=cp.asarray(offsets),
            slots=cp.asarray(slots),
            owners=cp.empty(int(invocation.rows), dtype=cp.int32),
            factors=factors,
            labels=labels,
            original_dense_bytes=int(
                invocation.float_inputs.nbytes + invocation.int_inputs.nbytes
            ),
            original_coordinate_bytes=int(invocation.skim_coordinate_bytes),
            semantic_program=semantic_program,
            raw_source_manifest=tuple(
                float_manifest + int_manifest + coordinate_manifest
            ),
            oracle_dense_bytes_read_for_compile=0,
            oracle_comparison_performed=bool(validate_oracle),
        )
        plan.execute()
        cp.cuda.Stream.null.synchronize()
        # Oracle comparison happens only after construction.  It is deliberately
        # not used to choose a formula, factor kind, value, or coordinate source.
        if validate_oracle:
            comparisons = (
                ("float_inputs", plan.invocation.float_inputs, invocation.float_inputs),
                ("int_inputs", plan.invocation.int_inputs, invocation.int_inputs),
            )
            for label, generated, oracle in comparisons:
                if not bool(cp.array_equal(
                    cp.ascontiguousarray(generated).view(cp.uint8),
                    cp.ascontiguousarray(oracle).view(cp.uint8),
                )):
                    different = cp.asnumpy(generated != oracle)
                    positions = np.argwhere(different)
                    first = tuple(int(x) for x in positions[0]) if positions.size else None
                    raise ValueError(
                        f"Phase 29 raw-table {label} differs from the dense oracle at {first}"
                    )
            for factor, label, oracle in zip(
                coordinate_factors, coordinate_labels, oracle_coordinates
            ):
                generated = factor.target
                if not bool(cp.array_equal(generated, oracle)):
                    raise ValueError(
                        f"Phase 29 raw-table coordinate {label} differs from the dense oracle"
                    )
        return plan

    def raw_manifest(self):
        return {
            "contract": "declared_one_row_per_tour_and_land_use_sources",
            "sources": list(self.raw_source_manifest),
            "source_count": len(self.raw_source_manifest),
            "dense_oracle_bytes_read_for_compile": self.oracle_dense_bytes_read_for_compile,
            "availability_formulas": len(_SEMANTIC_INT),
            "parking_rate_source": "land_use.PRKCST_or_free_parking_at_work",
            "oracle_comparison_performed": self.oracle_comparison_performed,
        }
