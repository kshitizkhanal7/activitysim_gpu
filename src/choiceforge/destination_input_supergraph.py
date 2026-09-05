"""Phase 50-52 compact destination-input generation and fused CUDA execution.

ActivitySim's public tour-mode preprocessor expands one owner and each sampled
destination into 41 dense row fields, then Sharrow resolves, packs, and uploads
those fields plus six groups of skim coordinates.  This module intercepts the
location-logsum contract before that expansion.  It uploads compact owner
state and sampled destination IDs, generates the exact dense ABI on CUDA, and
hands its utilities to the already-qualified resident nested-logit graph.

The implementation is intentionally public-MTC-specific and fail closed.  A
new row source, noncontiguous owner group, noncanonical zone universe, or skim
direction changes the ABI and stops the run instead of silently falling back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cuda_backend import _cupy
from .native_abi_bootstrap import (
    NativeSkimCube,
    NativeStrictAbiPlan,
    compile_native_strict_abi,
)
from .nested_logit import mtc21_nested_logsums_cuda
from .raw_table_input_generation import _density_band, _scaled_lognormal
from .semantic_input_generation import _AVAILABILITY_LABELS, _availability_expression
from .sharrow_cuda import _shared_memory_bytes, generate_cuda_source
from .sharrow_ir import specification_ir


_FLOAT_LABELS = {
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
}

_OWNER_INT_LABELS = (
    "name:auto_ownership",
    "name:age",
    "name:is_joint",
    "name:is_atwork_subtour",
    "name:work_tour_is_SOV",
    "name:number_of_participants",
    "column:hhsize",
    "name:work_tour_is_bike",
    "column:is_indiv",
    "column:num_workers",
    "name:is_escort",
)

_DESTINATION_INT_LABELS = {
    "column:dest_topology",
    "column:destination_in_cbd",
}

_INT_LABELS = set(_OWNER_INT_LABELS) | _DESTINATION_INT_LABELS | set(
    _AVAILABILITY_LABELS
)

_GENERATOR_CACHE: dict[str, Any] = {}
_FUSED_UTILITY_CACHE: dict[str, Any] = {}
_ROW_OWNER_KERNEL = None
_PHASE52_SOURCE = Path(__file__).with_name("kernels") / "phase52_public_destination_tile4.cu"


def prewarm_phase52_public_runtime(cp=None) -> dict[str, Any]:
    """Compile the checked-in Phase 52 program before ActivitySim component timers.

    CuPy's own content-addressed disk cache supplies the cross-process binary
    cache.  This function adds a checked-in source boundary and a process-local
    executable cache keyed by the exact source SHA-256.
    """
    cp = cp or _cupy()
    started = time.perf_counter()
    if not _PHASE52_SOURCE.exists():
        return {
            "available": False,
            "compiled": False,
            "seconds": 0.0,
            "source_sha256": None,
            "source_path": str(_PHASE52_SOURCE),
        }
    source = _PHASE52_SOURCE.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    kernel = _FUSED_UTILITY_CACHE.get(source_sha256)
    compiled = kernel is None
    if kernel is None:
        kernel = cp.RawKernel(
            source,
            "choiceforge_strict_ir_v3",
            options=("--std=c++11", "--fmad=true", "--prec-div=true", "--ftz=true"),
        )
        kernel.compile()
        _FUSED_UTILITY_CACHE[source_sha256] = kernel
    cp.cuda.Stream.null.synchronize()
    return {
        "available": True,
        "compiled": compiled,
        "seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "source_path": str(_PHASE52_SOURCE),
        "cache_contract": "checked-in-source-sha256-plus-cupy-disk-binary-cache",
    }


def _label(source) -> str:
    return ":".join(str(part) for part in source)


def _period_positions(values) -> np.ndarray:
    lookup = {"EA": 0, "AM": 1, "MD": 2, "PM": 3, "EV": 4}
    try:
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(
            f"Phase 50 has no public skim-period coordinate for {exc.args[0]!r}"
        ) from exc


def _owner_topology(index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(index, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("Phase 50 requires a nonempty one-dimensional owner index")
    first = np.r_[True, ids[1:] != ids[:-1]]
    starts = np.flatnonzero(first).astype(np.int64)
    offsets = np.r_[starts, len(ids)].astype(np.int64)
    # Every transition begins a run by construction.  Repeated run keys are
    # therefore the only way an owner can be noncontiguous; checking just the
    # compact run keys avoids allocating two dense validation vectors.
    if len(np.unique(ids[starts])) != len(starts):
        raise ValueError("Phase 50 sampled destination owners are not contiguous")
    return ids, starts, offsets


def _stable_owner(values, starts, offsets, label, *, dtype=None) -> np.ndarray:
    values = np.asarray(values, dtype=dtype)
    owner = values[starts]
    # Compare neighbours within each run.  This is equivalent to expanding
    # every owner value back to all rows, but eliminates that large temporary
    # allocation for each of the compact owner columns.
    different = np.asarray(values[1:] != values[:-1], dtype=bool)
    both_missing = np.asarray(pd.isna(values[1:]) & pd.isna(values[:-1]), dtype=bool)
    different &= ~both_missing
    different[offsets[1:-1] - 1] = False
    if bool(np.any(different)):
        raise ValueError(f"Phase 50 owner source {label!r} varies inside a sample group")
    return np.ascontiguousarray(owner)


def _setting_value(model_settings, name, purpose):
    value = getattr(model_settings, name)
    if isinstance(value, dict):
        if purpose not in value:
            raise ValueError(f"Phase 50 {name} has no value for purpose {purpose!r}")
        value = value[purpose]
    return value


def _time_state(
    choosers,
    model_settings,
    network_los,
    purpose,
    *,
    in_period_col,
    out_period_col,
    duration_col,
):
    if (in_period_col is None) != (out_period_col is None):
        raise ValueError("Phase 50 requires both explicit period columns or neither")
    if in_period_col is not None:
        incoming = network_los.skim_time_period_label(
            choosers[in_period_col], as_cat=True
        )
        outgoing = network_los.skim_time_period_label(
            choosers[out_period_col], as_cat=True
        )
    else:
        incoming_value = _setting_value(model_settings, "IN_PERIOD", purpose)
        outgoing_value = _setting_value(model_settings, "OUT_PERIOD", purpose)
        incoming = network_los.skim_time_period_label(
            incoming_value, as_cat=True, broadcast_to=choosers.index
        )
        outgoing = network_los.skim_time_period_label(
            outgoing_value, as_cat=True, broadcast_to=choosers.index
        )
    if duration_col is not None:
        duration = np.asarray(choosers[duration_col], dtype=np.int16)
    else:
        duration = np.full(
            len(choosers),
            int(_setting_value(model_settings, "IN_PERIOD", purpose))
            - int(_setting_value(model_settings, "OUT_PERIOD", purpose)),
            dtype=np.int16,
        )
    return (
        _period_positions(np.asarray(outgoing).astype(str)),
        _period_positions(np.asarray(incoming).astype(str)),
        duration,
    )


def _mapped(mapping, bands):
    return np.asarray([mapping[int(value)] for value in bands], dtype=np.float64)


def _wait_table(land_use, origin, draws, constants) -> np.ndarray:
    """Return exact float32 totals for every owner and destination density band."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.shape != (len(origin), 6):
        raise ValueError(
            "Phase 50 requires six compact controlled normal draws per owner; "
            f"received {draws.shape}"
        )
    origin_band = _density_band(land_use, origin)
    destination_bands = np.arange(1, 6, dtype=np.int64)
    lower = float(constants["min_waitTime"])
    upper = float(constants["max_waitTime"])
    families = (
        ("Taxi_waitTime_mean", "Taxi_waitTime_sd"),
        ("TNC_single_waitTime_mean", "TNC_single_waitTime_sd"),
        ("TNC_shared_waitTime_mean", "TNC_shared_waitTime_sd"),
    )
    result = np.empty((len(origin), 5, 3), dtype=np.float32)
    for family, (mean_key, sd_key) in enumerate(families):
        origin_wait = _scaled_lognormal(
            draws[:, family * 2],
            _mapped(constants[mean_key], origin_band),
            _mapped(constants[sd_key], origin_band),
            lower,
            upper,
        )
        for position, band in enumerate(destination_bands):
            destination_wait = _scaled_lognormal(
                draws[:, family * 2 + 1],
                np.full(len(origin), constants[mean_key][int(band)], dtype=np.float64),
                np.full(len(origin), constants[sd_key][int(band)], dtype=np.float64),
                lower,
                upper,
            )
            result[:, position, family] = origin_wait + destination_wait
    return result


def _land_signature(land_use, columns) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(land_use.index, dtype=np.int64)))
    for name in columns:
        values = np.ascontiguousarray(np.asarray(land_use[name]))
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class CompactDestinationPacket:
    offsets: Any
    owner_float: Any
    owner_int: Any
    owner_origin: Any
    owner_out_period: Any
    owner_in_period: Any
    owner_duration: Any
    row_destination: Any
    wait_table: Any
    compact_bytes: int
    owners: int
    workspace_hits: int = 0
    workspace_allocations: int = 0


class DestinationInputSupergraph:
    """Compile and execute the public destination logsum from compact state."""

    version = 1

    def __init__(
        self,
        bridge,
        *,
        cbd_threshold: int,
        cp=None,
        fused: bool = False,
        tile_rows: int = 1,
        persistent: bool = False,
    ):
        self.cp = cp or _cupy()
        self.bridge = bridge
        self.cbd_threshold = int(cbd_threshold)
        self.fused = bool(fused)
        self.tile_rows = int(tile_rows)
        if self.tile_rows not in {1, 2, 4}:
            raise ValueError("destination fused tile_rows must be 1, 2, or 4")
        if self.tile_rows > 1 and not self.fused:
            raise ValueError("destination row tiling requires fused execution")
        self.persistent = bool(persistent)
        self._land_signature = None
        self._land_float = None
        self._land_int = None
        self._events: list[dict[str, Any]] = []
        self._device_buffers: dict[tuple[Any, ...], Any] = {}
        self._utility_buffer = None
        self._native_plan_cache: dict[str, NativeStrictAbiPlan] = {}
        self._semantic_plan_cache: dict[tuple[str, str], tuple[Any, ...]] = {}

    @staticmethod
    def _capacity(rows: int) -> int:
        rows = max(1, int(rows))
        return 1 << (rows - 1).bit_length()

    def _upload(self, name: str, host):
        host = np.ascontiguousarray(host)
        if not self.persistent:
            return self.cp.asarray(host), False
        tail = tuple(host.shape[1:])
        key = (name, tail, host.dtype.str)
        buffer = self._device_buffers.get(key)
        hit = buffer is not None and buffer.shape[0] >= host.shape[0]
        if not hit:
            buffer = self.cp.empty(
                (self._capacity(host.shape[0]),) + tail, dtype=host.dtype
            )
            self._device_buffers[key] = buffer
        view = buffer[: host.shape[0]]
        view.set(host)
        return view, hit

    def _utilities(self, rows: int, alternatives: int):
        hit = (
            self._utility_buffer is not None
            and self._utility_buffer.shape[0] >= rows
            and self._utility_buffer.shape[1] == alternatives
        )
        if not hit:
            self._utility_buffer = self.cp.empty(
                (self._capacity(rows), alternatives), dtype=self.cp.float32
            )
        return self._utility_buffer[:rows], hit

    def _resident_land(self, land_use):
        columns = (
            "TERMINAL", "PRKCST", "OPRKCST", "density_index", "TOPOLOGY",
            "area_type", "TOTPOP", "TOTEMP", "TOTACRE",
        )
        missing = sorted(set(columns) - set(land_use.columns))
        if missing:
            raise ValueError("Phase 50 land-use columns are absent: " + ", ".join(missing))
        zone_ids = np.asarray(land_use.index, dtype=np.int64)
        if not np.array_equal(zone_ids, np.arange(len(zone_ids), dtype=np.int64)):
            raise ValueError("Phase 50 requires the reviewed zero-based dense zone universe")
        if "access_dist_transit" in land_use:
            raise ValueError("Phase 50 has not qualified transit subzone access distances")
        signature = _land_signature(land_use, columns)
        if self._land_signature is None:
            land_float = np.column_stack(
                [np.asarray(land_use[name], dtype=np.float64) for name in columns[:4]]
            )
            land_int = np.column_stack(
                (
                    np.asarray(land_use["TOPOLOGY"], dtype=np.int64),
                    np.asarray(land_use["area_type"], dtype=np.int64),
                    _density_band(land_use, zone_ids).astype(np.int64),
                )
            )
            i32 = np.iinfo(np.int32)
            if self.fused and (land_int.min() < i32.min or land_int.max() > i32.max):
                raise ValueError("Phase 51 land-use integer state exceeds int32")
            self._land_float = self.cp.asarray(np.ascontiguousarray(land_float))
            land_int_storage = land_int.astype(np.int32) if self.fused else land_int
            self._land_int = self.cp.asarray(np.ascontiguousarray(land_int_storage))
            self._land_signature = signature
        elif signature != self._land_signature:
            raise ValueError("Phase 50 resident land-use table changed during the model")
        return self._land_float, self._land_int

    def _packet(
        self,
        state,
        choosers,
        land_use,
        constants,
        model_settings,
        network_los,
        purpose,
        *,
        in_period_col,
        out_period_col,
        duration_col,
    ) -> CompactDestinationPacket:
        owner_ids, starts, offsets = _owner_topology(choosers.index)
        owners = len(starts)
        orig_name = model_settings.CHOOSER_ORIG_COL_NAME
        dest_name = model_settings.ALT_DEST_COL_NAME
        # ActivitySim deliberately permits the school/workplace location
        # chooser to omit tour fields.  The public preprocessor's _DF_IS_TOUR
        # branch then supplies one participant and false tour-category flags.
        # Preserve that contract here instead of requiring the union of every
        # chooser schema used by the five destination components.
        is_tour = "tour_type" in choosers.columns
        required = {
            orig_name, dest_name, "hhsize", "density_index", "age",
            "auto_ownership", "num_workers", "value_of_time",
        }
        if is_tour:
            required.update(("number_of_participants", "free_parking_at_work"))
        missing = sorted(required - set(choosers.columns))
        if missing:
            raise ValueError("Phase 50 chooser columns are absent: " + ", ".join(missing))
        origin = _stable_owner(choosers[orig_name], starts, offsets, orig_name, dtype=np.int32)
        destination = np.ascontiguousarray(choosers[dest_name], dtype=np.int32)
        if (
            origin.min() < 0 or destination.min() < 0
            or origin.max() >= len(land_use) or destination.max() >= len(land_use)
        ):
            raise ValueError("Phase 50 origin or destination is outside the dense zone universe")
        out_row, in_row, duration_row = _time_state(
            choosers, model_settings, network_los, purpose,
            in_period_col=in_period_col,
            out_period_col=out_period_col,
            duration_col=duration_col,
        )
        out_period = _stable_owner(out_row, starts, offsets, "out_period", dtype=np.int8)
        in_period = _stable_owner(in_row, starts, offsets, "in_period", dtype=np.int8)
        duration = _stable_owner(duration_row, starts, offsets, "duration", dtype=np.int16)

        tour_type = (
            _stable_owner(choosers["tour_type"].astype(str), starts, offsets, "tour_type")
            if is_tour else np.full(owners, "", dtype="<U1")
        )
        category = (
            _stable_owner(
                choosers["tour_category"].astype(str), starts, offsets, "tour_category"
            )
            if "tour_category" in choosers else np.full(owners, "", dtype="<U1")
        )
        auto = _stable_owner(
            choosers["auto_ownership"], starts, offsets, "auto_ownership", dtype=np.int64
        )
        age = _stable_owner(choosers["age"], starts, offsets, "age", dtype=np.int64)
        participants = (
            _stable_owner(
                choosers["number_of_participants"], starts, offsets,
                "number_of_participants", dtype=np.int64,
            )
            if is_tour else np.ones(owners, dtype=np.int64)
        )
        hhsize = _stable_owner(choosers["hhsize"], starts, offsets, "hhsize", dtype=np.int64)
        workers = _stable_owner(
            choosers["num_workers"], starts, offsets, "num_workers", dtype=np.int64
        )
        free_parking = (
            _stable_owner(
                choosers["free_parking_at_work"], starts, offsets,
                "free_parking_at_work", dtype=bool,
            )
            if is_tour else np.zeros(owners, dtype=bool)
        )
        value_of_time = _stable_owner(
            choosers["value_of_time"], starts, offsets, "value_of_time", dtype=np.float64
        )
        density = _stable_owner(
            choosers["density_index"], starts, offsets, "density_index", dtype=np.float64
        )
        if not np.isfinite(value_of_time).all() or np.any(value_of_time == 0):
            raise ValueError("Phase 50 value_of_time must be finite and nonzero")

        parent_sov = np.zeros(owners, dtype=np.int64)
        parent_bike = np.zeros(owners, dtype=np.int64)
        if "parent_tour_id" in choosers:
            parent = _stable_owner(
                choosers["parent_tour_id"], starts, offsets, "parent_tour_id"
            )
            tours = state.get_dataframe("tours")
            parent_modes = tours["tour_mode"].reindex(parent)
            if parent_modes.isna().any():
                raise ValueError("Phase 50 parent tour mode lookup is incomplete")
            modes = parent_modes.astype(str).to_numpy()
            parent_sov = np.isin(modes, ["DRIVEALONEFREE", "DRIVEALONEPAY"]).astype(np.int64)
            parent_bike = (modes == "BIKE").astype(np.int64)

        mandatory = (category == "mandatory").astype(np.int64)
        free = ((tour_type == "work") & free_parking).astype(np.int64)
        owner_float = np.column_stack(
            (
                (1.0 / value_of_time).astype(np.float32),
                density.astype(np.float32),
                np.full(owners, float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]), dtype=np.float32),
                np.full(owners, float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]), dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        owner_int = np.column_stack(
            (
                auto,
                age,
                (category == "joint").astype(np.int64),
                (category == "atwork").astype(np.int64),
                parent_sov,
                participants,
                hhsize,
                parent_bike,
                (category != "joint").astype(np.int64),
                workers,
                (tour_type == "escort").astype(np.int64),
                mandatory,
                free,
            )
        ).astype(np.int64, copy=False)
        i32 = np.iinfo(np.int32)
        if self.fused and (owner_int.min() < i32.min or owner_int.max() > i32.max):
            raise ValueError("Phase 51 compact owner integer state exceeds int32")

        rng = state.get_rn_generator()
        # ActivitySim's broadcast=True implementation first generates on this
        # exact unique-index Series and then expands back to all sampled rows.
        # Call that underlying public contract directly: the controlled RNG
        # advances identically, while the dense N_rows x 6 reindex disappears.
        compact_index = pd.Index(owner_ids[starts], name=choosers.index.name)
        compact_draws = rng.normal_for_df(
            compact_index.to_series(), broadcast=False, size=6
        )
        compact_draws = np.asarray(compact_draws, dtype=np.float64)
        waits = _wait_table(land_use, origin, compact_draws, constants)

        owner_int_storage = owner_int.astype(np.int32) if self.fused else owner_int
        host_arrays = (
            offsets, owner_float, owner_int_storage, origin, out_period, in_period,
            duration, destination, waits,
        )
        uploaded = [
            self._upload(name, item)
            for name, item in zip(
                (
                    "offsets", "owner_float", "owner_int", "owner_origin",
                    "owner_out_period", "owner_in_period", "owner_duration",
                    "row_destination", "wait_table",
                ),
                host_arrays,
            )
        ]
        device = [item[0] for item in uploaded]
        return CompactDestinationPacket(
            *device,
            compact_bytes=int(sum(np.asarray(item).nbytes for item in host_arrays)),
            owners=owners,
            workspace_hits=sum(bool(item[1]) for item in uploaded),
            workspace_allocations=sum(not bool(item[1]) for item in uploaded),
        )

    @staticmethod
    def _group_abi(invocation):
        sources = tuple(invocation.skim_input_sources)
        ranks = tuple(invocation.skim_input_ranks)
        groups = tuple(invocation.skim_input_groups)
        count = int(invocation.logical_skim_bindings)
        if not (len(sources) == len(ranks) == len(groups) == count):
            raise ValueError("Phase 50 skim source metadata is incomplete")
        arguments = invocation.skim_arguments
        result = {}
        position = count
        for group in sorted(set(groups)):
            representative = groups.index(group)
            rank = int(ranks[representative])
            origin = arguments[position]
            destination = arguments[position + 1]
            position += 2
            period = arguments[position] if rank == 3 else None
            position += int(rank == 3)
            dest_count = int(arguments[position])
            position += 1
            time_count = int(arguments[position]) if rank == 3 else 1
            position += int(rank == 3)
            result[group] = {
                "direction": str(sources[representative][1]),
                "rank": rank,
                "origin": origin,
                "destination": destination,
                "period": period,
                "dest_count": dest_count,
                "time_count": time_count,
            }
        if position != len(arguments):
            raise ValueError("Phase 50 could not parse grouped skim arguments")
        return sources, ranks, groups, result

    def _generator(self, invocation):
        cp = self.cp
        float_labels = tuple(_label(item) for item in invocation.float_input_sources)
        int_labels = tuple(_label(item) for item in invocation.int_input_sources)
        if set(float_labels) != _FLOAT_LABELS or len(float_labels) != len(_FLOAT_LABELS):
            raise ValueError("Phase 50 float row-source ABI changed")
        if set(int_labels) != _INT_LABELS or len(int_labels) != len(_INT_LABELS):
            unknown = sorted(set(int_labels) ^ _INT_LABELS)
            raise ValueError("Phase 50 integer row-source ABI changed: " + ", ".join(unknown))
        fp = {name: index for index, name in enumerate(float_labels)}
        ip = {name: index for index, name in enumerate(int_labels)}
        sources, ranks, groups, group_abi = self._group_abi(invocation)
        binding_by_source = {source: index for index, source in enumerate(sources)}
        selected_sources = []

        def gather(source):
            if source not in binding_by_source:
                raise ValueError(f"Phase 50 required resident skim {source!r} is absent")
            if source not in selected_sources:
                selected_sources.append(source)
            slot = selected_sources.index(source)
            binding = binding_by_source[source]
            direction = str(source[1])
            if direction in {"odt_skims", "odr_skims", "od_skims"}:
                origin, destination = "origin", "destination"
            elif direction in {"dot_skims", "dor_skims", "od_skims_reverse"}:
                origin, destination = "destination", "origin"
            else:
                raise ValueError(f"Phase 50 unsupported skim direction {direction!r}")
            group = groups[binding]
            if int(ranks[binding]) == 3:
                period = "out_period" if direction in {"odt_skims", "dor_skims"} else "in_period"
                index = (
                    f"(({origin} * sg{group}_dest_count + {destination}) * "
                    f"sg{group}_time_count + {period})"
                )
            else:
                index = f"({origin} * sg{group}_dest_count + {destination})"
            return f"skim_{slot}[{index}]"

        availability = []
        for label in sorted(_AVAILABILITY_LABELS):
            expression = _availability_expression(label, gather, ip["name:auto_ownership"])
            availability.append(
                f"      int_inputs[row * int_columns + {ip[label]}] = ({expression}) ? 1LL : 0LL;"
            )

        float_assignments = {
            "column:terminal_time": "(float)land_float[destination * 4 + 0]",
            "column:ivot": "owner_float[owner * 4 + 0]",
            "column:daily_parking_cost": (
                "(float)(((owner_int[owner * 13 + 12] != 0LL) ? 0.0 : "
                "land_float[destination * 4 + ((owner_int[owner * 13 + 11] != 0LL) ? 1 : 2)]) "
                "* (double)owner_duration[owner])"
            ),
            "column:density_index": "owner_float[owner * 4 + 1]",
            "column:origin_walk_time": "owner_float[owner * 4 + 2]",
            "column:destination_walk_time": "owner_float[owner * 4 + 3]",
            "column:dest_density_index": "(float)land_float[destination * 4 + 3]",
            "column:totalWaitTaxi": "wait_table[(owner * 5 + destination_band) * 3 + 0]",
            "column:totalWaitSingleTNC": "wait_table[(owner * 5 + destination_band) * 3 + 1]",
            "column:totalWaitSharedTNC": "wait_table[(owner * 5 + destination_band) * 3 + 2]",
        }
        float_lines = [
            f"      float_inputs[row * float_columns + {fp[label]}] = {expression};"
            for label, expression in float_assignments.items()
        ]
        owner_int_position = {name: position for position, name in enumerate(_OWNER_INT_LABELS)}
        int_lines = [
            f"      int_inputs[row * int_columns + {ip[label]}] = owner_int[owner * 13 + {owner_int_position[label]}];"
            for label in _OWNER_INT_LABELS
        ]
        int_lines.extend((
            f"      int_inputs[row * int_columns + {ip['column:dest_topology']}] = land_int[destination * 3 + 0];",
            f"      int_inputs[row * int_columns + {ip['column:destination_in_cbd']}] = (land_int[destination * 3 + 1] < cbd_threshold) ? 1LL : 0LL;",
        ))

        parameters = []
        arguments = []
        for slot, source in enumerate(selected_sources):
            parameters.append(f"    const float* skim_{slot}")
            arguments.append(invocation.skim_arguments[binding_by_source[source]])
        used_groups = sorted({groups[binding_by_source[source]] for source in selected_sources})
        for group in used_groups:
            metadata = group_abi[group]
            parameters.append(f"    long long sg{group}_dest_count")
            arguments.append(np.int64(metadata["dest_count"]))
            if metadata["rank"] == 3:
                parameters.append(f"    long long sg{group}_time_count")
                arguments.append(np.int64(metadata["time_count"]))

        coordinate_lines = []
        for group in sorted(group_abi):
            metadata = group_abi[group]
            direction = metadata["direction"]
            reverse = direction in {"dot_skims", "dor_skims", "od_skims_reverse"}
            parameters.extend((
                f"    long long* cg{group}_origin",
                f"    long long* cg{group}_destination",
            ))
            arguments.extend((metadata["origin"], metadata["destination"]))
            coordinate_lines.extend((
                f"      cg{group}_origin[row] = {'destination' if reverse else 'origin'};",
                f"      cg{group}_destination[row] = {'origin' if reverse else 'destination'};",
            ))
            if metadata["rank"] == 3:
                parameters.append(f"    long long* cg{group}_period")
                arguments.append(metadata["period"])
                period = "out_period" if direction in {"odt_skims", "dor_skims"} else "in_period"
                coordinate_lines.append(f"      cg{group}_period[row] = {period};")

        schema = {
            "float": float_labels,
            "int": int_labels,
            "skim": [list(item) for item in sources],
            "groups": {
                str(key): (value["direction"], value["rank"])
                for key, value in group_abi.items()
            },
        }
        key = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        kernel = _GENERATOR_CACHE.get(key)
        compiled = kernel is None
        if kernel is None:
            extra = ",\n" + ",\n".join(parameters) if parameters else ""
            source = f'''extern "C" __global__ void phase50_destination_inputs(
    float* float_inputs,
    long long* int_inputs,
    const long long* offsets,
    const float* owner_float,
    const long long* owner_int,
    const int* owner_origin,
    const signed char* owner_out_period,
    const signed char* owner_in_period,
    const short* owner_duration,
    const int* row_destination,
    const float* wait_table,
    const double* land_float,
    const long long* land_int,
    int cbd_threshold,
    int owner_count,
    int float_columns,
    int int_columns{extra}) {{
  const int owner = blockIdx.x;
  if (owner >= owner_count) return;
  const long long origin = owner_origin[owner];
  const long long out_period = owner_out_period[owner];
  const long long in_period = owner_in_period[owner];
  for (long long row = offsets[owner] + threadIdx.x;
       row < offsets[owner + 1]; row += blockDim.x) {{
      const long long destination = row_destination[row];
      const int destination_band = (int)land_int[destination * 3 + 2] - 1;
{chr(10).join(float_lines)}
{chr(10).join(int_lines)}
{chr(10).join(availability)}
{chr(10).join(coordinate_lines)}
  }}
}}
'''
            kernel = cp.RawKernel(
                source,
                "phase50_destination_inputs",
                options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=true"),
            )
            kernel.compile()
            _GENERATOR_CACHE[key] = kernel
        return kernel, tuple(arguments), key, compiled

    def _fused_utility(self, document, native):
        """Compile strict utility expressions directly over the compact packet."""
        invocation = native.invocation
        bindings = native.bindings
        float_labels = tuple(_label(item) for item in invocation.float_input_sources)
        int_labels = tuple(_label(item) for item in invocation.int_input_sources)
        if set(float_labels) != _FLOAT_LABELS or len(float_labels) != len(_FLOAT_LABELS):
            raise ValueError("Phase 51 float row-source ABI changed")
        if set(int_labels) != _INT_LABELS or len(int_labels) != len(_INT_LABELS):
            unknown = sorted(set(int_labels) ^ _INT_LABELS)
            raise ValueError("Phase 51 integer row-source ABI changed: " + ", ".join(unknown))

        tile_index = "tile_row * 10 + " if self.tile_rows > 1 else ""
        int_tile_index = "tile_row * 31 + " if self.tile_rows > 1 else ""
        float_variables = {
            label: f"phase51_float_values[{tile_index}{position}]"
            for position, label in enumerate(float_labels)
        }
        int_variables = {
            label: f"(long long)phase51_int_values[{int_tile_index}{position}]"
            for position, label in enumerate(int_labels)
        }
        float_expressions = {
            "column:terminal_time": "(float)phase51_land_float[destination * 4 + 0]",
            "column:ivot": "phase51_owner_float[owner * 4 + 0]",
            "column:daily_parking_cost": (
                "(float)(((phase51_owner_int[owner * 13 + 12] != 0LL) ? 0.0 : "
                "phase51_land_float[destination * 4 + "
                "((phase51_owner_int[owner * 13 + 11] != 0LL) ? 1 : 2)]) * "
                "(double)phase51_owner_duration[owner])"
            ),
            "column:density_index": "phase51_owner_float[owner * 4 + 1]",
            "column:origin_walk_time": "phase51_owner_float[owner * 4 + 2]",
            "column:destination_walk_time": "phase51_owner_float[owner * 4 + 3]",
            "column:dest_density_index": "(float)phase51_land_float[destination * 4 + 3]",
            "column:totalWaitTaxi": (
                "phase51_wait_table[(owner * 5 + destination_band) * 3 + 0]"
            ),
            "column:totalWaitSingleTNC": (
                "phase51_wait_table[(owner * 5 + destination_band) * 3 + 1]"
            ),
            "column:totalWaitSharedTNC": (
                "phase51_wait_table[(owner * 5 + destination_band) * 3 + 2]"
            ),
        }
        owner_int_position = {
            name: position for position, name in enumerate(_OWNER_INT_LABELS)
        }
        int_expressions = {
            label: f"phase51_owner_int[owner * 13 + {owner_int_position[label]}]"
            for label in _OWNER_INT_LABELS
        }
        int_expressions.update(
            {
                "column:dest_topology": "phase51_land_int[destination * 3 + 0]",
                "column:destination_in_cbd": (
                    "phase51_land_int[destination * 3 + 1] < phase51_cbd_threshold"
                ),
            }
        )

        skim_bindings = {
            item.source: item for item in bindings if item.storage_kind == "skim"
        }

        def gather(source):
            if source not in skim_bindings:
                raise ValueError(f"Phase 51 required resident skim {source!r} is absent")
            binding = skim_bindings[source]
            direction = str(source[1])
            if direction in {"odt_skims", "odr_skims", "od_skims"}:
                origin, destination = "origin", "destination"
            elif direction in {"dot_skims", "dor_skims", "od_skims_reverse"}:
                origin, destination = "destination", "origin"
            else:
                raise ValueError(f"Phase 51 unsupported skim direction {direction!r}")
            data_prefix = f"skim_{binding.slot}"
            dimension_prefix = f"skim_group_{binding.skim_group}"
            if binding.skim_rank == 3:
                period = "out_period" if direction in {"odt_skims", "dor_skims"} else "in_period"
                index = (
                    f"(({origin} * {dimension_prefix}_dest_count + {destination}) * "
                    f"{dimension_prefix}_time_count + {period})"
                )
            else:
                index = f"({origin} * {dimension_prefix}_dest_count + {destination})"
            return f"{data_prefix}_data[{index}]"

        for label in sorted(_AVAILABILITY_LABELS):
            int_expressions[label] = _availability_expression(
                label, gather, "phase51_owner_int[owner * 13 + 0]"
            )
        if set(float_expressions) != set(float_labels):
            raise ValueError("Phase 51 fused floating source contract is incomplete")
        if set(int_expressions) != set(int_labels):
            missing = sorted(set(int_labels) - set(int_expressions))
            raise ValueError(
                "Phase 51 fused integer source contract is incomplete: " + ", ".join(missing)
            )

        scalar_prelude = [
            "    const long long owner = phase51_row_owner[row];",
            "    const long long origin = phase51_owner_origin[owner];",
            "    const long long out_period = phase51_owner_out_period[owner];",
            "    const long long in_period = phase51_owner_in_period[owner];",
            "    const long long destination = phase51_row_destination[row];",
            "    const int destination_band = "
            "(int)phase51_land_int[destination * 3 + 2] - 1;",
        ]
        float_target = (
            "phase51_float_values[tile_row * 10 + {position}]"
            if self.tile_rows > 1 else "phase51_float_values[{position}]"
        )
        int_target = (
            "phase51_int_values[tile_row * 31 + {position}]"
            if self.tile_rows > 1 else "phase51_int_values[{position}]"
        )
        row_thread = "row_thread" if self.tile_rows > 1 else "(int)threadIdx.x"
        scalar_prelude.extend((
            f"    if ({row_thread} < 10) {{",
            f"        switch ({row_thread}) {{",
        ))
        scalar_prelude.extend(
            f"        case {position}: {float_target.format(position=position)} = "
            f"(float)({float_expressions[label]}); break;"
            for position, label in enumerate(float_labels)
        )
        scalar_prelude.extend(
            (
                "        }",
                "    }",
                f"    if ({row_thread} < 31) {{",
                f"        switch ({row_thread}) {{",
            )
        )
        scalar_prelude.extend(
            f"        case {position}: {int_target.format(position=position)} = "
            f"(int)({int_expressions[label]}); break;"
            for position, label in enumerate(int_labels)
        )
        # Do not synchronize here.  The generated grouped-skim index prelude
        # immediately follows and does not consume these values; its barrier
        # also makes the compact values visible before feature evaluation.
        # Keeping both barriers costs one block-wide rendezvous per sampled
        # row (4.7 million at the qualification scale).
        scalar_prelude.extend(("        }", "    }"))
        if self.tile_rows > 1:
            prelude = ["    if (row < rows) {"]
            prelude.extend("    " + line for line in scalar_prelude)
            prelude.extend(
                (
                    "        if (row_thread == 0) {",
                    "            phase52_origin[tile_row] = (int)origin;",
                    "            phase52_destination[tile_row] = (int)destination;",
                    "            phase52_out_period[tile_row] = (int)out_period;",
                    "            phase52_in_period[tile_row] = (int)in_period;",
                    "        }",
                    "    }",
                    "    __syncthreads();",
                )
            )
        else:
            prelude = scalar_prelude

        float_reference_by_slot = {
            slot: float_variables[label] for slot, label in enumerate(float_labels)
        }
        int_reference_by_slot = {
            slot: int_variables[label] for slot, label in enumerate(int_labels)
        }
        row_references = {}
        for binding in bindings:
            if binding.storage_kind == "float64":
                row_references[binding.source] = float_reference_by_slot[binding.slot]
            elif binding.storage_kind == "int64":
                row_references[binding.source] = int_reference_by_slot[binding.slot]
        group_coordinates = {}
        for binding in skim_bindings.values():
            direction = str(binding.source[1])
            reverse = direction in {"dot_skims", "dor_skims", "od_skims_reverse"}
            period = (
                "out_period" if direction in {"odt_skims", "dor_skims"} else "in_period"
            ) if binding.skim_rank == 3 else None
            if self.tile_rows > 1:
                origin_ref = "phase52_destination[gather_row]" if reverse else "phase52_origin[gather_row]"
                destination_ref = "phase52_origin[gather_row]" if reverse else "phase52_destination[gather_row]"
                period_ref = (
                    "phase52_out_period[gather_row]"
                    if direction in {"odt_skims", "dor_skims"}
                    else "phase52_in_period[gather_row]"
                ) if binding.skim_rank == 3 else None
                coordinates = (origin_ref, destination_ref, period_ref)
            else:
                coordinates = (
                    "destination" if reverse else "origin",
                    "origin" if reverse else "destination",
                    period,
                )
            prior = group_coordinates.setdefault(binding.skim_group, coordinates)
            if prior != coordinates:
                raise ValueError("Phase 51 skim group mixes coordinate directions")

        extra_parameters = (
            "    const int* phase51_row_owner",
            "    const float* phase51_owner_float",
            "    const int* phase51_owner_int",
            "    const int* phase51_owner_origin",
            "    const signed char* phase51_owner_out_period",
            "    const signed char* phase51_owner_in_period",
            "    const short* phase51_owner_duration",
            "    const int* phase51_row_destination",
            "    const float* phase51_wait_table",
            "    const double* phase51_land_float",
            "    const int* phase51_land_int",
            "    int phase51_cbd_threshold",
        )
        if self.tile_rows > 1:
            block_prelude = (
                f"    __shared__ float phase51_float_values[{self.tile_rows * 10}];\n"
                f"    __shared__ int phase51_int_values[{self.tile_rows * 31}];\n"
                f"    __shared__ int phase52_origin[{self.tile_rows}];\n"
                f"    __shared__ int phase52_destination[{self.tile_rows}];\n"
                f"    __shared__ int phase52_out_period[{self.tile_rows}];\n"
                f"    __shared__ int phase52_in_period[{self.tile_rows}];"
            )
        else:
            block_prelude = (
                "    __shared__ float phase51_float_values[10];\n"
                "    __shared__ int phase51_int_values[31];"
            )
        source, source_sha256 = generate_cuda_source(
            document,
            list(bindings),
            capture_features=False,
            locality_tile_rows=self.tile_rows,
            locality_optimized=self.tile_rows > 1,
            group_skim_indices=True,
            sparse_zero_coefficients=False,
            expression_float32=True,
            fused_utility_accumulation=True,
            row_source_references=row_references,
            group_coordinate_references=group_coordinates,
            extra_kernel_parameters=extra_parameters,
            block_prelude=block_prelude,
            row_prelude="\n".join(prelude),
        )
        unresolved = (
            "float_inputs[row *",
            "int_inputs[row *",
            "skim_group_0_orig[row]",
            "skim_group_0_dest[row]",
        )
        retained = [pattern for pattern in unresolved if pattern in source]
        if retained:
            raise ValueError(f"Phase 51 fused source retains legacy reads: {retained}")
        if "const long long owner = phase51_row_owner[row]" not in source:
            raise ValueError("Phase 51 compiler did not emit compact row-owner execution")
        export_path = os.environ.get("CHOICEFORGE_PHASE52_EXPORT_CUDA")
        if self.tile_rows > 1 and export_path:
            export = Path(export_path)
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_text(source, encoding="utf-8", newline="\n")
        kernel = _FUSED_UTILITY_CACHE.get(source_sha256)
        compiled = kernel is None
        if kernel is None:
            kernel = self.cp.RawKernel(
                source,
                "choiceforge_strict_ir_v3",
                options=("--std=c++11", "--fmad=true", "--prec-div=true", "--ftz=true"),
            )
            kernel.compile()
            _FUSED_UTILITY_CACHE[source_sha256] = kernel
        return kernel, source_sha256, compiled

    def _row_owner(self, packet, rows):
        """Build a 32-bit row-to-owner execution map entirely on CUDA."""
        global _ROW_OWNER_KERNEL
        if _ROW_OWNER_KERNEL is None:
            source = r'''extern "C" __global__ void phase51_row_owner(
    const long long* offsets,
    int* row_owner,
    int owners) {
  const int owner = (int)blockIdx.x;
  if (owner >= owners) return;
  for (long long row = offsets[owner] + threadIdx.x;
       row < offsets[owner + 1]; row += blockDim.x) {
    row_owner[row] = owner;
  }
}'''
            _ROW_OWNER_KERNEL = self.cp.RawKernel(
                source, "phase51_row_owner", options=("--std=c++11",)
            )
            _ROW_OWNER_KERNEL.compile()
        key = ("row_owner", (), np.dtype(np.int32).str)
        buffer = self._device_buffers.get(key) if self.persistent else None
        hit = buffer is not None and buffer.shape[0] >= rows
        if not hit:
            buffer = self.cp.empty(
                self._capacity(rows) if self.persistent else rows,
                dtype=self.cp.int32,
            )
            if self.persistent:
                self._device_buffers[key] = buffer
        row_owner = buffer[:rows]
        _ROW_OWNER_KERNEL(
            (packet.owners,),
            (32,),
            (packet.offsets, row_owner, np.int32(packet.owners)),
        )
        return row_owner, hit

    def compute(
        self,
        state,
        choosers,
        tour_purpose,
        logsum_settings,
        model_settings,
        network_los,
        chunk_size,
        chunk_tag,
        trace_label,
        in_period_col=None,
        out_period_col=None,
        duration_col=None,
    ):
        """ActivitySim-compatible preprocessor replacement for Phase 50."""
        from activitysim.core import config, simulate
        from activitysim.core.configuration.logit import (
            TourLocationComponentSettings,
            TourModeComponentSettings,
        )
        from .cuda_skims import cuda_cube_from_activitysim

        if chunk_size:
            raise ValueError("Phase 50 requires the qualified unchunked public contract")
        if isinstance(model_settings, dict):
            model_settings = TourLocationComponentSettings.model_validate(model_settings)
        if isinstance(logsum_settings, dict):
            logsum_settings = TourModeComponentSettings.model_validate(logsum_settings)
        started = time.perf_counter()
        semantic_key = (str(logsum_settings.SPEC), str(tour_purpose))
        semantic = self._semantic_plan_cache.get(semantic_key) if self.persistent else None
        semantic_plan_hit = semantic is not None
        if semantic is None:
            spec = state.filesystem.read_model_spec(file_name=logsum_settings.SPEC)
            coefficients = state.filesystem.get_segment_coefficients(
                logsum_settings, tour_purpose
            )
            spec = simulate.eval_coefficients(state, spec, coefficients, estimator=None)
            numeric_nest = config.get_logit_model_settings(logsum_settings)
            numeric_nest = simulate.eval_nest_coefficients(
                numeric_nest, coefficients, trace_label
            )
            constants = config.get_model_constants(logsum_settings)
            scalar_environment = state.get_global_constants().copy()
            scalar_environment.update(constants)
            scalar_environment.update(coefficients)
            document = specification_ir(spec.reset_index())
            semantic = (numeric_nest, constants, scalar_environment, document)
            if self.persistent:
                self._semantic_plan_cache[semantic_key] = semantic
        else:
            numeric_nest, constants, scalar_environment, document = semantic

        skim_dict = network_los.get_default_skim_dict()
        origin_name = model_settings.CHOOSER_ORIG_COL_NAME
        destination_name = model_settings.ALT_DEST_COL_NAME
        skims = {
            "odt_skims": skim_dict.wrap_3d(origin_name, destination_name, "out_period"),
            "dot_skims": skim_dict.wrap_3d(destination_name, origin_name, "in_period"),
            "odr_skims": skim_dict.wrap_3d(origin_name, destination_name, "in_period"),
            "dor_skims": skim_dict.wrap_3d(destination_name, origin_name, "out_period"),
            "od_skims": skim_dict.wrap(origin_name, destination_name),
        }

        def cube_loader(source):
            _, direction, key = source
            wrapper_name = "od_skims" if direction == "od_skims_reverse" else direction
            if wrapper_name not in skims:
                raise ValueError(f"Phase 50 skim direction {direction!r} is absent")
            data, dest_count, time_count, rank = cuda_cube_from_activitysim(
                skims[wrapper_name], key
            )
            return NativeSkimCube(data, dest_count, time_count, rank)

        scalar_signature = hashlib.sha256(
            json.dumps(
                sorted(
                    (str(key), type(value).__name__, repr(value))
                    for key, value in scalar_environment.items()
                    if np.isscalar(value)
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        native_key = f"{document['sha256']}:{scalar_signature}:{id(network_los)}"
        native = self._native_plan_cache.get(native_key) if self.persistent else None
        native_plan_hit = native is not None
        if native is None:
            native = compile_native_strict_abi(
                document,
                scalar_environment,
                cube_loader,
                rows=len(choosers),
                minimal_row_state=self.fused,
                minimal_output_state=self.persistent,
                cache_codegen=True,
                compile_kernel=not self.fused,
            )
            if self.persistent:
                template_invocation = replace(
                    native.invocation,
                    utilities=self.cp.empty(
                        (1, native.invocation.alternatives), dtype=self.cp.float32
                    ),
                    rows=1,
                    grid=(1,),
                )
                self._native_plan_cache[native_key] = NativeStrictAbiPlan(
                    template_invocation, native.bindings, dict(native.manifest)
                )
        if self.persistent:
            utilities, utility_workspace_hit = self._utilities(
                len(choosers), native.invocation.alternatives
            )
            native = NativeStrictAbiPlan(
                replace(
                    native.invocation,
                    utilities=utilities,
                    rows=len(choosers),
                    grid=((len(choosers) + self.tile_rows - 1) // self.tile_rows,),
                ),
                native.bindings,
                {**native.manifest, "template_cache_hit": native_plan_hit},
            )
        else:
            utility_workspace_hit = False
        if self.fused:
            fused_kernel, schema_sha256, fused_compiled = self._fused_utility(
                document, native
            )
        else:
            fused_kernel = None
            schema_sha256 = None
            fused_compiled = False
        compiled = time.perf_counter()
        land_use = state.get_dataframe("land_use")
        land_float, land_int = self._resident_land(land_use)
        packet = self._packet(
            state,
            choosers,
            land_use,
            constants,
            model_settings,
            network_los,
            tour_purpose,
            in_period_col=in_period_col,
            out_period_col=out_period_col,
            duration_col=duration_col,
        )
        prepared = time.perf_counter()
        row_owner = None
        row_owner_workspace_hit = False
        row_owner_complete = prepared
        if self.fused:
            row_owner, row_owner_workspace_hit = self._row_owner(packet, len(choosers))
            self.cp.cuda.Stream.null.synchronize()
            row_owner_complete = time.perf_counter()
        compact_arguments = (
            row_owner if self.fused else packet.offsets,
            packet.owner_float,
            packet.owner_int,
            packet.owner_origin,
            packet.owner_out_period,
            packet.owner_in_period,
            packet.owner_duration,
            packet.row_destination,
            packet.wait_table,
            land_float,
            land_int,
            np.int32(self.cbd_threshold),
        )
        if self.fused:
            generated = prepared
            fused_kernel(
                ((len(choosers) + self.tile_rows - 1) // self.tile_rows,),
                (256,),
                (
                    native.invocation.float_inputs,
                    native.invocation.int_inputs,
                    native.invocation.float_scalars,
                    native.invocation.int_scalars,
                    native.invocation.coefficients,
                    native.invocation.features,
                    native.invocation.utilities,
                    np.int64(len(choosers)),
                ) + native.invocation.skim_arguments + compact_arguments,
                shared_mem=_shared_memory_bytes(
                    native.invocation.terms,
                    native.invocation.logical_skim_bindings,
                    len(set(native.invocation.skim_input_groups)),
                    self.tile_rows,
                    self.tile_rows > 1,
                    self.tile_rows == 1,
                ),
            )
            utilities = native.invocation.utilities
            generator_compiled = False
        else:
            kernel, generator_arguments, schema_sha256, generator_compiled = self._generator(
                native.invocation
            )
            kernel(
                (packet.owners,),
                (32,),
                (
                    native.invocation.float_inputs,
                    native.invocation.int_inputs,
                ) + compact_arguments + (
                    np.int32(packet.owners),
                    np.int32(native.invocation.float_inputs.shape[1]),
                    np.int32(native.invocation.int_inputs.shape[1]),
                ) + generator_arguments,
            )
            self.cp.cuda.Stream.null.synchronize()
            generated = time.perf_counter()
            utilities = native.invocation.execute()
        self.cp.cuda.Stream.null.synchronize()
        utility_complete = time.perf_counter()
        logsums = mtc21_nested_logsums_cuda(
            utilities,
            numeric_nest,
            document["alternatives"],
            return_device=True,
            numeric_policy="activitysim_pandas_float64",
        )
        self.cp.cuda.Stream.null.synchronize()
        nested_complete = time.perf_counter()
        metadata = {
            "trace_label": str(trace_label),
            "chooser_ids": np.asarray(choosers.index, dtype=np.int64),
        }
        self.bridge.publish(logsums, metadata, host_materialized=False)
        coordinate_columns = sum(
            2 + int(rank == 3)
            for group, rank in {
                group: rank
                for group, rank in zip(
                    native.invocation.skim_input_groups,
                    native.invocation.skim_input_ranks,
                )
            }.items()
        )
        dense_bytes = int(
            len(choosers)
            * (
                len(native.invocation.float_input_sources) * 4
                + len(native.invocation.int_input_sources) * 8
                + coordinate_columns * 8
            )
        )
        minimal_bootstrap_bytes = int(
            native.invocation.float_inputs.nbytes
            + native.invocation.int_inputs.nbytes
            + native.invocation.skim_coordinate_bytes
        )
        self._events.append(
            {
                "phase": 52 if self.persistent else (51 if self.fused else 50),
                "trace_label": str(trace_label),
                "rows": int(len(choosers)),
                "owners": int(packet.owners),
                "terms": int(native.invocation.terms),
                "alternatives": int(native.invocation.alternatives),
                "float_row_sources": len(native.invocation.float_input_sources),
                "int_row_sources": len(native.invocation.int_input_sources),
                "skim_coordinate_groups": len(set(native.invocation.skim_input_groups)),
                "dense_preprocessor_rows_avoided": int(len(choosers)),
                "dense_preprocessor_values_avoided": int(len(choosers) * 41),
                "dense_host_pack_bytes_avoided": dense_bytes,
                "compact_upload_bytes": int(packet.compact_bytes),
                "net_upload_bytes_avoided": dense_bytes - int(packet.compact_bytes),
                "binding_resolution_calls": 0,
                "host_dense_pack_calls": 0,
                "fallback_used": False,
                "schema_sha256": schema_sha256,
                "native_abi_sha256": native.manifest["schema_sha256"],
                "native_codegen_cache_hit": native.manifest["codegen_cache_hit"],
                "native_kernel_compiled": native.manifest["compiled_this_call"],
                "semantic_plan_cache_hit": semantic_plan_hit,
                "native_plan_cache_hit": native_plan_hit,
                "utility_workspace_hit": utility_workspace_hit,
                "packet_workspace_hits": packet.workspace_hits,
                "packet_workspace_allocations": packet.workspace_allocations,
                "row_owner_workspace_hit": row_owner_workspace_hit,
                "tile_rows": self.tile_rows,
                "generator_compiled": bool(generator_compiled),
                "fused_kernel_compiled": bool(fused_compiled),
                "dense_device_abi_bytes_eliminated": dense_bytes if self.fused else 0,
                "minimal_bootstrap_bytes": minimal_bootstrap_bytes,
                "row_owner_device_bytes": (
                    int(row_owner.nbytes) if row_owner is not None else 0
                ),
                "row_owner_kernel_seconds": row_owner_complete - prepared,
                "compile_seconds": compiled - started,
                "compact_prepare_seconds": prepared - compiled,
                "device_generate_seconds": generated - prepared,
                "utility_kernel_seconds": (
                    utility_complete - row_owner_complete
                    if self.fused else utility_complete - generated
                ),
                "fused_kernel_seconds": (
                    utility_complete - row_owner_complete if self.fused else 0.0
                ),
                "nested_kernel_seconds": nested_complete - utility_complete,
                "total_seconds": nested_complete - started,
            }
        )
        return pd.Series(np.zeros(len(choosers), dtype=np.float64), index=choosers.index)

    def summary(self) -> dict:
        events = list(self._events)
        return {
            "contract_version": self.version,
            "calls": len(events),
            "rows": int(sum(item["rows"] for item in events)),
            "owners": int(sum(item["owners"] for item in events)),
            "dense_preprocessor_rows_avoided": int(
                sum(item["dense_preprocessor_rows_avoided"] for item in events)
            ),
            "dense_preprocessor_values_avoided": int(
                sum(item["dense_preprocessor_values_avoided"] for item in events)
            ),
            "dense_host_pack_bytes_avoided": int(
                sum(item["dense_host_pack_bytes_avoided"] for item in events)
            ),
            "compact_upload_bytes": int(sum(item["compact_upload_bytes"] for item in events)),
            "net_upload_bytes_avoided": int(
                sum(item["net_upload_bytes_avoided"] for item in events)
            ),
            "binding_resolution_calls": int(
                sum(item["binding_resolution_calls"] for item in events)
            ),
            "host_dense_pack_calls": int(sum(item["host_dense_pack_calls"] for item in events)),
            "fallback_calls": int(sum(bool(item["fallback_used"]) for item in events)),
            "fused_calls": int(sum(item.get("phase") in {51, 52} for item in events)),
            "phase52_calls": int(sum(item.get("phase") == 52 for item in events)),
            "semantic_plan_cache_hits": int(
                sum(bool(item.get("semantic_plan_cache_hit")) for item in events)
            ),
            "native_plan_cache_hits": int(
                sum(bool(item.get("native_plan_cache_hit")) for item in events)
            ),
            "utility_workspace_hits": int(
                sum(bool(item.get("utility_workspace_hit")) for item in events)
            ),
            "packet_workspace_hits": int(
                sum(item.get("packet_workspace_hits", 0) for item in events)
            ),
            "packet_workspace_allocations": int(
                sum(item.get("packet_workspace_allocations", 0) for item in events)
            ),
            "row_owner_workspace_hits": int(
                sum(bool(item.get("row_owner_workspace_hit")) for item in events)
            ),
            "tile_rows": sorted({int(item.get("tile_rows", 1)) for item in events}),
            "dense_device_abi_bytes_eliminated": int(
                sum(item.get("dense_device_abi_bytes_eliminated", 0) for item in events)
            ),
            "minimal_bootstrap_bytes": int(
                sum(item.get("minimal_bootstrap_bytes", 0) for item in events)
            ),
            "row_owner_device_bytes": int(
                sum(item.get("row_owner_device_bytes", 0) for item in events)
            ),
            "row_owner_kernel_seconds": float(
                sum(item.get("row_owner_kernel_seconds", 0.0) for item in events)
            ),
            "device_generate_seconds": float(
                sum(item["device_generate_seconds"] for item in events)
            ),
            "utility_kernel_seconds": float(
                sum(item["utility_kernel_seconds"] for item in events)
            ),
            "fused_kernel_seconds": float(
                sum(item.get("fused_kernel_seconds", 0.0) for item in events)
            ),
            "total_seconds": float(sum(item["total_seconds"] for item in events)),
            "all_source_abis_exact": bool(
                events
                and all(
                    item["float_row_sources"] == 10
                    and item["int_row_sources"] == 31
                    and item["skim_coordinate_groups"] == 6
                    for item in events
                )
            ),
            "all_dense_device_abis_eliminated": bool(
                events
                and all(
                    item.get("phase") in {51, 52}
                    and item.get("dense_device_abi_bytes_eliminated", 0) > 0
                    for item in events
                )
            ),
            "events": events,
        }

    def release(self) -> None:
        """Drop every resident device reference after the final GPU consumer."""
        self._native_plan_cache.clear()
        self._semantic_plan_cache.clear()
        self._device_buffers.clear()
        self._utility_buffer = None
        self._land_float = None
        self._land_int = None
        self._land_signature = None


class FusedDestinationInputSupergraph(DestinationInputSupergraph):
    """Phase 51 compact-source strict utility runtime."""

    version = 2

    def __init__(self, bridge, *, cbd_threshold: int, cp=None):
        super().__init__(bridge, cbd_threshold=cbd_threshold, cp=cp, fused=True)


class PersistentTiledDestinationInputSupergraph(DestinationInputSupergraph):
    """Phase 52 prewarmed, workspace-reusing compact destination runtime."""

    version = 3

    def __init__(self, bridge, *, cbd_threshold: int, cp=None, tile_rows: int | None = None):
        if tile_rows is None:
            tile_rows = int(os.environ.get("CHOICEFORGE_PHASE52_TILE_ROWS", "4"))
        super().__init__(
            bridge,
            cbd_threshold=cbd_threshold,
            cp=cp,
            fused=True,
            tile_rows=tile_rows,
            persistent=True,
        )
