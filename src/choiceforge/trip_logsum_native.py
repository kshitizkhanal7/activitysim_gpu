"""Native GPU runtime for trip-destination mode-choice logsums.

The legacy path expands a large pandas preprocessor table before evaluating
the 21-alternative trip-mode utility. Phases 36 and 37 replaced it with a
compact raw packet and then fused ABI preparation with utility evaluation.
Phase 38 additionally normalizes facts repeated for every sampled destination:
each row carries only its coordinates and directional-state selector, while
stable trip/tour/person/household facts and controlled waits are stored once
per trip direction. Unknown layouts, changed stable facts, and changed waits
fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

import numpy as np
import pandas as pd

from .cuda_backend import _cupy


_AVAILABILITY_KERNEL_CACHE = {}
_DEVICE_PREPARATION_KERNEL_CACHE = {}
_FUSED_UTILITY_KERNEL_CACHE = {}
_RESIDENT_LAND_CACHE = {}
_NORMALIZED_DEVICE_WORKSPACE = {}


_RAW_INT_COLUMNS = {
    name: position
    for position, name in enumerate(
        (
            "origin", "destination", "period", "outbound",
            "first", "last", "free_parking", "tour_mode", "parent_mode",
            "is_atwork_subtour", "auto_ownership", "age", "participants",
            "hhsize",
        )
    )
}
_RAW_FLOAT_COLUMNS = {"duration": 0, "value_of_time": 1}
_ROW_COORD_COLUMNS = {"origin": 0, "destination": 1}
_STABLE_INT_COLUMNS = {
    name: position
    for position, name in enumerate(
        name for name in _RAW_INT_COLUMNS if name not in _ROW_COORD_COLUMNS
    )
}


def _equal_with_missing(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        equal = np.asarray(actual == expected, dtype=bool)
    except (TypeError, ValueError):
        equal = np.zeros(actual.shape, dtype=bool)
    return equal | (np.asarray(pd.isna(actual)) & np.asarray(pd.isna(expected)))


def _normalized_row_layout(frame, draws, stable_columns):
    """Return directional state representatives and fail-closed row selectors."""
    rows = len(frame)
    draws = np.asarray(draws)
    if rows == 0 or rows % 2:
        raise ValueError("Phase 38 requires a nonempty paired directional frame")
    if draws.shape != (rows, 3):
        raise ValueError("Phase 38 requires exactly three draws for every row")
    half = rows // 2
    row_ids = np.asarray(frame.index)
    if not _equal_with_missing(row_ids[:half], row_ids[half:]).all():
        raise ValueError("Phase 38 directional frames have different trip row order")
    _, first_indices, first_selectors = np.unique(
        row_ids[:half], return_index=True, return_inverse=True
    )
    state_rows_per_direction = len(first_indices)
    if state_rows_per_direction > np.iinfo(np.int32).max:
        raise ValueError("Phase 38 normalized state exceeds int32 selectors")
    state_first_indices = np.r_[first_indices, first_indices + half]
    state_selectors = np.r_[
        first_selectors, first_selectors + state_rows_per_direction
    ].astype(np.int32, copy=False)

    # These values originate at trip/tour/person/household grain. A duplicate
    # trip row that changes any of them would make normalization unsafe.
    trip_selectors = np.r_[first_selectors, first_selectors]
    for column in stable_columns:
        values = np.asarray(frame[column])
        representative = values[first_indices]
        if not _equal_with_missing(values, representative[trip_selectors]).all():
            raise ValueError(
                f"Phase 38 stable column {column!r} varies within a trip"
            )
    representative_draws = draws[state_first_indices]
    if not _equal_with_missing(draws, representative_draws[state_selectors]).all():
        raise ValueError("Phase 38 controlled wait draws vary within directional state")
    return state_first_indices, state_selectors, state_rows_per_direction


def _upload_normalized(cp, name, host):
    """Upload into a process-resident, grow-only device workspace."""
    host = np.ascontiguousarray(host)
    trailing_shape = tuple(host.shape[1:])
    key = (name, host.dtype.str, trailing_shape)
    cached = _NORMALIZED_DEVICE_WORKSPACE.get(key)
    hit = cached is not None and cached.shape[0] >= host.shape[0]
    if not hit:
        cached = cp.empty(host.shape, dtype=host.dtype)
        _NORMALIZED_DEVICE_WORKSPACE[key] = cached
    view = cached[: host.shape[0]]
    view.set(host)
    return view, hit


def clear_trip_device_state_cache():
    """Drop resident land aliases and Phase 38 normalized workspaces."""
    _RESIDENT_LAND_CACHE.clear()
    _NORMALIZED_DEVICE_WORKSPACE.clear()

_AVAILABILITY = {
    "name:sov_available": ("SOV_TIME", 0, None),
    "name:sovtoll_available": ("SOVTOLL_VTOLL", 0, None),
    "name:hov2_available": ("HOV2_TIME", 0, None),
    "name:hov2toll_available": ("HOV2TOLL_VTOLL", 0, None),
    "name:hov3_available": ("HOV3_TIME", 0, None),
    "name:hov3toll_available": ("HOV3TOLL_VTOLL", 0, None),
    "name:walk_local_available": ("WLK_LOC_WLK_TOTIVT", 0, None),
    "name:walk_lrf_available": ("WLK_LRF_WLK_KEYIVT", 0, 10),
    "name:walk_express_available": ("WLK_EXP_WLK_KEYIVT", 0, 11),
    "name:walk_heavyrail_available": ("WLK_HVY_WLK_KEYIVT", 0, 12),
    "name:walk_commuter_available": ("WLK_COM_WLK_KEYIVT", 0, 13),
    "name:drive_local_available_outbound": ("DRV_LOC_WLK_TOTIVT", 1, None),
    "name:drive_local_available_inbound": ("WLK_LOC_DRV_TOTIVT", 2, None),
    "name:drive_lrf_available_outbound": ("DRV_LRF_WLK_KEYIVT", 1, 15),
    "name:drive_lrf_available_inbound": ("WLK_LRF_DRV_KEYIVT", 2, 15),
    "name:drive_express_available_outbound": ("DRV_EXP_WLK_KEYIVT", 1, 16),
    "name:drive_express_available_inbound": ("WLK_EXP_DRV_KEYIVT", 2, 16),
    "name:drive_heavyrail_available_outbound": ("DRV_HVY_WLK_KEYIVT", 1, 17),
    "name:drive_heavyrail_available_inbound": ("WLK_HVY_DRV_KEYIVT", 2, 17),
    "name:drive_commuter_available_outbound": ("DRV_COM_WLK_KEYIVT", 1, 18),
    "name:drive_commuter_available_inbound": ("WLK_COM_DRV_KEYIVT", 2, 18),
}


def _values(frame, name, dtype=None):
    result = frame[name].to_numpy(copy=False)
    return np.asarray(result, dtype=dtype) if dtype is not None else np.asarray(result)


def _land(land_use, zones, column):
    values = land_use[column].reindex(zones).to_numpy()
    if np.any(np.asarray(values) != np.asarray(values)):
        raise ValueError(f"trip native land-use lookup {column!r} contains missing zones")
    return np.asarray(values)


def _period(values):
    array = np.asarray(values)
    if array.dtype.kind in "iu":
        result = array.astype(np.int64, copy=False)
        if result.size and (result.min() < 0 or result.max() > 4):
            raise ValueError("trip native period index is outside 0..4")
        return result
    mapping = {"EA": 0, "AM": 1, "MD": 2, "PM": 3, "EV": 4}
    try:
        return np.asarray([mapping[str(item)] for item in array], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"trip native period is unsupported: {exc.args[0]!r}") from exc


def _density_band(land_use, zones):
    measure = (
        _land(land_use, zones, "TOTPOP") + _land(land_use, zones, "TOTEMP")
    ) / (_land(land_use, zones, "TOTACRE") / 640.0)
    return np.select(
        [measure <= 500, measure <= 2000, measure <= 5000, measure <= 15000],
        [5, 4, 3, 2],
        default=1,
    ).astype(np.int64)


def _mapped(mapping, bands):
    return np.asarray([mapping[int(item)] for item in bands], dtype=np.float64)


def _wait(draw, mean, sd, lower, upper):
    x = 1.0 + (sd * sd) / (mean * mean)
    mu = np.log(mean / np.sqrt(x))
    sigma = np.sqrt(np.log(x))
    return np.exp(np.asarray(draw, dtype=np.float64) * sigma + mu).clip(lower, upper)


def _source_text(source):
    label = ":".join(map(str, source))
    aliases = {
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
    return aliases.get(label, label)


@dataclass
class TripLogsumNativeTelemetry:
    rows: int
    compact_host_bytes: int
    host_build_seconds: float
    upload_seconds: float
    availability_kernel_seconds: float
    utility_kernel_seconds: float
    backend: str = "phase35_host_abi"
    device_preparation_kernel_seconds: float = 0.0
    compact_device_input_bytes: int = 0
    dense_host_abi_bytes_avoided: int = 0
    resident_land_bytes: int = 0
    dense_device_abi_bytes_eliminated: int = 0
    coordinate_device_bytes_eliminated: int = 0
    fused_kernel_seconds: float = 0.0
    minimal_bootstrap_bytes: int = 0
    normalized_trip_rows: int = 0
    normalized_state_rows: int = 0
    normalized_row_bytes: int = 0
    normalized_state_bytes: int = 0
    phase37_compact_bytes_eliminated: int = 0
    resident_workspace_hits: int = 0
    resident_workspace_arrays: int = 0
    normalized_contract_valid: bool = False


class TripLogsumNativePlan:
    """Populate and execute one strict trip-mode utility invocation."""

    def __init__(self, invocation, *, document=None, bindings=None):
        self.invocation = invocation
        self.document = document
        self.bindings = tuple(bindings) if bindings is not None else None
        self.cp = _cupy()
        self.float_labels = tuple(map(_source_text, invocation.float_input_sources))
        self.int_labels = tuple(map(_source_text, invocation.int_input_sources))
        expected_float = {
            "column:total_terminal_time", "column:ivot", "column:total_parking_cost",
            "column:density_index", "column:origin_walk_time",
            "column:destination_walk_time", "column:origTaxiWaitTime",
            "column:origSingleTNCWaitTime", "column:origSharedTNCWaitTime",
            "column:i_tour_mode", "column:origin_density_index",
        }
        expected_int = {
            "name:sov_available", "name:auto_ownership", "name:age", "name:is_joint",
            "name:is_atwork_subtour", "name:work_tour_is_SOV", "name:sovtoll_available",
            "name:hov2_available", "name:number_of_participants", "column:hhsize",
            "name:hov2toll_available", "name:hov3_available", "name:hov3toll_available",
            "column:trip_topology", "name:work_tour_is_bike",
            "name:walk_local_available", "name:walk_lrf_available",
            "name:walk_express_available", "name:walk_heavyrail_available",
            "name:walk_commuter_available", "name:outbound",
            "name:drive_local_available_outbound", "name:inbound",
            "name:drive_local_available_inbound", "name:drive_lrf_available_outbound",
            "name:drive_lrf_available_inbound", "name:drive_express_available_outbound",
            "name:drive_express_available_inbound", "name:drive_heavyrail_available_outbound",
            "name:drive_heavyrail_available_inbound", "name:drive_commuter_available_outbound",
            "name:drive_commuter_available_inbound", "name:tour_mode_is_auto",
            "name:tour_mode_is_walk", "name:tour_mode_is_bike",
            "name:tour_mode_is_walk_transit", "name:tour_mode_is_drive_transit",
            "name:tour_mode_is_ride_hail", "column:is_indiv", "column:tour_mode_is_SOV",
            "column:tour_mode_is_SR2", "column:tour_mode_is_SR3P",
            "column:walk_ferry_available", "column:drive_ferry_available",
            "column:first_trip",
        }
        if set(self.float_labels) != expected_float or set(self.int_labels) != expected_int:
            raise ValueError("trip native input ABI differs from the reviewed 11/45 contract")
        self._availability_kernel, self._availability_args = self._compile_availability()
        self._device_preparation_kernel = None
        self._device_preparation_args = None
        self._coordinate_args = None
        self._skim_dest_count = None
        self._skim_time_count = None
        self._fused_utility_kernel = None
        self._fused_extra_args = None
        self._normalized_utility_kernel = None
        self._normalized_extra_args = None

    def _skim_argument(self, key):
        wanted = ("skim", "odt_skims", key)
        try:
            position = self.invocation.skim_input_sources.index(wanted)
        except ValueError as exc:
            raise ValueError(f"trip native required skim {key!r} is absent") from exc
        return self.invocation.skim_arguments[position]

    def _compile_availability(self):
        selected = []
        assignments = []
        int_pos = {label: pos for pos, label in enumerate(self.int_labels)}
        mode_pos = self.float_labels.index("column:i_tour_mode")
        auto_pos = int_pos["name:auto_ownership"]
        outbound_pos = int_pos["name:outbound"]
        for label, (key, direction, threshold) in _AVAILABILITY.items():
            if key not in selected:
                selected.append(key)
            skim_pos = selected.index(key)
            conditions = [f"skim_{skim_pos}[skim_index] > 0.0f"]
            if direction == 0 and label.startswith("name:walk_"):
                pass
            elif direction == 1:
                conditions.extend((f"ints[ib + {auto_pos}] > 0", f"ints[ib + {outbound_pos}] != 0"))
            elif direction == 2:
                conditions.extend((f"ints[ib + {auto_pos}] > 0", f"ints[ib + {outbound_pos}] == 0"))
            if threshold is not None:
                conditions.append(f"floats[fb + {mode_pos}] >= {float(threshold):.1f}f")
            assignments.append(
                f"  ints[ib + {int_pos[label]}] = ({' && '.join(conditions)}) ? 1LL : 0LL;"
            )
        # Ferry availability includes its parent LRF condition.
        walk_ferry = int_pos["column:walk_ferry_available"]
        drive_ferry = int_pos["column:drive_ferry_available"]
        walk_lrf = int_pos["name:walk_lrf_available"]
        drive_out = int_pos["name:drive_lrf_available_outbound"]
        drive_in = int_pos["name:drive_lrf_available_inbound"]
        for key in ("WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT", "WLK_LRF_DRV_FERRYIVT"):
            if key not in selected:
                selected.append(key)
        wf, dfo, dfi = (selected.index(key) for key in (
            "WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT", "WLK_LRF_DRV_FERRYIVT"
        ))
        assignments.extend((
            f"  ints[ib + {walk_ferry}] = (ints[ib + {walk_lrf}] && skim_{wf}[skim_index] > 0.0f) ? 1LL : 0LL;",
            f"  ints[ib + {drive_ferry}] = ((ints[ib + {outbound_pos}] && ints[ib + {drive_out}] && skim_{dfo}[skim_index] > 0.0f) || (!ints[ib + {outbound_pos}] && ints[ib + {drive_in}] && skim_{dfi}[skim_index] > 0.0f)) ? 1LL : 0LL;",
        ))
        parameters = ",\n".join(f"  const float* skim_{i}" for i in range(len(selected)))
        source = f'''extern "C" __global__ void trip_native_availability(
  float* floats, long long* ints, const long long* origin,
  const long long* destination, const long long* period,
  long long rows, int float_columns, int int_columns,
  long long dest_count, long long time_count,
{parameters}) {{
  const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= rows) return;
  const long long fb = row * float_columns;
  const long long ib = row * int_columns;
  const long long skim_index = (origin[row] * dest_count + destination[row]) * time_count + period[row];
{chr(10).join(assignments)}
}}'''
        kernel = _AVAILABILITY_KERNEL_CACHE.get(source)
        if kernel is None:
            kernel = self.cp.RawKernel(source, "trip_native_availability", options=("--std=c++11", "--fmad=false"))
            kernel.compile()
            _AVAILABILITY_KERNEL_CACHE[source] = kernel
        return kernel, tuple(self._skim_argument(key) for key in selected)

    def _coordinate_contract(self):
        """Return every grouped coordinate buffer and its declared direction."""
        args = list(self.invocation.skim_arguments)
        position = self.invocation.logical_skim_bindings
        groups = self.invocation.skim_input_groups
        sources = self.invocation.skim_input_sources
        ranks = self.invocation.skim_input_ranks
        result = []
        odt_dimensions = None
        for group in sorted(set(groups)):
            representative = groups.index(group)
            direction = sources[representative][1]
            rank = ranks[representative]
            if direction not in {
                "odt_skims", "dot_skims", "od_skims", "od_skims_reverse"
            }:
                raise ValueError(
                    f"trip native coordinate direction {direction!r} is unsupported"
                )
            origin = args[position]
            destination = args[position + 1]
            position += 2
            period = None
            if rank == 3:
                period = args[position]
                position += 1
            dest_count = int(args[position])
            position += 1
            time_count = 1
            if rank == 3:
                time_count = int(args[position])
                position += 1
            result.append(
                (direction, rank, origin, destination, period, dest_count, time_count)
            )
            if direction == "odt_skims":
                odt_dimensions = (dest_count, time_count)
        if position != len(args) or odt_dimensions is None:
            raise ValueError("trip native grouped skim ABI is malformed")
        return tuple(result), odt_dimensions

    @staticmethod
    def _mode_mask(values):
        mask = 0
        for value in values:
            number = int(value)
            if number < 0 or number >= 63:
                raise ValueError("trip native mode index is outside mask range 0..62")
            mask |= 1 << number
        return np.uint64(mask)

    def _compile_device_preparation(self):
        """Compile the Phase 36 compact raw-state to complete ABI kernel."""
        float_pos = {label: pos for pos, label in enumerate(self.float_labels)}
        int_pos = {label: pos for pos, label in enumerate(self.int_labels)}
        ri = _RAW_INT_COLUMNS
        rf = _RAW_FLOAT_COLUMNS
        assigned_float = set()
        assigned_int = set()
        statements = []

        def assign_float(label, expression):
            assigned_float.add(label)
            statements.append(f"  floats[fb + {float_pos[label]}] = (float)({expression});")

        def assign_int(label, expression):
            assigned_int.add(label)
            statements.append(f"  ints[ib + {int_pos[label]}] = ({expression}) ? 1LL : 0LL;")

        assign_float(
            "column:total_terminal_time",
            "((outbound && first) ? 0.0 : terminal[origin]) + "
            "((!outbound && last) ? 0.0 : terminal[destination])",
        )
        assign_float("column:ivot", f"1.0 / raw_floats[rfd + {rf['value_of_time']}]")
        assign_float(
            "column:total_parking_cost",
            "(origin_duration * parking[origin] + destination_duration * "
            "parking[destination]) / 2.0",
        )
        assign_float(
            "column:density_index",
            "outbound ? density[destination] : density[origin]",
        )
        assign_float("column:origin_walk_time", "walk_time")
        assign_float("column:destination_walk_time", "walk_time")
        assign_float("column:origTaxiWaitTime", "waits[row * 3]")
        assign_float("column:origSingleTNCWaitTime", "waits[row * 3 + 1]")
        assign_float("column:origSharedTNCWaitTime", "waits[row * 3 + 2]")
        assign_float("column:i_tour_mode", "mode")
        assign_float(
            "column:origin_density_index",
            "outbound ? density[origin] : density[destination]",
        )

        value_ints = {
            "name:auto_ownership": f"raw_ints[rib + {ri['auto_ownership']}]",
            "name:age": f"raw_ints[rib + {ri['age']}]",
            "name:is_joint": f"raw_ints[rib + {ri['participants']}] > 1",
            "name:is_atwork_subtour": f"raw_ints[rib + {ri['is_atwork_subtour']}] != 0",
            "name:work_tour_is_SOV": f"raw_ints[rib + {ri['parent_mode']}] == 1",
            "name:number_of_participants": f"raw_ints[rib + {ri['participants']}]",
            "column:hhsize": f"raw_ints[rib + {ri['hhsize']}]",
            "column:trip_topology": "outbound ? topology[destination] : topology[origin]",
            "name:work_tour_is_bike": f"raw_ints[rib + {ri['parent_mode']}] == 2",
            "name:outbound": "outbound",
            "name:inbound": "!outbound",
            "name:tour_mode_is_auto": "((auto_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_walk": "mode == walk_mode",
            "name:tour_mode_is_bike": "mode == bike_mode",
            "name:tour_mode_is_walk_transit": "((walk_transit_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_drive_transit": "((drive_transit_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_ride_hail": "((ride_hail_modes >> mode) & 1ULL) != 0ULL",
            "column:is_indiv": f"raw_ints[rib + {ri['participants']}] == 1",
            "column:tour_mode_is_SOV": "((sov_modes >> mode) & 1ULL) != 0ULL",
            "column:tour_mode_is_SR2": "((sr2_modes >> mode) & 1ULL) != 0ULL",
            "column:tour_mode_is_SR3P": "((sr3_modes >> mode) & 1ULL) != 0ULL",
            "column:first_trip": "first",
        }
        integer_value_labels = {
            "name:auto_ownership", "name:age", "name:number_of_participants",
            "column:hhsize", "column:trip_topology",
        }
        for label, expression in value_ints.items():
            assigned_int.add(label)
            suffix = "" if label in integer_value_labels else " ? 1LL : 0LL"
            statements.append(
                f"  ints[ib + {int_pos[label]}] = (long long)({expression}){suffix};"
            )

        selected = []
        for label, (key, direction, threshold) in _AVAILABILITY.items():
            if key not in selected:
                selected.append(key)
            skim_pos = selected.index(key)
            conditions = [f"skim_{skim_pos}[skim_index] > 0.0f"]
            if direction == 1:
                conditions.extend(("auto_ownership > 0", "outbound"))
            elif direction == 2:
                conditions.extend(("auto_ownership > 0", "!outbound"))
            if threshold is not None:
                conditions.append(f"mode >= {int(threshold)}")
            assign_int(label, " && ".join(conditions))
        for key in (
            "WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT",
            "WLK_LRF_DRV_FERRYIVT",
        ):
            if key not in selected:
                selected.append(key)
        wf, dfo, dfi = (
            selected.index(key)
            for key in (
                "WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT",
                "WLK_LRF_DRV_FERRYIVT",
            )
        )
        assign_int(
            "column:walk_ferry_available",
            f"ints[ib + {int_pos['name:walk_lrf_available']}] && "
            f"skim_{wf}[skim_index] > 0.0f",
        )
        assign_int(
            "column:drive_ferry_available",
            f"(outbound && ints[ib + {int_pos['name:drive_lrf_available_outbound']}] "
            f"&& skim_{dfo}[skim_index] > 0.0f) || (!outbound && "
            f"ints[ib + {int_pos['name:drive_lrf_available_inbound']}] && "
            f"skim_{dfi}[skim_index] > 0.0f)",
        )
        if assigned_float != set(self.float_labels):
            raise ValueError("Phase 36 float ABI preparation contract is incomplete")
        if assigned_int != set(self.int_labels):
            missing = sorted(set(self.int_labels) - assigned_int)
            raise ValueError(f"Phase 36 integer ABI preparation is incomplete: {missing}")

        coordinate_contract, odt_dimensions = self._coordinate_contract()
        coordinate_parameters = []
        coordinate_arguments = []
        coordinate_statements = []
        for group, (direction, rank, origin_arg, destination_arg, period_arg, _, _) in enumerate(
            coordinate_contract
        ):
            coordinate_parameters.extend(
                (f"  long long* group_{group}_origin", f"  long long* group_{group}_destination")
            )
            coordinate_arguments.extend((origin_arg, destination_arg))
            reverse = direction in {"dot_skims", "od_skims_reverse"}
            coordinate_statements.extend(
                (
                    f"  group_{group}_origin[row] = {'destination' if reverse else 'origin'};",
                    f"  group_{group}_destination[row] = {'origin' if reverse else 'destination'};",
                )
            )
            if rank == 3:
                coordinate_parameters.append(f"  long long* group_{group}_period")
                coordinate_arguments.append(period_arg)
                coordinate_statements.append(f"  group_{group}_period[row] = period;")

        skim_parameters = [
            f"  const float* skim_{number}" for number in range(len(selected))
        ]
        parameters = ",\n".join(skim_parameters + coordinate_parameters)
        source = f'''extern "C" __global__ void trip_device_prepare(
  float* floats, long long* ints, const int* raw_ints,
  const double* raw_floats, const float* waits,
  const double* terminal, const double* parking, const double* density,
  const long long* topology, long long rows, int float_columns,
  int int_columns, int raw_int_columns, int raw_float_columns,
  long long land_size, long long dest_count, long long time_count,
  double walk_time, unsigned long long auto_modes,
  unsigned long long walk_transit_modes, unsigned long long drive_transit_modes,
  unsigned long long ride_hail_modes, unsigned long long sov_modes,
  unsigned long long sr2_modes, unsigned long long sr3_modes,
  int walk_mode, int bike_mode,
{parameters}) {{
  const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= rows) return;
  const long long fb = row * float_columns;
  const long long ib = row * int_columns;
  const long long rib = row * raw_int_columns;
  const long long rfd = row * raw_float_columns;
  const long long origin = raw_ints[rib + {ri['origin']}];
  const long long destination = raw_ints[rib + {ri['destination']}];
  const long long period = raw_ints[rib + {ri['period']}];
  const bool outbound = raw_ints[rib + {ri['outbound']}] != 0;
  const bool first = raw_ints[rib + {ri['first']}] != 0;
  const bool last = raw_ints[rib + {ri['last']}] != 0;
  const bool free_parking = raw_ints[rib + {ri['free_parking']}] != 0;
  const int mode = raw_ints[rib + {ri['tour_mode']}];
  const int auto_ownership = raw_ints[rib + {ri['auto_ownership']}];
  const double duration = raw_floats[rfd + {rf['duration']}];
  const double origin_duration = first ? ((!outbound) ? duration * (!free_parking) : 0.0) : 1.0;
  const double destination_duration = last ? ((!outbound) ? duration * (!free_parking) : 0.0) : 1.0;
  const long long skim_index = (origin * dest_count + destination) * time_count + period;
{chr(10).join(statements)}
{chr(10).join(coordinate_statements)}
}}'''
        kernel = _DEVICE_PREPARATION_KERNEL_CACHE.get(source)
        if kernel is None:
            kernel = self.cp.RawKernel(
                source,
                "trip_device_prepare",
                options=("--std=c++11", "--fmad=false", "--prec-div=true"),
            )
            kernel.compile()
            _DEVICE_PREPARATION_KERNEL_CACHE[source] = kernel
        return (
            kernel,
            tuple(self._skim_argument(key) for key in selected),
            tuple(coordinate_arguments),
            odt_dimensions[0],
            odt_dimensions[1],
        )

    def _compile_fused_utility(self, *, normalized=False):
        """Compile a fused kernel for Phase 37 rows or Phase 38 normalized state."""
        phase = "Phase 38" if normalized else "Phase 37"
        if self.document is None or self.bindings is None:
            raise ValueError(f"{phase} fusion requires reviewed IR and typed bindings")
        from .sharrow_cuda import generate_cuda_source

        ri = _STABLE_INT_COLUMNS if normalized else _RAW_INT_COLUMNS
        rf = _RAW_FLOAT_COLUMNS
        float_variables = {
            label: f"phase37_float_{position}"
            for position, label in enumerate(self.float_labels)
        }
        int_variables = {
            label: f"phase37_int_{position}"
            for position, label in enumerate(self.int_labels)
        }
        if normalized:
            prelude = [
                "    const long long phase37_state_row = phase38_row_state[row];",
                f"    const long long phase37_rib = phase37_state_row * {len(_STABLE_INT_COLUMNS)};",
                f"    const long long phase37_rfb = phase37_state_row * {len(_RAW_FLOAT_COLUMNS)};",
                "    const long long phase37_wait_base = phase37_state_row * 3;",
                f"    const long long origin = phase38_row_coordinates[row * {len(_ROW_COORD_COLUMNS)} + {_ROW_COORD_COLUMNS['origin']}];",
                f"    const long long destination = phase38_row_coordinates[row * {len(_ROW_COORD_COLUMNS)} + {_ROW_COORD_COLUMNS['destination']}];",
            ]
        else:
            prelude = [
                f"    const long long phase37_rib = row * {len(_RAW_INT_COLUMNS)};",
                f"    const long long phase37_rfb = row * {len(_RAW_FLOAT_COLUMNS)};",
                "    const long long phase37_wait_base = row * 3;",
                f"    const long long origin = phase37_raw_ints[phase37_rib + {ri['origin']}];",
                f"    const long long destination = phase37_raw_ints[phase37_rib + {ri['destination']}];",
            ]
        prelude.extend([
            f"    const long long period = phase37_raw_ints[phase37_rib + {ri['period']}];",
            f"    const bool outbound = phase37_raw_ints[phase37_rib + {ri['outbound']}] != 0;",
            f"    const bool first = phase37_raw_ints[phase37_rib + {ri['first']}] != 0;",
            f"    const bool last = phase37_raw_ints[phase37_rib + {ri['last']}] != 0;",
            f"    const bool free_parking = phase37_raw_ints[phase37_rib + {ri['free_parking']}] != 0;",
            f"    const int mode = phase37_raw_ints[phase37_rib + {ri['tour_mode']}];",
            f"    const int auto_ownership = phase37_raw_ints[phase37_rib + {ri['auto_ownership']}];",
            f"    const double duration = phase37_raw_floats[phase37_rfb + {rf['duration']}];",
            "    const double origin_duration = first ? ((!outbound) ? duration * (!free_parking) : 0.0) : 1.0;",
            "    const double destination_duration = last ? ((!outbound) ? duration * (!free_parking) : 0.0) : 1.0;",
            "    const long long phase37_skim_index = (origin * phase37_dest_count + destination) * phase37_time_count + period;",
        ])
        float_expressions = {
            "column:total_terminal_time": (
                "((outbound && first) ? 0.0 : phase37_terminal[origin]) + "
                "((!outbound && last) ? 0.0 : phase37_terminal[destination])"
            ),
            "column:ivot": (
                f"1.0 / phase37_raw_floats[phase37_rfb + {rf['value_of_time']}]"
            ),
            "column:total_parking_cost": (
                "(origin_duration * phase37_parking[origin] + destination_duration * "
                "phase37_parking[destination]) / 2.0"
            ),
            "column:density_index": (
                "outbound ? phase37_density[destination] : phase37_density[origin]"
            ),
            "column:origin_walk_time": "phase37_walk_time",
            "column:destination_walk_time": "phase37_walk_time",
            "column:origTaxiWaitTime": "phase37_waits[phase37_wait_base]",
            "column:origSingleTNCWaitTime": "phase37_waits[phase37_wait_base + 1]",
            "column:origSharedTNCWaitTime": "phase37_waits[phase37_wait_base + 2]",
            "column:i_tour_mode": "mode",
            "column:origin_density_index": (
                "outbound ? phase37_density[origin] : phase37_density[destination]"
            ),
        }
        selected = []
        availability_expressions = {}
        for label, (key, direction, threshold) in _AVAILABILITY.items():
            if key not in selected:
                selected.append(key)
            number = selected.index(key)
            conditions = [
                f"phase37_avail_skim_{number}[phase37_skim_index] > 0.0f"
            ]
            if direction == 1:
                conditions.extend(("auto_ownership > 0", "outbound"))
            elif direction == 2:
                conditions.extend(("auto_ownership > 0", "!outbound"))
            if threshold is not None:
                conditions.append(f"mode >= {int(threshold)}")
            availability_expressions[label] = " && ".join(conditions)
        for key in (
            "WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT",
            "WLK_LRF_DRV_FERRYIVT",
        ):
            if key not in selected:
                selected.append(key)
        wf, dfo, dfi = (
            selected.index(key)
            for key in (
                "WLK_LRF_WLK_FERRYIVT", "DRV_LRF_WLK_FERRYIVT",
                "WLK_LRF_DRV_FERRYIVT",
            )
        )
        availability_expressions["column:walk_ferry_available"] = (
            f"({availability_expressions['name:walk_lrf_available']}) && "
            f"phase37_avail_skim_{wf}[phase37_skim_index] > 0.0f"
        )
        availability_expressions["column:drive_ferry_available"] = (
            f"(outbound && ({availability_expressions['name:drive_lrf_available_outbound']}) "
            f"&& phase37_avail_skim_{dfo}[phase37_skim_index] > 0.0f) || "
            f"(!outbound && ({availability_expressions['name:drive_lrf_available_inbound']}) "
            f"&& phase37_avail_skim_{dfi}[phase37_skim_index] > 0.0f)"
        )
        int_expressions = {
            "name:auto_ownership": f"phase37_raw_ints[phase37_rib + {ri['auto_ownership']}]",
            "name:age": f"phase37_raw_ints[phase37_rib + {ri['age']}]",
            "name:is_joint": f"phase37_raw_ints[phase37_rib + {ri['participants']}] > 1",
            "name:is_atwork_subtour": f"phase37_raw_ints[phase37_rib + {ri['is_atwork_subtour']}] != 0",
            "name:work_tour_is_SOV": f"phase37_raw_ints[phase37_rib + {ri['parent_mode']}] == 1",
            "name:number_of_participants": f"phase37_raw_ints[phase37_rib + {ri['participants']}]",
            "column:hhsize": f"phase37_raw_ints[phase37_rib + {ri['hhsize']}]",
            "column:trip_topology": "outbound ? phase37_topology[destination] : phase37_topology[origin]",
            "name:work_tour_is_bike": f"phase37_raw_ints[phase37_rib + {ri['parent_mode']}] == 2",
            "name:outbound": "outbound",
            "name:inbound": "!outbound",
            "name:tour_mode_is_auto": "((phase37_auto_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_walk": "mode == phase37_walk_mode",
            "name:tour_mode_is_bike": "mode == phase37_bike_mode",
            "name:tour_mode_is_walk_transit": "((phase37_walk_transit_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_drive_transit": "((phase37_drive_transit_modes >> mode) & 1ULL) != 0ULL",
            "name:tour_mode_is_ride_hail": "((phase37_ride_hail_modes >> mode) & 1ULL) != 0ULL",
            "column:is_indiv": f"phase37_raw_ints[phase37_rib + {ri['participants']}] == 1",
            "column:tour_mode_is_SOV": "((phase37_sov_modes >> mode) & 1ULL) != 0ULL",
            "column:tour_mode_is_SR2": "((phase37_sr2_modes >> mode) & 1ULL) != 0ULL",
            "column:tour_mode_is_SR3P": "((phase37_sr3_modes >> mode) & 1ULL) != 0ULL",
            "column:first_trip": "first",
            **availability_expressions,
        }
        if set(float_expressions) != set(self.float_labels):
            raise ValueError(f"{phase} fused float contract is incomplete")
        if set(int_expressions) != set(self.int_labels):
            missing = sorted(set(self.int_labels) - set(int_expressions))
            raise ValueError(f"{phase} fused integer contract is incomplete: {missing}")
        for label in self.float_labels:
            prelude.append(
                f"    const float {float_variables[label]} = (float)({float_expressions[label]});"
            )
        for label in self.int_labels:
            prelude.append(
                f"    const long long {int_variables[label]} = (long long)({int_expressions[label]});"
            )
        float_reference_by_slot = {
            slot: float_variables[label]
            for slot, label in enumerate(self.float_labels)
        }
        int_reference_by_slot = {
            slot: int_variables[label]
            for slot, label in enumerate(self.int_labels)
        }
        row_references = {}
        for binding in self.bindings:
            if binding.storage_kind == "float64":
                row_references[binding.source] = float_reference_by_slot[binding.slot]
            elif binding.storage_kind == "int64":
                row_references[binding.source] = int_reference_by_slot[binding.slot]
        group_coordinates = {}
        for binding in self.bindings:
            if binding.storage_kind != "skim":
                continue
            direction = binding.source[1]
            reverse = direction in {"dot_skims", "od_skims_reverse"}
            coordinates = (
                "destination" if reverse else "origin",
                "origin" if reverse else "destination",
                "period" if binding.skim_rank == 3 else None,
            )
            previous = group_coordinates.setdefault(binding.skim_group, coordinates)
            if previous != coordinates:
                raise ValueError(f"{phase} skim group mixes coordinate directions")
        extra_parameters = (
            "    const int* phase37_raw_ints",
            "    const double* phase37_raw_floats",
            "    const float* phase37_waits",
        ) + (
            (
                "    const int* phase38_row_coordinates",
                "    const int* phase38_row_state",
            ) if normalized else ()
        ) + (
            "    const double* phase37_terminal",
            "    const double* phase37_parking",
            "    const double* phase37_density",
            "    const long long* phase37_topology",
            "    long long phase37_dest_count",
            "    long long phase37_time_count",
            "    double phase37_walk_time",
            "    unsigned long long phase37_auto_modes",
            "    unsigned long long phase37_walk_transit_modes",
            "    unsigned long long phase37_drive_transit_modes",
            "    unsigned long long phase37_ride_hail_modes",
            "    unsigned long long phase37_sov_modes",
            "    unsigned long long phase37_sr2_modes",
            "    unsigned long long phase37_sr3_modes",
            "    int phase37_walk_mode",
            "    int phase37_bike_mode",
        ) + tuple(
            f"    const float* phase37_avail_skim_{number}"
            for number in range(len(selected))
        )
        source, _ = generate_cuda_source(
            self.document,
            self.bindings,
            capture_features=False,
            locality_tile_rows=1,
            locality_optimized=False,
            group_skim_indices=True,
            sparse_zero_coefficients=False,
            expression_float32=True,
            fused_utility_accumulation=True,
            row_source_references=row_references,
            group_coordinate_references=group_coordinates,
            extra_kernel_parameters=extra_parameters,
            row_prelude="\n".join(prelude),
        )
        unresolved_patterns = (
            r"float_inputs\s*\[\s*row\s*\*",
            r"int_inputs\s*\[\s*row\s*\*",
            r"skim_group_\d+_(?:orig|dest|time)\s*\[\s*row\s*\]",
        )
        unresolved = [
            pattern for pattern in unresolved_patterns if re.search(pattern, source)
        ]
        if unresolved:
            raise ValueError(
                f"{phase} fused source retains legacy row reads: {unresolved}"
            )
        kernel = _FUSED_UTILITY_KERNEL_CACHE.get(source)
        if kernel is None:
            kernel = self.cp.RawKernel(
                source,
                "choiceforge_strict_ir_v3",
                options=("--std=c++11", "--fmad=true", "--prec-div=true", "--ftz=true"),
            )
            kernel.compile()
            _FUSED_UTILITY_KERNEL_CACHE[source] = kernel
        _, dimensions = self._coordinate_contract()
        self._skim_dest_count, self._skim_time_count = dimensions
        return kernel, tuple(self._skim_argument(key) for key in selected)

    def _resident_land(self, land_use):
        key = id(land_use)
        cached = _RESIDENT_LAND_CACHE.get(key)
        if cached is not None:
            return cached
        labels = np.asarray(land_use.index, dtype=np.int64)
        if labels.size == 0 or labels.min() < 0 or len(np.unique(labels)) != len(labels):
            raise ValueError("Phase 36 land-use index must be unique nonnegative integers")
        size = int(labels.max()) + 1
        if size > max(1_000_000, 16 * len(labels)):
            raise ValueError("Phase 36 land-use index is too sparse for resident lookup")
        valid = np.zeros(size, dtype=bool)
        valid[labels] = True

        def dense(column, dtype):
            values = np.asarray(land_use[column], dtype=dtype)
            target = np.zeros(size, dtype=dtype)
            target[labels] = values
            if np.issubdtype(np.dtype(dtype), np.floating) and not np.isfinite(values).all():
                raise ValueError(f"Phase 36 land-use column {column!r} is not finite")
            return self.cp.asarray(target)

        device = (
            dense("TERMINAL", np.float64), dense("PRKCST", np.float64),
            dense("density_index", np.float64), dense("TOPOLOGY", np.int64),
        )
        cached = {
            "valid": valid,
            "size": size,
            "device": device,
            "bytes": int(sum(item.nbytes for item in device)),
        }
        _RESIDENT_LAND_CACHE[key] = cached
        return cached

    @staticmethod
    def _checked_int32(values, label):
        array = np.asarray(values)
        if array.size:
            info = np.iinfo(np.int32)
            if np.nanmin(array) < info.min or np.nanmax(array) > info.max:
                raise ValueError(f"Phase 36 raw column {label!r} exceeds int32")
        return np.asarray(array, dtype=np.int32)

    def _compact_packet(self, frame, land_use, tours, constants, draws, phase):
        """Build and validate the exact compact row contract shared by Phases 36/37."""
        origin = _values(frame, "_choiceforge_origin", np.int64)
        destination = _values(frame, "_choiceforge_destination", np.int64)
        period = _period(_values(frame, "trip_period"))
        original_origin = _values(frame, str(constants["orig_col_name"]), np.int64)
        outbound = _values(frame, "outbound", bool)
        first = _values(frame, "trip_num", np.int64) == 1
        last = _values(frame, "trip_num", np.int64) == _values(
            frame, "trip_count", np.int64
        )
        free = (_values(frame, "tour_type").astype(str) == "work") & _values(
            frame, "free_parking_at_work", bool
        )
        mode_map = constants["I_MODE_MAP"]
        try:
            mode = np.asarray(
                [mode_map[str(item)] for item in _values(frame, "tour_mode")],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(f"{phase} tour mode is unsupported: {exc.args[0]!r}") from exc
        parent = tours["tour_mode"].reindex(
            _values(frame, "parent_tour_id")
        ).fillna("").to_numpy()
        parent_mode = np.select(
            [np.isin(parent, ["DRIVEALONEFREE", "DRIVEALONEPAY"]), parent == "BIKE"],
            [1, 2],
            default=0,
        )
        participants = _values(frame, "number_of_participants", np.int64)
        raw_int_values = {
            "origin": origin, "destination": destination, "period": period,
            "outbound": outbound, "first": first, "last": last,
            "free_parking": free, "tour_mode": mode,
            "parent_mode": parent_mode,
            "is_atwork_subtour": ~np.asarray(frame["parent_tour_id"].isna()),
            "auto_ownership": _values(frame, "auto_ownership", np.int64),
            "age": _values(frame, "age", np.int64),
            "participants": participants,
            "hhsize": _values(frame, "hhsize", np.int64),
        }
        raw_ints = np.column_stack(
            [self._checked_int32(raw_int_values[name], name) for name in _RAW_INT_COLUMNS]
        ).astype(np.int32, copy=False)
        raw_floats = np.column_stack(
            [
                _values(frame, "duration", np.float64),
                _values(frame, "value_of_time", np.float64),
            ]
        ).astype(np.float64, copy=False)
        if not np.isfinite(raw_floats).all():
            raise ValueError(f"{phase} raw floating-point state is not finite")
        bands = _density_band(land_use, original_origin)
        waits = np.column_stack(
            [
                _wait(
                    np.asarray(draws)[:, number],
                    _mapped(constants[f"{family}_waitTime_mean"], bands),
                    _mapped(constants[f"{family}_waitTime_sd"], bands),
                    float(constants["min_waitTime"]),
                    float(constants["max_waitTime"]),
                )
                for number, family in enumerate(
                    ("Taxi", "TNC_single", "TNC_shared")
                )
            ]
        ).astype(np.float32, copy=False)
        land = self._resident_land(land_use)
        for label, zones in (
            ("origin", origin), ("destination", destination),
            ("original_origin", original_origin),
        ):
            if zones.size and (
                zones.min() < 0 or zones.max() >= land["size"]
                or not np.all(land["valid"][zones])
            ):
                raise ValueError(f"{phase} {label} contains an unknown land-use zone")
        if origin.size and (
            origin.max() >= self._skim_dest_count
            or destination.max() >= self._skim_dest_count
            or period.min() < 0 or period.max() >= self._skim_time_count
        ):
            raise ValueError(f"{phase} compact skim coordinate is outside the cube")
        mode_masks = (
            self._mode_mask(constants["I_AUTO_MODES"]),
            self._mode_mask(constants["I_WALK_TRANSIT_MODES"]),
            self._mode_mask(constants["I_DRIVE_TRANSIT_MODES"]),
            self._mode_mask(constants["I_RIDE_HAIL_MODES"]),
            self._mode_mask(constants["I_SOV_MODES"]),
            self._mode_mask(constants["I_SR2_MODES"]),
            self._mode_mask(constants["I_SR3P_MODES"]),
        )
        walk_time = float(constants["shortWalk"]) * 60.0 / float(
            constants["walkSpeed"]
        )
        return raw_ints, raw_floats, waits, land, mode_masks, walk_time

    def _normalized_packet(self, frame, land_use, tours, constants, draws):
        """Build Phase 38 row coordinates plus deduplicated directional state."""
        stable_columns = (
            "trip_period", str(constants["orig_col_name"]), "outbound",
            "trip_num", "trip_count", "tour_type", "free_parking_at_work",
            "tour_mode", "parent_tour_id", "auto_ownership", "age",
            "number_of_participants", "hhsize", "duration", "value_of_time",
        )
        state_first, row_state, unique_trip_rows = _normalized_row_layout(
            frame, draws, stable_columns
        )
        state = frame.iloc[state_first]
        origin = _values(frame, "_choiceforge_origin", np.int64)
        destination = _values(frame, "_choiceforge_destination", np.int64)
        row_coordinates = np.column_stack((
            self._checked_int32(origin, "origin"),
            self._checked_int32(destination, "destination"),
        )).astype(np.int32, copy=False)

        period = _period(_values(state, "trip_period"))
        original_origin = _values(
            state, str(constants["orig_col_name"]), np.int64
        )
        outbound = _values(state, "outbound", bool)
        first = _values(state, "trip_num", np.int64) == 1
        last = _values(state, "trip_num", np.int64) == _values(
            state, "trip_count", np.int64
        )
        free = (_values(state, "tour_type").astype(str) == "work") & _values(
            state, "free_parking_at_work", bool
        )
        mode_map = constants["I_MODE_MAP"]
        try:
            mode = np.asarray(
                [mode_map[str(item)] for item in _values(state, "tour_mode")],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Phase 38 tour mode is unsupported: {exc.args[0]!r}"
            ) from exc
        parent = tours["tour_mode"].reindex(
            _values(state, "parent_tour_id")
        ).fillna("").to_numpy()
        parent_mode = np.select(
            [np.isin(parent, ["DRIVEALONEFREE", "DRIVEALONEPAY"]), parent == "BIKE"],
            [1, 2],
            default=0,
        )
        participants = _values(state, "number_of_participants", np.int64)
        state_int_values = {
            "period": period,
            "outbound": outbound,
            "first": first,
            "last": last,
            "free_parking": free,
            "tour_mode": mode,
            "parent_mode": parent_mode,
            "is_atwork_subtour": ~np.asarray(state["parent_tour_id"].isna()),
            "auto_ownership": _values(state, "auto_ownership", np.int64),
            "age": _values(state, "age", np.int64),
            "participants": participants,
            "hhsize": _values(state, "hhsize", np.int64),
        }
        state_ints = np.column_stack([
            self._checked_int32(state_int_values[name], name)
            for name in _STABLE_INT_COLUMNS
        ]).astype(np.int32, copy=False)
        state_floats = np.column_stack((
            _values(state, "duration", np.float64),
            _values(state, "value_of_time", np.float64),
        )).astype(np.float64, copy=False)
        if not np.isfinite(state_floats).all():
            raise ValueError("Phase 38 normalized floating-point state is not finite")
        bands = _density_band(land_use, original_origin)
        state_draws = np.asarray(draws, dtype=np.float64)[state_first]
        state_waits = np.column_stack([
            _wait(
                state_draws[:, number],
                _mapped(constants[f"{family}_waitTime_mean"], bands),
                _mapped(constants[f"{family}_waitTime_sd"], bands),
                float(constants["min_waitTime"]),
                float(constants["max_waitTime"]),
            )
            for number, family in enumerate(("Taxi", "TNC_single", "TNC_shared"))
        ]).astype(np.float32, copy=False)

        land = self._resident_land(land_use)
        for label, zones in (
            ("origin", origin), ("destination", destination),
            ("original_origin", original_origin),
        ):
            if zones.size and (
                zones.min() < 0 or zones.max() >= land["size"]
                or not np.all(land["valid"][zones])
            ):
                raise ValueError(f"Phase 38 {label} contains an unknown land-use zone")
        if origin.size and (
            origin.max() >= self._skim_dest_count
            or destination.max() >= self._skim_dest_count
            or period.min() < 0 or period.max() >= self._skim_time_count
        ):
            raise ValueError("Phase 38 normalized skim coordinate is outside the cube")
        mode_masks = (
            self._mode_mask(constants["I_AUTO_MODES"]),
            self._mode_mask(constants["I_WALK_TRANSIT_MODES"]),
            self._mode_mask(constants["I_DRIVE_TRANSIT_MODES"]),
            self._mode_mask(constants["I_RIDE_HAIL_MODES"]),
            self._mode_mask(constants["I_SOV_MODES"]),
            self._mode_mask(constants["I_SR2_MODES"]),
            self._mode_mask(constants["I_SR3P_MODES"]),
        )
        walk_time = float(constants["shortWalk"]) * 60.0 / float(
            constants["walkSpeed"]
        )
        return (
            row_coordinates, row_state, state_ints, state_floats, state_waits,
            unique_trip_rows, land, mode_masks, walk_time,
        )

    def populate_normalized(self, frame, land_use, tours, constants, draws):
        """Run Phase 38 from normalized directional state and resident buffers."""
        started = time.perf_counter()
        rows = len(frame)
        if rows != self.invocation.rows:
            raise ValueError("trip native frame length differs from invocation")
        if self._normalized_utility_kernel is None:
            (
                self._normalized_utility_kernel,
                self._normalized_extra_args,
            ) = self._compile_fused_utility(normalized=True)
        (
            row_coordinates, row_state, state_ints, state_floats, state_waits,
            unique_trip_rows, land, mode_masks, walk_time,
        ) = self._normalized_packet(frame, land_use, tours, constants, draws)
        built = time.perf_counter()
        device_arrays = []
        workspace_hits = 0
        for name, host in (
            ("row_coordinates", row_coordinates),
            ("row_state", row_state),
            ("state_ints", state_ints),
            ("state_floats", state_floats),
            ("state_waits", state_waits),
        ):
            device, hit = _upload_normalized(self.cp, name, host)
            device_arrays.append(device)
            workspace_hits += int(hit)
        self.cp.cuda.Stream.null.synchronize()
        uploaded = time.perf_counter()
        (
            row_coordinates_device, row_state_device, state_ints_device,
            state_floats_device, state_waits_device,
        ) = device_arrays
        extra_arguments = (
            state_ints_device, state_floats_device, state_waits_device,
            row_coordinates_device, row_state_device,
            *land["device"], np.int64(self._skim_dest_count),
            np.int64(self._skim_time_count), np.float64(walk_time),
            *mode_masks, np.int32(constants["I_WALK_MODE"]),
            np.int32(constants["I_BIKE_MODE"]), *self._normalized_extra_args,
        )
        self._normalized_utility_kernel(
            self.invocation.grid,
            self.invocation.block,
            (
                self.invocation.float_inputs,
                self.invocation.int_inputs,
                self.invocation.float_scalars,
                self.invocation.int_scalars,
                self.invocation.coefficients,
                self.invocation.features,
                self.invocation.utilities,
                np.int64(rows),
            ) + self.invocation.skim_arguments + extra_arguments,
            shared_mem=self.invocation.shared_mem,
        )
        self.cp.cuda.Stream.null.synchronize()
        completed = time.perf_counter()
        row_bytes = int(row_coordinates.nbytes + row_state.nbytes)
        state_bytes = int(
            state_ints.nbytes + state_floats.nbytes + state_waits.nbytes
        )
        compact_bytes = row_bytes + state_bytes
        phase37_bytes = int(
            rows * (
                len(_RAW_INT_COLUMNS) * 4 + len(_RAW_FLOAT_COLUMNS) * 8 + 3 * 4
            )
        )
        coordinate_contract, _ = self._coordinate_contract()
        coordinate_columns = sum(
            2 + int(rank == 3) for _, rank, *_ in coordinate_contract
        )
        dense_device_bytes = int(
            rows * (len(self.float_labels) * 4 + len(self.int_labels) * 8)
        )
        coordinate_bytes = int(rows * coordinate_columns * 8)
        minimal_bootstrap = int(
            self.invocation.float_inputs.nbytes
            + self.invocation.int_inputs.nbytes
            + self.invocation.skim_coordinate_bytes
        )
        return self.invocation.utilities, TripLogsumNativeTelemetry(
            rows=rows,
            compact_host_bytes=compact_bytes,
            host_build_seconds=built - started,
            upload_seconds=uploaded - built,
            availability_kernel_seconds=0.0,
            utility_kernel_seconds=completed - uploaded,
            backend="phase38_normalized_fused_utility",
            compact_device_input_bytes=compact_bytes,
            dense_host_abi_bytes_avoided=dense_device_bytes + coordinate_bytes,
            resident_land_bytes=land["bytes"],
            dense_device_abi_bytes_eliminated=dense_device_bytes,
            coordinate_device_bytes_eliminated=coordinate_bytes,
            fused_kernel_seconds=completed - uploaded,
            minimal_bootstrap_bytes=minimal_bootstrap,
            normalized_trip_rows=int(unique_trip_rows),
            normalized_state_rows=int(len(state_ints)),
            normalized_row_bytes=row_bytes,
            normalized_state_bytes=state_bytes,
            phase37_compact_bytes_eliminated=phase37_bytes - compact_bytes,
            resident_workspace_hits=workspace_hits,
            resident_workspace_arrays=len(device_arrays),
            normalized_contract_valid=True,
        )

    def populate_fused(self, frame, land_use, tours, constants, draws):
        """Evaluate utilities without materializing the 11/45 device ABI."""
        started = time.perf_counter()
        rows = len(frame)
        if rows != self.invocation.rows:
            raise ValueError("trip native frame length differs from invocation")
        if self._fused_utility_kernel is None:
            (
                self._fused_utility_kernel,
                self._fused_extra_args,
            ) = self._compile_fused_utility()
        raw_ints, raw_floats, waits, land, mode_masks, walk_time = (
            self._compact_packet(
                frame, land_use, tours, constants, draws, "Phase 37"
            )
        )
        built = time.perf_counter()
        raw_ints_device = self.cp.asarray(raw_ints)
        raw_floats_device = self.cp.asarray(raw_floats)
        waits_device = self.cp.asarray(waits)
        self.cp.cuda.Stream.null.synchronize()
        uploaded = time.perf_counter()
        extra_arguments = (
            raw_ints_device, raw_floats_device, waits_device,
            *land["device"], np.int64(self._skim_dest_count),
            np.int64(self._skim_time_count), np.float64(walk_time),
            *mode_masks, np.int32(constants["I_WALK_MODE"]),
            np.int32(constants["I_BIKE_MODE"]), *self._fused_extra_args,
        )
        self._fused_utility_kernel(
            self.invocation.grid,
            self.invocation.block,
            (
                self.invocation.float_inputs,
                self.invocation.int_inputs,
                self.invocation.float_scalars,
                self.invocation.int_scalars,
                self.invocation.coefficients,
                self.invocation.features,
                self.invocation.utilities,
                np.int64(rows),
            ) + self.invocation.skim_arguments + extra_arguments,
            shared_mem=self.invocation.shared_mem,
        )
        self.cp.cuda.Stream.null.synchronize()
        completed = time.perf_counter()
        compact_bytes = int(raw_ints.nbytes + raw_floats.nbytes + waits.nbytes)
        coordinate_contract, _ = self._coordinate_contract()
        coordinate_columns = sum(
            2 + int(rank == 3)
            for _, rank, *_ in coordinate_contract
        )
        dense_device_bytes = int(
            rows * (len(self.float_labels) * 4 + len(self.int_labels) * 8)
        )
        coordinate_bytes = int(rows * coordinate_columns * 8)
        minimal_bootstrap = int(
            self.invocation.float_inputs.nbytes
            + self.invocation.int_inputs.nbytes
            + self.invocation.skim_coordinate_bytes
        )
        return self.invocation.utilities, TripLogsumNativeTelemetry(
            rows=rows,
            compact_host_bytes=compact_bytes,
            host_build_seconds=built - started,
            upload_seconds=uploaded - built,
            availability_kernel_seconds=0.0,
            utility_kernel_seconds=completed - uploaded,
            backend="phase37_fused_raw_utility",
            compact_device_input_bytes=compact_bytes,
            dense_host_abi_bytes_avoided=dense_device_bytes + coordinate_bytes,
            resident_land_bytes=land["bytes"],
            dense_device_abi_bytes_eliminated=dense_device_bytes,
            coordinate_device_bytes_eliminated=coordinate_bytes,
            fused_kernel_seconds=completed - uploaded,
            minimal_bootstrap_bytes=minimal_bootstrap,
        )

    def populate_device(self, frame, land_use, tours, constants, draws):
        """Generate the full strict ABI on CUDA from a compact raw state packet."""
        started = time.perf_counter()
        rows = len(frame)
        if rows != self.invocation.rows:
            raise ValueError("trip native frame length differs from invocation")
        if self._device_preparation_kernel is None:
            (
                self._device_preparation_kernel,
                self._device_preparation_args,
                self._coordinate_args,
                self._skim_dest_count,
                self._skim_time_count,
            ) = self._compile_device_preparation()
        origin = _values(frame, "_choiceforge_origin", np.int64)
        destination = _values(frame, "_choiceforge_destination", np.int64)
        period = _period(_values(frame, "trip_period"))
        original_origin = _values(frame, str(constants["orig_col_name"]), np.int64)
        outbound = _values(frame, "outbound", bool)
        first = _values(frame, "trip_num", np.int64) == 1
        last = _values(frame, "trip_num", np.int64) == _values(
            frame, "trip_count", np.int64
        )
        free = (_values(frame, "tour_type").astype(str) == "work") & _values(
            frame, "free_parking_at_work", bool
        )
        mode_map = constants["I_MODE_MAP"]
        mode = np.asarray(
            [mode_map[str(item)] for item in _values(frame, "tour_mode")],
            dtype=np.int64,
        )
        parent = tours["tour_mode"].reindex(
            _values(frame, "parent_tour_id")
        ).fillna("").to_numpy()
        parent_mode = np.select(
            [np.isin(parent, ["DRIVEALONEFREE", "DRIVEALONEPAY"]), parent == "BIKE"],
            [1, 2],
            default=0,
        )
        atwork = ~np.asarray(frame["parent_tour_id"].isna())
        auto = _values(frame, "auto_ownership", np.int64)
        participants = _values(frame, "number_of_participants", np.int64)
        raw_int_values = {
            "origin": origin, "destination": destination, "period": period,
            "outbound": outbound,
            "first": first, "last": last, "free_parking": free,
            "tour_mode": mode, "parent_mode": parent_mode,
            "is_atwork_subtour": atwork, "auto_ownership": auto,
            "age": _values(frame, "age", np.int64),
            "participants": participants, "hhsize": _values(frame, "hhsize", np.int64),
        }
        raw_ints = np.column_stack(
            [
                self._checked_int32(raw_int_values[name], name)
                for name in _RAW_INT_COLUMNS
            ]
        ).astype(np.int32, copy=False)
        raw_floats = np.column_stack(
            [
                _values(frame, "duration", np.float64),
                _values(frame, "value_of_time", np.float64),
            ]
        ).astype(np.float64, copy=False)

        # Preserve Phase 35/ActivitySim wait arithmetic exactly; only the final
        # float32 results cross the compact boundary. Phase 37 can qualify a
        # CUDA transcendental implementation separately if it is worthwhile.
        bands = _density_band(land_use, original_origin)
        waits = np.column_stack(
            [
                _wait(
                    np.asarray(draws)[:, number],
                    _mapped(constants[f"{family}_waitTime_mean"], bands),
                    _mapped(constants[f"{family}_waitTime_sd"], bands),
                    float(constants["min_waitTime"]),
                    float(constants["max_waitTime"]),
                )
                for number, family in enumerate(
                    ("Taxi", "TNC_single", "TNC_shared")
                )
            ]
        ).astype(np.float32, copy=False)
        land = self._resident_land(land_use)
        for label, zones in (
            ("origin", origin), ("destination", destination),
            ("original_origin", original_origin),
        ):
            if zones.size and (
                zones.min() < 0 or zones.max() >= land["size"]
                or not np.all(land["valid"][zones])
            ):
                raise ValueError(f"Phase 36 {label} contains an unknown land-use zone")
        if origin.size and (
            origin.max() >= self._skim_dest_count
            or destination.max() >= self._skim_dest_count
            or period.min() < 0 or period.max() >= self._skim_time_count
        ):
            raise ValueError("Phase 36 compact skim coordinate is outside the cube")
        built = time.perf_counter()
        raw_ints_device = self.cp.asarray(raw_ints)
        raw_floats_device = self.cp.asarray(raw_floats)
        waits_device = self.cp.asarray(waits)
        self.cp.cuda.Stream.null.synchronize()
        uploaded = time.perf_counter()
        mode_masks = (
            self._mode_mask(constants["I_AUTO_MODES"]),
            self._mode_mask(constants["I_WALK_TRANSIT_MODES"]),
            self._mode_mask(constants["I_DRIVE_TRANSIT_MODES"]),
            self._mode_mask(constants["I_RIDE_HAIL_MODES"]),
            self._mode_mask(constants["I_SOV_MODES"]),
            self._mode_mask(constants["I_SR2_MODES"]),
            self._mode_mask(constants["I_SR3P_MODES"]),
        )
        walk_time = float(constants["shortWalk"]) * 60.0 / float(
            constants["walkSpeed"]
        )
        self._device_preparation_kernel(
            ((rows + 255) // 256,),
            (256,),
            (
                self.invocation.float_inputs, self.invocation.int_inputs,
                raw_ints_device, raw_floats_device, waits_device,
                *land["device"], np.int64(rows),
                np.int32(len(self.float_labels)), np.int32(len(self.int_labels)),
                np.int32(len(_RAW_INT_COLUMNS)), np.int32(len(_RAW_FLOAT_COLUMNS)),
                np.int64(land["size"]), np.int64(self._skim_dest_count),
                np.int64(self._skim_time_count), np.float64(walk_time),
                *mode_masks, np.int32(constants["I_WALK_MODE"]),
                np.int32(constants["I_BIKE_MODE"]),
                *self._device_preparation_args, *self._coordinate_args,
            ),
        )
        self.cp.cuda.Stream.null.synchronize()
        prepared = time.perf_counter()
        utilities = self.invocation.execute()
        self.cp.cuda.Stream.null.synchronize()
        utility_done = time.perf_counter()
        compact_bytes = int(raw_ints.nbytes + raw_floats.nbytes + waits.nbytes)
        telemetry = TripLogsumNativeTelemetry(
            rows=rows,
            compact_host_bytes=compact_bytes,
            host_build_seconds=built - started,
            upload_seconds=uploaded - built,
            availability_kernel_seconds=0.0,
            utility_kernel_seconds=utility_done - prepared,
            backend="phase36_device_abi",
            device_preparation_kernel_seconds=prepared - uploaded,
            compact_device_input_bytes=compact_bytes,
            dense_host_abi_bytes_avoided=int(
                rows * (
                    len(self.float_labels) * 4 + len(self.int_labels) * 8
                    + 3 * 8  # former host origin/destination/period coordinates
                )
            ),
            resident_land_bytes=land["bytes"],
        )
        return utilities, telemetry

    def populate(self, frame, land_use, tours, constants, draws):
        started = time.perf_counter()
        rows = len(frame)
        if rows != self.invocation.rows:
            raise ValueError("trip native frame length differs from invocation")
        origin = _values(frame, "_choiceforge_origin", np.int64)
        destination = _values(frame, "_choiceforge_destination", np.int64)
        period = _period(_values(frame, "trip_period"))
        outbound = _values(frame, "outbound", bool)
        inbound = ~outbound
        first = _values(frame, "trip_num", np.int64) == 1
        last = _values(frame, "trip_num", np.int64) == _values(frame, "trip_count", np.int64)
        free = (_values(frame, "tour_type").astype(str) == "work") & _values(frame, "free_parking_at_work", bool)
        origin_duration = np.where(first, np.where(inbound, _values(frame, "duration", np.float64) * ~free, 0), 1)
        dest_duration = np.where(last, np.where(inbound, _values(frame, "duration", np.float64) * ~free, 0), 1)
        mode_map = constants["I_MODE_MAP"]
        mode = np.asarray([mode_map[str(item)] for item in _values(frame, "tour_mode")], dtype=np.int64)
        parent = tours["tour_mode"].reindex(_values(frame, "parent_tour_id")).fillna("").to_numpy()
        original_origin = _values(frame, str(constants["orig_col_name"]), np.int64)
        land_columns = ["TERMINAL", "PRKCST", "density_index", "TOPOLOGY"]
        origin_land = land_use[land_columns].reindex(origin)
        destination_land = land_use[land_columns].reindex(destination)
        if origin_land.isna().any().any() or destination_land.isna().any().any():
            raise ValueError("trip native directional land-use lookup contains missing zones")
        original_land = land_use[["TOTPOP", "TOTEMP", "TOTACRE"]].reindex(original_origin)
        if original_land.isna().any().any():
            raise ValueError("trip native wait-time land-use lookup contains missing zones")
        density_measure = (
            original_land["TOTPOP"].to_numpy(copy=False)
            + original_land["TOTEMP"].to_numpy(copy=False)
        ) / (original_land["TOTACRE"].to_numpy(copy=False) / 640.0)
        bands = np.select(
            [density_measure <= 500, density_measure <= 2000,
             density_measure <= 5000, density_measure <= 15000],
            [5, 4, 3, 2], default=1,
        ).astype(np.int64)
        waits = []
        for number, family in enumerate(("Taxi", "TNC_single", "TNC_shared")):
            waits.append(_wait(
                np.asarray(draws)[:, number],
                _mapped(constants[f"{family}_waitTime_mean"], bands),
                _mapped(constants[f"{family}_waitTime_sd"], bands),
                float(constants["min_waitTime"]), float(constants["max_waitTime"]),
            ))
        float_values = {
            "column:total_terminal_time": np.where(outbound & first, 0, origin_land["TERMINAL"].to_numpy(copy=False)) + np.where(inbound & last, 0, destination_land["TERMINAL"].to_numpy(copy=False)),
            "column:ivot": 1.0 / _values(frame, "value_of_time", np.float64),
            "column:total_parking_cost": (origin_duration * origin_land["PRKCST"].to_numpy(copy=False) + dest_duration * destination_land["PRKCST"].to_numpy(copy=False)) / 2.0,
            "column:density_index": np.where(outbound, destination_land["density_index"].to_numpy(copy=False), origin_land["density_index"].to_numpy(copy=False)),
            "column:origin_walk_time": float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]),
            "column:destination_walk_time": float(constants["shortWalk"]) * 60.0 / float(constants["walkSpeed"]),
            "column:origTaxiWaitTime": waits[0],
            "column:origSingleTNCWaitTime": waits[1],
            "column:origSharedTNCWaitTime": waits[2],
            "column:i_tour_mode": mode,
            "column:origin_density_index": np.where(outbound, origin_land["density_index"].to_numpy(copy=False), destination_land["density_index"].to_numpy(copy=False)),
        }
        auto = _values(frame, "auto_ownership", np.int64)
        participants = _values(frame, "number_of_participants", np.int64)
        int_values = {
            "name:auto_ownership": auto, "name:age": _values(frame, "age", np.int64),
            "name:is_joint": participants > 1, "name:is_atwork_subtour": ~np.asarray(frame["parent_tour_id"].isna()),
            "name:work_tour_is_SOV": np.isin(parent, ["DRIVEALONEFREE", "DRIVEALONEPAY"]),
            "name:number_of_participants": participants, "column:hhsize": _values(frame, "hhsize", np.int64),
            "column:trip_topology": np.where(outbound, destination_land["TOPOLOGY"].to_numpy(copy=False), origin_land["TOPOLOGY"].to_numpy(copy=False)),
            "name:work_tour_is_bike": parent == "BIKE", "name:outbound": outbound,
            "name:inbound": inbound, "name:tour_mode_is_auto": np.isin(mode, constants["I_AUTO_MODES"]),
            "name:tour_mode_is_walk": mode == int(constants["I_WALK_MODE"]),
            "name:tour_mode_is_bike": mode == int(constants["I_BIKE_MODE"]),
            "name:tour_mode_is_walk_transit": np.isin(mode, constants["I_WALK_TRANSIT_MODES"]),
            "name:tour_mode_is_drive_transit": np.isin(mode, constants["I_DRIVE_TRANSIT_MODES"]),
            "name:tour_mode_is_ride_hail": np.isin(mode, constants["I_RIDE_HAIL_MODES"]),
            "column:is_indiv": participants == 1, "column:tour_mode_is_SOV": np.isin(mode, constants["I_SOV_MODES"]),
            "column:tour_mode_is_SR2": np.isin(mode, constants["I_SR2_MODES"]),
            "column:tour_mode_is_SR3P": np.isin(mode, constants["I_SR3P_MODES"]),
            "column:first_trip": first,
        }
        availability_labels = set(_AVAILABILITY) | {"column:walk_ferry_available", "column:drive_ferry_available"}
        floats = np.column_stack([
            np.full(rows, float_values[label]) if np.isscalar(float_values[label]) else float_values[label]
            for label in self.float_labels
        ]).astype(np.float32, copy=False)
        ints = np.column_stack([
            np.zeros(rows, dtype=np.int64) if label in availability_labels else int_values[label]
            for label in self.int_labels
        ]).astype(np.int64, copy=False)
        built = time.perf_counter()
        self.invocation.float_inputs.set(floats)
        self.invocation.int_inputs.set(ints)
        # Populate every grouped coordinate vector using the declared direction.
        args = list(self.invocation.skim_arguments)
        position = self.invocation.logical_skim_bindings
        groups = self.invocation.skim_input_groups
        sources = self.invocation.skim_input_sources
        ranks = self.invocation.skim_input_ranks
        odt_coords = None
        for group in sorted(set(groups)):
            representative = groups.index(group)
            direction = sources[representative][1]
            rank = ranks[representative]
            if direction in {"odt_skims", "od_skims"}:
                o, d = origin, destination
            elif direction in {"dot_skims", "od_skims_reverse"}:
                o, d = destination, origin
            else:
                raise ValueError(f"trip native coordinate direction {direction!r} is unsupported")
            args[position].set(o); args[position + 1].set(d)
            if direction == "odt_skims":
                odt_coords = (args[position], args[position + 1], args[position + 2])
            position += 2
            if rank == 3:
                args[position].set(period); position += 1
            position += 1 + int(rank == 3)
        if position != len(args) or odt_coords is None:
            raise ValueError("trip native grouped skim ABI is malformed")
        self.cp.cuda.Stream.null.synchronize()
        uploaded = time.perf_counter()
        self._availability_kernel(
            ((rows + 255) // 256,), (256,),
            (self.invocation.float_inputs, self.invocation.int_inputs, *odt_coords,
             np.int64(rows), np.int32(len(self.float_labels)), np.int32(len(self.int_labels)),
             np.int64(len(land_use)), np.int64(5), *self._availability_args),
        )
        self.cp.cuda.Stream.null.synchronize()
        availability_done = time.perf_counter()
        utility_started = time.perf_counter()
        utilities = self.invocation.execute()
        self.cp.cuda.Stream.null.synchronize()
        utility_done = time.perf_counter()
        telemetry = TripLogsumNativeTelemetry(
            rows=rows,
            compact_host_bytes=int(floats.nbytes + ints.nbytes + origin.nbytes + destination.nbytes + period.nbytes),
            host_build_seconds=built - started,
            upload_seconds=uploaded - built,
            availability_kernel_seconds=availability_done - uploaded,
            utility_kernel_seconds=utility_done - utility_started,
        )
        return utilities, telemetry
