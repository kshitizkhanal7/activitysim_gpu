"""Native raw-trip input generation for trip-destination mode-choice logsums.

The legacy path expands a large pandas preprocessor table before evaluating
the 21-alternative trip-mode utility.  This module implements the reviewed
Prototype MTC preprocessor contract directly: compact raw trip columns and
three controlled normal draws are packed once, availability flags are formed
on CUDA from resident skims, and the existing strict utility invocation reads
the resulting 11-float/45-integer ABI. Unknown source labels fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .cuda_backend import _cupy


_AVAILABILITY_KERNEL_CACHE = {}

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


class TripLogsumNativePlan:
    """Populate and execute one strict trip-mode utility invocation."""

    def __init__(self, invocation):
        self.invocation = invocation
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
