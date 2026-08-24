"""Budgeted public-OMX skim cache for sealed device-resident execution.

The ActivitySim MTC mode-choice specification references a small logical hot
set inside a much larger OMX collection.  This module derives that set from
the reviewed strict IR, loads only those matrices as float32, deduplicates
directional aliases, and exposes an exact all-binding probe used by the public
qualification benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np

from .cuda_backend import _cupy
from .gpu_native import GpuOnlyViolation, _is_cuda_array


MTC_PERIODS = ("EA", "AM", "MD", "PM", "EV")


@dataclass(frozen=True, order=True)
class LogicalSkim:
    direction: str
    key: str
    rank: int

    @property
    def physical_key(self) -> str:
        # DOT is a reversed lookup into the same time-dependent cube as ODT.
        return f"time:{self.key}" if self.rank == 3 else f"static:{self.key}"


@dataclass(frozen=True)
class HotSkimTelemetry:
    logical_bindings: int
    physical_cubes: int
    source_float64_bytes: int
    resident_float32_bytes: int
    budget_bytes: int
    disk_read_seconds: float
    device_upload_seconds: float
    matrix_sha256: Mapping[str, str]


def _walk(node: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk(value)


def logical_skims_from_ir(document: Mapping[str, Any]) -> tuple[LogicalSkim, ...]:
    """Extract the exact logical skim bindings referenced by strict IR."""

    found: set[LogicalSkim] = set()
    for node in _walk(document.get("terms", ())):
        if node.get("op") != "skim":
            continue
        direction = str(node.get("direction"))
        key_node = node.get("key", {})
        if not isinstance(key_node, Mapping) or key_node.get("op") != "const":
            raise ValueError("resident skim cache requires constant skim keys")
        if direction not in {
            "od_skims", "od_skims_reverse", "odt_skims", "dot_skims",
            "odr_skims", "dor_skims",
        }:
            raise ValueError(f"unsupported resident skim direction {direction!r}")
        found.add(
            LogicalSkim(
                direction=direction,
                key=str(key_node["value"]),
                rank=2 if direction in {"od_skims", "od_skims_reverse"} else 3,
            )
        )
    if not found:
        raise ValueError("strict IR does not reference any skims")
    return tuple(sorted(found))


def _matrix_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


class ResidentOmxSkimCache:
    """Host/device hot set loaded from one public OMX file under a hard budget."""

    def __init__(
        self,
        *,
        logical: tuple[LogicalSkim, ...],
        host_cubes: Mapping[str, np.ndarray] | None,
        device_cubes: Mapping[str, Any],
        zone_count: int,
        telemetry: HotSkimTelemetry,
    ) -> None:
        self.logical = logical
        self.host_cubes = dict(host_cubes or {})
        self.device_cubes = dict(device_cubes)
        self.zone_count = int(zone_count)
        self.telemetry = telemetry

    @classmethod
    def load(
        cls,
        omx_path: Path | str,
        document: Mapping[str, Any],
        *,
        budget_bytes: int,
        keep_host: bool = True,
    ) -> "ResidentOmxSkimCache":
        import h5py

        cp = _cupy()
        omx_path = Path(omx_path)
        logical = logical_skims_from_ir(document)
        physical: dict[str, LogicalSkim] = {}
        for item in logical:
            physical.setdefault(item.physical_key, item)

        with h5py.File(omx_path, "r") as omx:
            data = omx["data"]
            first = next(iter(data.values()))
            if first.ndim != 2 or first.shape[0] != first.shape[1]:
                raise ValueError("OMX skim matrices must be square")
            zones = int(first.shape[0])
            resident_bytes = sum(
                zones * zones * np.dtype(np.float32).itemsize * (5 if x.rank == 3 else 1)
                for x in physical.values()
            )
            source_bytes = resident_bytes * 2
            if resident_bytes > int(budget_bytes):
                raise MemoryError(
                    "hot skim set exceeds budget: "
                    f"required={resident_bytes} budget={int(budget_bytes)}"
                )
            required_names = []
            for item in physical.values():
                required_names.extend(
                    [f"{item.key}__{period}" for period in MTC_PERIODS]
                    if item.rank == 3
                    else [item.key]
                )
            missing = sorted(set(required_names).difference(data.keys()))
            if missing:
                raise KeyError(f"OMX is missing required hot skims: {missing[:5]}")

            host_cubes: dict[str, np.ndarray] = {}
            device_cubes: dict[str, Any] = {}
            hashes: dict[str, str] = {}
            disk_seconds = 0.0
            upload_seconds = 0.0
            for physical_key, item in sorted(physical.items()):
                read_started = time.perf_counter()
                if item.rank == 2:
                    host = np.asarray(data[item.key], dtype=np.float32)
                else:
                    host = np.empty((zones, zones, len(MTC_PERIODS)), dtype=np.float32)
                    for period_number, period in enumerate(MTC_PERIODS):
                        host[:, :, period_number] = np.asarray(
                            data[f"{item.key}__{period}"], dtype=np.float32
                        )
                host = np.ascontiguousarray(host)
                disk_seconds += time.perf_counter() - read_started
                hashes[physical_key] = _matrix_hash(host)
                upload_started = time.perf_counter()
                device = cp.ascontiguousarray(cp.asarray(host))
                cp.cuda.Stream.null.synchronize()
                upload_seconds += time.perf_counter() - upload_started
                device_cubes[physical_key] = device
                if keep_host:
                    host_cubes[physical_key] = host

        actual_bytes = sum(int(value.nbytes) for value in device_cubes.values())
        if actual_bytes != resident_bytes:
            raise AssertionError("resident hot-skim byte accounting changed")
        return cls(
            logical=logical,
            host_cubes=host_cubes,
            device_cubes=device_cubes,
            zone_count=zones,
            telemetry=HotSkimTelemetry(
                logical_bindings=len(logical),
                physical_cubes=len(physical),
                source_float64_bytes=source_bytes,
                resident_float32_bytes=resident_bytes,
                budget_bytes=int(budget_bytes),
                disk_read_seconds=disk_seconds,
                device_upload_seconds=upload_seconds,
                matrix_sha256=hashes,
            ),
        )

    def runtime_columns(self) -> dict[str, Any]:
        """Return stable column names for zero-copy runtime registration."""

        return {
            f"cube_{number:03d}": self.device_cubes[key]
            for number, key in enumerate(sorted(self.device_cubes))
        }

    def _validate_probe_inputs(self, origin, destination, out_period, in_period) -> int:
        if not all(_is_cuda_array(x) for x in (origin, destination, out_period, in_period)):
            raise GpuOnlyViolation("resident skim probe inputs must stay on CUDA")
        shapes = {tuple(x.shape) for x in (origin, destination, out_period, in_period)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise ValueError("resident skim probe inputs must be equal 1-D arrays")
        return int(origin.size)

    @staticmethod
    def _probe_source(logical: tuple[LogicalSkim, ...]) -> str:
        parameters = [f"const float* cube_{i}" for i in range(len(logical))]
        lines = []
        for number, item in enumerate(logical):
            if item.rank == 2:
                left, right = (
                    ("d", "o") if item.direction == "od_skims_reverse" else ("o", "d")
                )
                index = f"{left} * zones + {right}"
            else:
                period = (
                    "pin" if item.direction in {"dot_skims", "odr_skims"} else "pout"
                )
                left, right = (
                    ("d", "o")
                    if item.direction in {"dot_skims", "dor_skims"}
                    else ("o", "d")
                )
                index = f"(({left} * zones + {right}) * 5 + {period})"
            salt = (0x9E3779B9 + number * 0x85EBCA6B) & 0xFFFFFFFF
            lines.append(
                f"unsigned int b{number}=__float_as_uint(cube_{number}[{index}]); "
                f"h1=(h1 ^ ((unsigned long long)b{number}+{salt}ULL))*1099511628211ULL; "
                f"h2+=((unsigned long long)(b{number}^{salt}))*14029467366897019727ULL; "
                "h2=(h2<<13)|(h2>>51);"
            )
        return """
extern "C" __global__ void resident_skim_probe(
const long long* origin, const long long* destination,
const int* out_period, const int* in_period, long long rows, long long zones,
unsigned long long* hash1, unsigned long long* hash2,
""" + ",\n".join(parameters) + ") {\n" + """
long long row=(long long)blockDim.x*blockIdx.x+threadIdx.x;
if(row>=rows) return;
long long o=origin[row], d=destination[row];
int pout=out_period[row], pin=in_period[row];
unsigned long long h1=1469598103934665603ULL;
unsigned long long h2=7809847782465536322ULL;
""" + "\n".join(lines) + "\nhash1[row]=h1; hash2[row]=h2;\n}\n"

    def probe_gpu(self, origin, destination, out_period, in_period):
        """Read every logical binding for every row and return exact bit hashes."""

        cp = _cupy()
        rows = self._validate_probe_inputs(origin, destination, out_period, in_period)
        for values, upper, name in (
            (origin, self.zone_count, "origin"),
            (destination, self.zone_count, "destination"),
            (out_period, 5, "out_period"),
            (in_period, 5, "in_period"),
        ):
            if bool(cp.any((values < 0) | (values >= upper)).item()):
                raise ValueError(f"{name} contains an out-of-range position")
        if not hasattr(self, "_probe_kernel"):
            self._probe_kernel = cp.RawKernel(
                self._probe_source(self.logical), "resident_skim_probe",
                options=("--std=c++11", "--fmad=false"),
            )
            self._probe_kernel.compile()
        result1 = cp.empty(rows, dtype=cp.uint64)
        result2 = cp.empty(rows, dtype=cp.uint64)
        arrays = tuple(self.device_cubes[item.physical_key] for item in self.logical)
        self._probe_kernel(
            ((rows + 255) // 256,), (256,),
            (
                origin, destination, out_period, in_period,
                np.int64(rows), np.int64(self.zone_count), result1, result2,
            ) + arrays,
        )
        return result1, result2

    def probe_cpu(self, origin, destination, out_period, in_period):
        """Independent NumPy implementation of the same all-binding proof."""

        if not self.host_cubes:
            raise RuntimeError("CPU proof requires keep_host=True")
        origin = np.asarray(origin, dtype=np.int64)
        destination = np.asarray(destination, dtype=np.int64)
        out_period = np.asarray(out_period, dtype=np.int32)
        in_period = np.asarray(in_period, dtype=np.int32)
        hash1 = np.full(origin.size, 1469598103934665603, dtype=np.uint64)
        hash2 = np.full(origin.size, 7809847782465536322, dtype=np.uint64)
        prime1 = np.uint64(1099511628211)
        prime2 = np.uint64(14029467366897019727)
        for number, item in enumerate(self.logical):
            cube = self.host_cubes[item.physical_key]
            if item.rank == 2:
                values = (
                    cube[destination, origin]
                    if item.direction == "od_skims_reverse"
                    else cube[origin, destination]
                )
            elif item.direction in {"dot_skims", "dor_skims"}:
                period = in_period if item.direction == "dot_skims" else out_period
                values = cube[destination, origin, period]
            else:
                period = in_period if item.direction == "odr_skims" else out_period
                values = cube[origin, destination, period]
            bits = values.view(np.uint32).astype(np.uint64)
            salt = np.uint64((0x9E3779B9 + number * 0x85EBCA6B) & 0xFFFFFFFF)
            hash1 = (hash1 ^ (bits + salt)) * prime1
            hash2 += (bits ^ salt) * prime2
            hash2 = (hash2 << np.uint64(13)) | (hash2 >> np.uint64(51))
        return hash1, hash2
