"""Versioned, byte-verified native skim store for Phase 31.

The store contains only the physical float32 cubes referenced by reviewed
utility IR.  Directional logical bindings share one physical cube.  A loader
double-buffers the payload through pinned host memory, verifies every byte in
payload order, and overlaps CUDA upload with the next host read without asking
ActivitySim or Sharrow to materialize their full skim dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .cuda_backend import _cupy
from .resident_skim_cache import MTC_PERIODS, LogicalSkim, logical_skims_from_ir


FORMAT = "choiceforge.native-skim-store.v1"
PAYLOAD_NAME = "payload.f32"
MANIFEST_NAME = "manifest.json"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _zone_sha256(zone_ids) -> str:
    return _array_sha256(np.ascontiguousarray(zone_ids, dtype=np.int64))


def skim_contract(document: Mapping[str, Any]) -> Mapping[str, Any]:
    logical = logical_skims_from_ir(document)
    return {
        "periods": list(MTC_PERIODS),
        "logical": [
            {"direction": item.direction, "key": item.key, "rank": item.rank}
            for item in logical
        ],
        "physical": sorted({item.physical_key for item in logical}),
    }


def skim_contract_sha256(document: Mapping[str, Any]) -> str:
    return _canonical_sha256(skim_contract(document))


def _physical_skims(document: Mapping[str, Any]) -> tuple[LogicalSkim, ...]:
    physical: dict[str, LogicalSkim] = {}
    for item in logical_skims_from_ir(document):
        physical.setdefault(item.physical_key, item)
    return tuple(physical[key] for key in sorted(physical))


@dataclass(frozen=True)
class NativeSkimStoreTelemetry:
    manifest_seconds: float
    verified_read_seconds: float
    device_upload_seconds: float
    total_load_seconds: float
    payload_bytes: int
    logical_bindings: int
    physical_cubes: int
    zone_count: int
    payload_sha256: str
    skim_contract_sha256: str
    verified_payload_bytes: int
    source: str


class NativeSkimStore:
    """Validated CUDA cubes loaded from one immutable Phase 31 artifact."""

    def __init__(self, path: Path, manifest, device_cubes, telemetry) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.device_cubes = dict(device_cubes)
        self.telemetry = telemetry
        self.zone_count = int(manifest["zone_count"])

    @classmethod
    def load(
        cls,
        path: Path | str,
        document: Mapping[str, Any],
        zone_ids,
        *,
        budget_bytes: int,
    ) -> "NativeSkimStore":
        started = time.perf_counter()
        path = Path(path)
        manifest_path = path / MANIFEST_NAME
        payload_path = path / PAYLOAD_NAME
        manifest_started = time.perf_counter()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("native skim store format is unsupported")
        manifest_core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if _canonical_sha256(manifest_core) != manifest.get("manifest_sha256"):
            raise ValueError("native skim store manifest hash differs")
        expected_contract = skim_contract_sha256(document)
        if manifest.get("skim_contract_sha256") != expected_contract:
            raise ValueError("native skim store skim contract differs from reviewed IR")
        if manifest.get("zone_sha256") != _zone_sha256(zone_ids):
            raise ValueError("native skim store zone identity/order differs")
        payload_bytes = int(manifest["payload_bytes"])
        if payload_bytes > int(budget_bytes):
            raise MemoryError(
                f"native skim store exceeds budget: required={payload_bytes} "
                f"budget={int(budget_bytes)}"
            )
        if payload_path.stat().st_size != payload_bytes:
            raise ValueError("native skim store payload size differs")
        entries = manifest.get("entries", [])
        if len(entries) != len(manifest_core["physical_keys"]):
            raise ValueError("native skim store entry count differs")
        if [item["physical_key"] for item in entries] != manifest_core["physical_keys"]:
            raise ValueError("native skim store physical key order differs")
        manifest_seconds = time.perf_counter() - manifest_started

        cp = _cupy()
        max_bytes = max(int(item["nbytes"]) for item in entries)
        # Two reusable pinned blocks avoid 149 allocation cycles and permit the
        # CPU read/hash of cube N+1 to overlap the H2D transfer of cube N.
        pinned_blocks = [cp.cuda.alloc_pinned_memory(max_bytes) for _ in range(2)]
        pinned_bytes = [
            np.frombuffer(block, dtype=np.uint8, count=max_bytes)
            for block in pinned_blocks
        ]
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        payload_digest = hashlib.sha256()
        device_cubes = {}
        verified_read_seconds = 0.0
        upload_seconds = 0.0
        contiguous_offset = 0
        payload_stream = payload_path.open("rb", buffering=0)
        for entry_number, entry in enumerate(entries):
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
            shape = tuple(int(value) for value in entry["shape"])
            if offset != contiguous_offset or nbytes != int(np.prod(shape)) * 4:
                raise ValueError("native skim store payload layout is not contiguous float32")
            slot = entry_number % 2
            upload_wait_started = time.perf_counter()
            streams[slot].synchronize()
            upload_seconds += time.perf_counter() - upload_wait_started
            read_started = time.perf_counter()
            bytes_read = payload_stream.readinto(pinned_bytes[slot][:nbytes])
            if bytes_read != nbytes:
                raise ValueError("native skim store payload ended before its manifest")
            host = pinned_bytes[slot][:nbytes].view(np.float32).reshape(shape)
            # The aggregate digest covers every payload byte and the hashed
            # manifest fixes all cube boundaries, shapes, and keys.  Per-cube
            # digests are retained for build provenance and error diagnosis;
            # hashing every cube a second time on the success path is redundant.
            payload_digest.update(host.view(np.uint8))
            verified_read_seconds += time.perf_counter() - read_started
            upload_started = time.perf_counter()
            device = cp.empty(shape, dtype=cp.float32)
            cp.cuda.runtime.memcpyAsync(
                int(device.data.ptr),
                int(host.ctypes.data),
                nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                streams[slot].ptr,
            )
            upload_seconds += time.perf_counter() - upload_started
            device_cubes[entry["physical_key"]] = device
            contiguous_offset += nbytes
        upload_wait_started = time.perf_counter()
        for stream in streams:
            stream.synchronize()
        upload_seconds += time.perf_counter() - upload_wait_started
        payload_stream.close()
        actual_payload_sha256 = payload_digest.hexdigest()
        if actual_payload_sha256 != manifest["payload_sha256"]:
            # This slow path is never paid by a valid artifact.  Locate the
            # corrupt cube so a fail-closed report remains actionable.
            corrupt_key = "unknown"
            for entry in entries:
                mapped = np.memmap(
                    payload_path,
                    mode="r",
                    dtype=np.float32,
                    offset=int(entry["offset"]),
                    shape=tuple(int(value) for value in entry["shape"]),
                    order="C",
                )
                if _array_sha256(mapped) != entry["sha256"]:
                    corrupt_key = entry["physical_key"]
                    break
            raise ValueError(
                f"native skim store cube {corrupt_key!r} is corrupt; "
                "aggregate payload hash differs"
            )
        actual_device_bytes = sum(int(value.nbytes) for value in device_cubes.values())
        if actual_device_bytes != payload_bytes:
            raise AssertionError("native skim store device byte accounting changed")
        telemetry = NativeSkimStoreTelemetry(
            manifest_seconds=manifest_seconds,
            verified_read_seconds=verified_read_seconds,
            device_upload_seconds=upload_seconds,
            total_load_seconds=time.perf_counter() - started,
            payload_bytes=payload_bytes,
            logical_bindings=int(manifest["logical_bindings"]),
            physical_cubes=len(entries),
            zone_count=int(manifest["zone_count"]),
            payload_sha256=actual_payload_sha256,
            skim_contract_sha256=expected_contract,
            verified_payload_bytes=payload_bytes,
            source=str(path.resolve()),
        )
        return cls(path, manifest, device_cubes, telemetry)

    def cube(self, source) -> tuple[Any, int, int, int]:
        _, direction, key = source
        rank = 2 if direction in {"od_skims", "od_skims_reverse"} else 3
        physical_key = f"static:{key}" if rank == 2 else f"time:{key}"
        try:
            data = self.device_cubes[physical_key]
        except KeyError as exc:
            raise KeyError(f"native skim store lacks {physical_key!r}") from exc
        return data, self.zone_count, 1 if rank == 2 else len(MTC_PERIODS), rank

    def telemetry_dict(self) -> Mapping[str, Any]:
        return asdict(self.telemetry)


def build_native_skim_store(
    omx_path: Path | str,
    land_use_path: Path | str,
    document: Mapping[str, Any],
    output: Path | str,
) -> Mapping[str, Any]:
    """Build one immutable artifact, refusing to overwrite any existing path."""
    import h5py
    import pandas as pd

    omx_path = Path(omx_path)
    land_use_path = Path(land_use_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native skim store {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    payload_partial = output / f"{PAYLOAD_NAME}.partial"
    manifest_partial = output / f"{MANIFEST_NAME}.partial"
    zone_frame = pd.read_csv(land_use_path, usecols=["TAZ"])
    zone_ids = zone_frame["TAZ"].to_numpy(dtype=np.int64)
    if zone_ids.size == 0 or np.unique(zone_ids).size != zone_ids.size:
        raise ValueError("native skim store requires unique nonempty TAZ values")
    source_positions = zone_ids - 1
    if np.any(source_positions < 0):
        raise ValueError("native skim store requires positive one-based source TAZ values")

    physical = _physical_skims(document)
    entries = []
    payload_digest = hashlib.sha256()
    offset = 0
    with h5py.File(omx_path, "r") as omx, payload_partial.open("xb") as payload:
        data = omx["data"]
        first = next(iter(data.values()))
        source_zones = int(first.shape[0])
        if first.ndim != 2 or first.shape[1] != source_zones:
            raise ValueError("native skim store source matrices must be square")
        if np.any(source_positions >= source_zones):
            raise ValueError("land-use TAZ is outside the OMX source")
        positions = np.asarray(source_positions, dtype=np.int64)
        for item in physical:
            required = (
                [item.key]
                if item.rank == 2
                else [f"{item.key}__{period}" for period in MTC_PERIODS]
            )
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"OMX is missing native skim {missing[0]!r}")
            if item.rank == 2:
                source = np.asarray(data[item.key], dtype=np.float32)
                cube = source[np.ix_(positions, positions)]
            else:
                cube = np.empty(
                    (zone_ids.size, zone_ids.size, len(MTC_PERIODS)),
                    dtype=np.float32,
                )
                for period_number, period in enumerate(MTC_PERIODS):
                    source = np.asarray(
                        data[f"{item.key}__{period}"], dtype=np.float32
                    )
                    cube[:, :, period_number] = source[np.ix_(positions, positions)]
            cube = np.ascontiguousarray(cube, dtype=np.float32)
            raw = cube.view(np.uint8)
            payload.write(raw)
            payload_digest.update(raw)
            entries.append(
                {
                    "physical_key": item.physical_key,
                    "source_key": item.key,
                    "rank": item.rank,
                    "shape": list(cube.shape),
                    "dtype": "float32",
                    "offset": offset,
                    "nbytes": int(cube.nbytes),
                    "sha256": _array_sha256(cube),
                }
            )
            offset += int(cube.nbytes)
        payload.flush()
        os.fsync(payload.fileno())

    contract = skim_contract(document)
    manifest_core = {
        "format": FORMAT,
        "payload": PAYLOAD_NAME,
        "payload_bytes": offset,
        "payload_sha256": payload_digest.hexdigest(),
        "zone_count": int(zone_ids.size),
        "zone_min": int(zone_ids.min()),
        "zone_max": int(zone_ids.max()),
        "zone_sha256": _zone_sha256(zone_ids),
        "periods": list(MTC_PERIODS),
        "logical_bindings": len(contract["logical"]),
        "physical_keys": contract["physical"],
        "skim_contract_sha256": _canonical_sha256(contract),
        "source_omx_bytes": int(omx_path.stat().st_size),
        "source_omx_sha256": _file_sha256(omx_path),
        "source_land_use_sha256": _file_sha256(land_use_path),
        "entries": entries,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": _canonical_sha256(manifest_core),
    }
    manifest_partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(payload_partial, output / PAYLOAD_NAME)
    os.replace(manifest_partial, output / MANIFEST_NAME)
    return manifest
