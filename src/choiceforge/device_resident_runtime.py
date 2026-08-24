"""Versioned, fail-closed execution for a device-resident model graph.

The runtime deliberately separates three boundaries:

* ingress may copy immutable model inputs to the device;
* sealed execution accepts and produces CUDA arrays only; and
* publication/checkpointing may copy explicitly named state back to the host.

It is an orchestration substrate rather than another choice kernel. Existing
qualified kernels remain responsible for modeled arithmetic; this module owns
their dependencies, table versions, lifecycle, telemetry, and restart record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cuda_backend import _cupy
from .gpu_native import ActivitySimRandomLedger, DeviceTable, GpuOnlyViolation


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ResidentStageRecord:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    host_launch_seconds: float
    device_seconds: float | None = None


@dataclass
class ResidentRuntimeTelemetry:
    ingress_calls: int = 0
    ingress_bytes: int = 0
    device_ingress_calls: int = 0
    device_ingress_bytes: int = 0
    publication_calls: int = 0
    publication_bytes: int = 0
    checkpoint_calls: int = 0
    checkpoint_bytes: int = 0
    forbidden_postseal_host_bytes: int = 0
    modeled_cpu_fallbacks: int = 0
    peak_persistent_state_bytes: int = 0
    stages: list[ResidentStageRecord] = field(default_factory=list)


class DeviceResidentRuntime:
    """Own a versioned CUDA table graph across multiple model components."""

    FORMAT_VERSION = 1

    def __init__(self, *, random_ledger: ActivitySimRandomLedger | None = None):
        self.cp = _cupy()
        self.tables: dict[str, DeviceTable] = {}
        self.versions: dict[str, int] = {}
        self.completed_stages: list[str] = []
        self.random_ledger = random_ledger or ActivitySimRandomLedger()
        self.telemetry = ResidentRuntimeTelemetry()
        self._sealed = False
        self._published = False
        self._pending_events: list[tuple[Any, Any, ResidentStageRecord]] = []

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def persistent_state_bytes(self) -> int:
        return sum(table.nbytes for table in self.tables.values())

    def _sample_state(self) -> None:
        self.telemetry.peak_persistent_state_bytes = max(
            self.telemetry.peak_persistent_state_bytes,
            self.persistent_state_bytes,
        )

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = str(name)
        if not normalized or normalized.startswith("_"):
            raise ValueError("resident table and stage names must be public nonempty names")
        return normalized

    def ingress_table(self, name: str, columns: Mapping[str, Any]) -> DeviceTable:
        name = self._validate_name(name)
        if self._sealed:
            attempted = sum(int(np.asarray(value).nbytes) for value in columns.values())
            self.telemetry.forbidden_postseal_host_bytes += attempted
            raise GpuOnlyViolation("device-resident ingress is sealed")
        if name in self.tables:
            raise ValueError(f"resident table {name!r} already exists")
        device = {
            str(column): self.cp.ascontiguousarray(self.cp.asarray(value))
            for column, value in columns.items()
        }
        table = DeviceTable(device)
        self.tables[name] = table
        self.versions[name] = 1
        self.telemetry.ingress_calls += 1
        self.telemetry.ingress_bytes += table.nbytes
        self._sample_state()
        return table

    def register_device_table(
        self, name: str, columns: Mapping[str, Any]
    ) -> DeviceTable:
        """Attach already-resident immutable CUDA state before sealing ingress.

        Large shared assets such as skim cubes are normally uploaded by a
        budget-aware cache loader.  Registering their existing device arrays
        avoids a second copy while keeping them inside the runtime's named,
        versioned state graph.  Host arrays and post-seal attachment fail
        closed under the same rules as ordinary ingress.
        """

        name = self._validate_name(name)
        if self._sealed:
            raise GpuOnlyViolation("device-resident ingress is sealed")
        if name in self.tables:
            raise ValueError(f"resident table {name!r} already exists")
        table = DeviceTable({str(column): value for column, value in columns.items()})
        self.tables[name] = table
        self.versions[name] = 1
        self.telemetry.device_ingress_calls += 1
        self.telemetry.device_ingress_bytes += table.nbytes
        self._sample_state()
        return table

    def seal_ingress(self) -> None:
        if not self.tables:
            raise RuntimeError("cannot seal an empty resident runtime")
        self._sealed = True

    def table(self, name: str) -> DeviceTable:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise KeyError(f"resident table {name!r} is unavailable") from exc

    def release_tables(self, *names: str) -> None:
        """Release intermediate device state after its final declared consumer."""

        if not self._sealed:
            raise GpuOnlyViolation("table lifecycle is only valid after sealed ingress")
        for name in names:
            normalized = self._validate_name(name)
            if normalized not in self.tables:
                raise KeyError(f"resident table {normalized!r} is unavailable")
        for name in names:
            del self.tables[str(name)]
        self._sample_state()

    def run_stage(
        self,
        name: str,
        *,
        reads: Sequence[str],
        writes: Sequence[str],
        operation: Callable[[Mapping[str, DeviceTable]], Mapping[str, Any]],
        replace: bool = False,
    ) -> Mapping[str, DeviceTable]:
        """Run one modeled component and atomically publish its CUDA outputs.

        ``operation`` receives only the declared input tables. It must return a
        mapping keyed by every declared output name. Each value may be a
        ``DeviceTable`` or a CUDA-column mapping. Host arrays fail closed.
        Results are committed only after every output validates.
        """

        name = self._validate_name(name)
        if not self._sealed:
            raise GpuOnlyViolation("seal ingress before modeled execution")
        read_names = tuple(self._validate_name(item) for item in reads)
        write_names = tuple(self._validate_name(item) for item in writes)
        if not write_names or len(set(write_names)) != len(write_names):
            raise ValueError("a stage must declare unique output tables")
        missing = [item for item in read_names if item not in self.tables]
        if missing:
            raise KeyError(f"stage {name!r} is missing inputs {missing}")
        conflicts = [item for item in write_names if item in self.tables]
        if conflicts and not replace:
            raise ValueError(f"stage {name!r} would overwrite tables {conflicts}")

        start_event = self.cp.cuda.Event()
        end_event = self.cp.cuda.Event()
        start_event.record()
        host_started = time.perf_counter()
        raw = operation({item: self.tables[item] for item in read_names})
        host_seconds = time.perf_counter() - host_started
        if set(raw) != set(write_names):
            raise ValueError(
                f"stage {name!r} returned {sorted(raw)}; expected {sorted(write_names)}"
            )
        pending: dict[str, DeviceTable] = {}
        for output_name in write_names:
            value = raw[output_name]
            pending[output_name] = (
                value if isinstance(value, DeviceTable) else DeviceTable(value)
            )
        end_event.record()

        for output_name, table in pending.items():
            self.tables[output_name] = table
            self.versions[output_name] = self.versions.get(output_name, 0) + 1
        record = ResidentStageRecord(
            name=name,
            reads=read_names,
            writes=write_names,
            host_launch_seconds=host_seconds,
        )
        self.telemetry.stages.append(record)
        self._pending_events.append((start_event, end_event, record))
        self.completed_stages.append(name)
        self._sample_state()
        return pending

    def synchronize(self) -> None:
        self.cp.cuda.Stream.null.synchronize()
        for start, end, record in self._pending_events:
            record.device_seconds = float(self.cp.cuda.get_elapsed_time(start, end) / 1000.0)
        self._pending_events.clear()

    def publish(
        self, selection: Mapping[str, Sequence[str] | None]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Download explicitly published outputs after the modeled graph completes."""

        if not self._sealed:
            raise GpuOnlyViolation("publication is only valid after sealed execution")
        self.synchronize()
        result: dict[str, dict[str, np.ndarray]] = {}
        for table_name, columns in selection.items():
            table = self.table(table_name)
            selected = tuple(columns) if columns is not None else tuple(table.columns)
            result[table_name] = {
                column: self.cp.asnumpy(table.columns[column]) for column in selected
            }
        byte_count = sum(
            value.nbytes for table in result.values() for value in table.values()
        )
        self.telemetry.publication_calls += 1
        self.telemetry.publication_bytes += int(byte_count)
        self._published = True
        return result

    def checkpoint(
        self,
        directory: Path | str,
        *,
        tables: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a self-contained, hash-verified host restart boundary."""

        if not self._sealed:
            raise GpuOnlyViolation("checkpointing is only valid after sealed execution")
        self.synchronize()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        table_manifest: dict[str, Any] = {}
        for table_position, table_name in enumerate(tables):
            table = self.table(table_name)
            columns: dict[str, Any] = {}
            for column_position, (column_name, value) in enumerate(table.columns.items()):
                storage_key = f"t{table_position:04d}_c{column_position:04d}"
                host = self.cp.asnumpy(value)
                arrays[storage_key] = host
                columns[column_name] = {
                    "storage_key": storage_key,
                    "dtype": str(host.dtype),
                    "shape": list(host.shape),
                    "sha256": _array_sha256(host),
                }
            table_manifest[table_name] = {
                "version": self.versions[table_name],
                "columns": columns,
            }
        archive = target / "state.npz"
        np.savez_compressed(archive, **arrays)
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "runtime": "choiceforge.device_resident",
            "tables": table_manifest,
            "completed_stages": list(self.completed_stages),
            "random_ledger": self.random_ledger.snapshot(),
            "state_archive": archive.name,
            "state_archive_sha256": _file_sha256(archive),
            "metadata": dict(metadata or {}),
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checkpoint_bytes = archive.stat().st_size + manifest_path.stat().st_size
        self.telemetry.checkpoint_calls += 1
        self.telemetry.checkpoint_bytes += checkpoint_bytes
        return manifest

    @classmethod
    def restore(cls, directory: Path | str) -> "DeviceResidentRuntime":
        target = Path(directory)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported resident checkpoint format")
        archive = target / manifest["state_archive"]
        if _file_sha256(archive) != manifest["state_archive_sha256"]:
            raise ValueError("resident checkpoint archive hash mismatch")
        runtime = cls(
            random_ledger=ActivitySimRandomLedger.restore(manifest["random_ledger"])
        )
        with np.load(archive) as loaded:
            for table_name, table_meta in manifest["tables"].items():
                columns: dict[str, np.ndarray] = {}
                for column_name, column_meta in table_meta["columns"].items():
                    value = loaded[column_meta["storage_key"]]
                    if list(value.shape) != column_meta["shape"]:
                        raise ValueError("resident checkpoint column shape mismatch")
                    if str(value.dtype) != column_meta["dtype"]:
                        raise ValueError("resident checkpoint column dtype mismatch")
                    if _array_sha256(value) != column_meta["sha256"]:
                        raise ValueError("resident checkpoint column hash mismatch")
                    columns[column_name] = value
                runtime.ingress_table(table_name, columns)
                runtime.versions[table_name] = int(table_meta["version"])
        runtime.completed_stages = [str(item) for item in manifest["completed_stages"]]
        runtime.seal_ingress()
        return runtime

    def assert_resident_contract(self) -> None:
        failures = {
            "forbidden_postseal_host_bytes": self.telemetry.forbidden_postseal_host_bytes,
            "modeled_cpu_fallbacks": self.telemetry.modeled_cpu_fallbacks,
        }
        if any(failures.values()):
            raise GpuOnlyViolation(f"device-resident contract failed: {failures}")

    def telemetry_dict(self) -> dict[str, Any]:
        self.synchronize()
        return asdict(self.telemetry)

    def cpu_fallback(self, stage_name: str) -> None:
        self.telemetry.modeled_cpu_fallbacks += 1
        raise GpuOnlyViolation(f"CPU fallback is forbidden for {stage_name!r}")
