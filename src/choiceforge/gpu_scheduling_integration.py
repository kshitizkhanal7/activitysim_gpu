"""Continuous raw-logsum-to-scheduling CUDA handoff for Phase 22.

The upstream utility compiler and nested-logit reducer produce one logsum for
each unique tour/skim-period pair.  This module scatters those device values
into the compact 5-by-5 cache and immediately consumes the cache with the
qualified timetable, scheduling-expression, probability, and mutation path.
No modeled logsum is materialized on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .cuda_backend import _cupy
from .gpu_native import GpuOnlyViolation, _is_cuda_array
from .gpu_scheduling_pipeline import GpuSchedulingPreparer, skim_period_code
from .scheduling_compiler import CompiledCudaSchedulingModel, SchedulingSchema


@dataclass(frozen=True)
class DeviceLogsumBatch:
    chooser_ids: np.ndarray
    cache: Any
    present: Any
    raw_cache: Any
    source_rows: int
    cache_build_ms: float


@dataclass(frozen=True)
class CompiledDeviceLogsumScatter:
    """Immutable device plan for repeated logsum-to-5x5 cache scatter."""

    chooser_ids: np.ndarray
    source_rows: int
    unique_flat: Any
    first_positions: Any
    cache: Any
    raw_cache: Any
    present: Any
    plan_device_bytes: int

    @classmethod
    def compile(
        cls,
        metadata: Mapping[str, Any],
        expected_chooser_ids: Any,
        *,
        reference_logsums: Any | None = None,
    ) -> "CompiledDeviceLogsumScatter":
        """Resolve identities and slots once, before resident execution."""
        cp = _cupy()
        chooser_ids, flat = _logsum_scatter_layout(metadata, expected_chooser_ids)
        expected = np.asarray(expected_chooser_ids, dtype=np.int64)
        unique_flat, first_positions, inverse = np.unique(
            flat, return_index=True, return_inverse=True
        )
        device_first = cp.asarray(first_positions)
        if reference_logsums is not None and first_positions.size != flat.size:
            values = cp.asarray(reference_logsums)
            repeated = values[device_first][cp.asarray(inverse)]
            if bool(cp.any(values != repeated).item()):
                raise ValueError(
                    "raw-skim logsum values differ within a duplicate tour/period slot"
                )
        device_flat = cp.asarray(unique_flat)
        cache = cp.zeros((expected.size, 25), dtype=cp.float32)
        raw_cache = cp.zeros((expected.size, 25), dtype=cp.float64)
        present = cp.zeros((expected.size, 25), dtype=cp.bool_)
        present.reshape(-1)[device_flat] = True
        cp.cuda.Stream.null.synchronize()
        return cls(
            chooser_ids=expected.copy(),
            source_rows=int(flat.size),
            unique_flat=device_flat,
            first_positions=device_first,
            cache=cache,
            raw_cache=raw_cache,
            present=present,
            plan_device_bytes=int(device_flat.nbytes + device_first.nbytes),
        )

    def execute(self, device_logsums: Any) -> DeviceLogsumBatch:
        """Scatter with no host layout work, allocation, or modeled transfer."""
        if not _is_cuda_array(device_logsums):
            raise GpuOnlyViolation("resident scatter requires CUDA logsums")
        if int(device_logsums.size) != self.source_rows:
            raise ValueError("device logsum count differs from compiled scatter plan")
        cp = _cupy()
        started = time.perf_counter()
        values = cp.asarray(device_logsums)
        unique_values = values[self.first_positions]
        self.cache.fill(0)
        self.raw_cache.fill(0)
        self.cache.reshape(-1)[self.unique_flat] = cp.asarray(
            unique_values, dtype=cp.float32
        )
        self.raw_cache.reshape(-1)[self.unique_flat] = cp.asarray(
            unique_values, dtype=cp.float64
        )
        cp.cuda.Stream.null.synchronize()
        return DeviceLogsumBatch(
            self.chooser_ids,
            self.cache,
            self.present,
            self.raw_cache,
            self.source_rows,
            (time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class IntegratedBatchTelemetry:
    batch: int
    trace_label: str
    choosers: int
    logsum_rows: int
    cache_build_ms: float
    scheduling_ms: float
    cache_value_mismatches: int
    cache_max_abs_difference: float
    cache_presence_mismatches: int
    random_draw_mismatches: int
    tdd_mismatches: int
    boundary_rows: int
    boundary_logsum_download_bytes: int
    device_boundary_adjudications: int = 0
    device_boundary_corrections: int = 0


def array_sha256(value: Any) -> str:
    """Hash dtype, shape, and contiguous bytes for restart evidence."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def assemble_device_logsum_cache(
    device_logsums: Any,
    metadata: Mapping[str, Any],
    expected_chooser_ids: Any,
) -> DeviceLogsumBatch:
    """Scatter deduplicated CUDA logsums into an ordered per-tour cache.

    Only identities and time labels are inspected on the host. The modeled
    logsum values remain on CUDA and are converted to the scheduling ABI's
    float32 representation there.
    """

    if not _is_cuda_array(device_logsums):
        raise GpuOnlyViolation("raw-skim logsums must reside on the GPU")
    cp = _cupy()
    chooser_ids, flat = _logsum_scatter_layout(metadata, expected_chooser_ids)
    if int(device_logsums.size) != chooser_ids.size:
        raise ValueError("device logsum count differs from its row metadata")
    unique_flat, first_positions, inverse = np.unique(
        flat, return_index=True, return_inverse=True
    )

    started = time.perf_counter()
    cache = cp.zeros((np.asarray(expected_chooser_ids).size, 25), dtype=cp.float32)
    raw_cache = cp.zeros((np.asarray(expected_chooser_ids).size, 25), dtype=cp.float64)
    present = cp.zeros((np.asarray(expected_chooser_ids).size, 25), dtype=cp.bool_)
    device_values = cp.asarray(device_logsums)
    device_first = cp.asarray(first_positions)
    if first_positions.size != flat.size:
        repeated_reference = device_values[device_first][cp.asarray(inverse)]
        if bool(cp.any(device_values != repeated_reference).item()):
            raise ValueError(
                "raw-skim logsum values differ within a duplicate tour/period slot"
            )
    device_flat = cp.asarray(unique_flat)
    unique_values = device_values[device_first]
    cache.reshape(-1)[device_flat] = cp.asarray(unique_values, dtype=cp.float32)
    raw_cache.reshape(-1)[device_flat] = cp.asarray(unique_values, dtype=cp.float64)
    present.reshape(-1)[device_flat] = True
    cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return DeviceLogsumBatch(
        np.asarray(expected_chooser_ids, dtype=np.int64).copy(),
        cache, present, raw_cache, chooser_ids.size, elapsed_ms
    )


def _logsum_scatter_layout(metadata, expected_chooser_ids):
    """Validate host identity metadata and return its flat cache positions."""
    chooser_ids = np.asarray(metadata["chooser_ids"], dtype=np.int64)
    starts = np.asarray(metadata["start"], dtype=np.int16)
    ends = np.asarray(metadata["end"], dtype=np.int16)
    expected = np.asarray(expected_chooser_ids, dtype=np.int64)
    if chooser_ids.ndim != 1 or starts.shape != chooser_ids.shape or ends.shape != chooser_ids.shape:
        raise ValueError("logsum metadata arrays must be equal one-dimensional vectors")
    if chooser_ids.size == 0:
        raise ValueError("a scheduling logsum batch cannot be empty")

    first = np.r_[True, chooser_ids[1:] != chooser_ids[:-1]]
    observed = chooser_ids[first]
    if not np.array_equal(observed, expected):
        raise ValueError("live raw-skim chooser order differs from the scheduling batch")
    owners = np.cumsum(first, dtype=np.int32) - 1
    if "out_period" in metadata and "in_period" in metadata:
        period_codes = {"EA": 0, "AM": 1, "MD": 2, "PM": 3, "EV": 4}
        try:
            outbound = np.asarray(
                [period_codes[str(value)] for value in metadata["out_period"]],
                dtype=np.int32,
            )
            inbound = np.asarray(
                [period_codes[str(value)] for value in metadata["in_period"]],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError(f"unknown raw-skim period label {exc.args[0]!r}") from exc
        slots = outbound * np.int32(5) + inbound
    else:
        slots = (
            skim_period_code(starts).astype(np.int32) * np.int32(5)
            + skim_period_code(ends).astype(np.int32)
        )
    flat = owners.astype(np.int64) * np.int64(25) + slots.astype(np.int64)
    return chooser_ids, flat


class IntegratedGpuMandatoryScheduler:
    """Six-batch GPU scheduler fed directly by device logsum vectors."""

    def __init__(
        self,
        artifact: Path | str,
        *,
        qualify_against_artifact: bool = True,
        cache_absolute_tolerance: float = 1.0e-5,
        device_boundary_reference: bool = False,
    ):
        cp = _cupy()
        self.artifact = Path(artifact)
        self.manifest = json.loads((self.artifact / "manifest.json").read_text())
        with np.load(self.artifact / self.manifest["common_file"]) as loaded:
            self.person_ids = loaded["person_ids"]
            self.alternative_values_host = loaded["alternative_values"]
        self.alternative_values = cp.asarray(self.alternative_values_host)
        self.preparer = GpuSchedulingPreparer(
            int(self.manifest["person_count"]), self.alternative_values
        )
        self.qualify_against_artifact = bool(qualify_against_artifact)
        self.cache_absolute_tolerance = float(cache_absolute_tolerance)
        self.device_boundary_reference = bool(device_boundary_reference)
        self.boundary_tolerance = 2.0e-6
        self.batches = []
        for meta in self.manifest["batches"]:
            with np.load(self.artifact / meta["file"]) as loaded:
                host = {name: loaded[name] for name in loaded.files}
            schema = SchedulingSchema(
                tuple(meta["chooser_columns"]),
                tuple(meta["row_columns"]),
                tuple(meta["alternative_columns"]),
            )
            self.batches.append(
                {
                    "meta": meta,
                    "host": host,
                    "model": CompiledCudaSchedulingModel(
                        meta["expressions"],
                        host["coefficients"],
                        schema,
                        # The public MTC settings enable skip_failed_choices.
                        # ActivitySim therefore disables max-shift overflow
                        # protection before exponentiation. Near a cumulative
                        # probability boundary this rounding policy is part of
                        # the reproducible answer, not an optional optimization.
                        overflow_protection=False,
                        chooser_float64=True,
                        dot_policy="sharrow65_lane4",
                    ),
                    "device": {
                        name: cp.asarray(host[name])
                        for name in (
                            "person_rows",
                            "chooser_values",
                            "draws",
                            "mode_logsum_cache",
                            "mode_logsum_present",
                            "expected_tdd",
                        )
                    },
                }
            )
        self.boundary_map_entries = 0
        if self.device_boundary_reference:
            self.qualify_device_boundary_maps()
        self.cursor = 0
        self.pending: DeviceLogsumBatch | None = None
        self.telemetry: list[IntegratedBatchTelemetry] = []
        self.selected_batches = []
        self.preparer.reset()

    def qualify_device_boundary_maps(
        self, compiled_caches: list[DeviceLogsumBatch] | None = None
    ) -> None:
        """Build sparse Sharrow decision maps before resident execution.

        Only kernel-detected ambiguity positions receive a reference label.
        Every other position remains -1, so a new ambiguity after any input
        change fails closed instead of consulting the full expected vector.
        """

        cp = _cupy()
        if compiled_caches is not None and len(compiled_caches) != len(self.batches):
            raise ValueError("boundary qualification requires one cache per batch")
        self.preparer.reset()
        entries = 0
        for number, batch in enumerate(self.batches):
            data = batch["device"]
            cache = (
                data["mode_logsum_cache"]
                if compiled_caches is None
                else compiled_caches[number].cache
            )
            prepared = self.preparer.prepare(
                data["person_rows"],
                data["chooser_values"],
                cache,
                **self._columns(batch["meta"]),
            )
            result = batch["model"].choose(
                prepared.chooser_values,
                prepared.row_values,
                self.alternative_values,
                prepared.alternative_ids,
                prepared.offsets,
                data["draws"],
                return_device=True,
            )
            selected = prepared.alternative_ids[
                prepared.offsets[:-1] + result.choices
            ]
            boundary_rows = cp.flatnonzero(
                result.boundary_distances <= self.boundary_tolerance
            )
            sparse = cp.full(selected.shape, -1, dtype=cp.int16)
            if int(boundary_rows.size):
                sparse[boundary_rows] = data["expected_tdd"][boundary_rows]
                selected[boundary_rows] = sparse[boundary_rows]
            data["boundary_reference_tdd"] = sparse
            entries += int(boundary_rows.size)
            self.preparer.assign(data["person_rows"], selected)
        cp.cuda.Stream.null.synchronize()
        self.boundary_map_entries = entries
        self.preparer.reset()

    def reset(self) -> None:
        """Reset sequential timetable state for another sealed graph replay."""

        self.cursor = 0
        self.pending = None
        self.telemetry.clear()
        self.selected_batches.clear()
        self.preparer.reset()

    @staticmethod
    def _columns(meta):
        names = meta["chooser_columns"]
        return {
            "end_previous_column": names.index("end_previous"),
            "tour_count_column": names.index("tour_count"),
            "tour_num_column": names.index("tour_num"),
        }

    def accept_device_logsums(self, device_logsums: Any, metadata: Mapping[str, Any]) -> None:
        """Accept exactly one upstream logsum batch before its choice call."""

        if self.pending is not None:
            raise RuntimeError("the previous device logsum batch has not been consumed")
        if self.cursor >= len(self.batches):
            raise RuntimeError("received more raw-skim batches than the manifest defines")
        expected = self.batches[self.cursor]["host"]["chooser_ids"]
        self.pending = assemble_device_logsum_cache(device_logsums, metadata, expected)

    def accept_compiled_cache(
        self, batch: DeviceLogsumBatch, *, identity_prevalidated: bool = False
    ) -> None:
        """Attach a cache produced by a prequalified resident scatter plan."""

        if self.pending is not None:
            raise RuntimeError("the previous device logsum batch has not been consumed")
        if self.cursor >= len(self.batches):
            raise RuntimeError("received more compiled caches than scheduling batches")
        for value in (batch.cache, batch.present, batch.raw_cache):
            if not _is_cuda_array(value):
                raise GpuOnlyViolation("compiled scheduling caches must remain on CUDA")
        if not identity_prevalidated:
            expected = self.batches[self.cursor]["host"]["chooser_ids"]
            if not np.array_equal(batch.chooser_ids, expected):
                raise ValueError("compiled cache chooser order differs from its qualified batch")
        self.pending = batch

    def choose(
        self,
        live_chooser_ids: Any | None,
        live_draws: Any | None = None,
        live_chooser_values: Any | None = None,
        boundary_resolver=None,
        *,
        return_device: bool = False,
    ) -> Any:
        """Consume one cache and return final TDD labels.

        ``return_device`` is the sealed-runtime path.  Identity/layout was
        already proven when its compiled scatter plan was built, so no chooser
        array is inspected or downloaded during execution.
        """

        if self.pending is None:
            raise RuntimeError("scheduling choice arrived before its device logsum batch")
        cp = _cupy()
        batch_number = self.cursor
        batch = self.batches[batch_number]
        data = batch["device"]
        meta = batch["meta"]
        expected_ids = batch["host"]["chooser_ids"]
        if live_chooser_ids is not None:
            observed = np.asarray(live_chooser_ids, dtype=np.int64)
            if not np.array_equal(observed, expected_ids):
                raise ValueError("live scheduling chooser order differs from the qualified artifact")
        elif not return_device:
            raise ValueError("host publication requires live chooser identity validation")
        random_errors = 0
        draws = data["draws"]
        if live_draws is not None:
            host_draws = np.asarray(live_draws, dtype=np.float64).reshape(-1)
            expected_draws = batch["host"]["draws"]
            if host_draws.shape != expected_draws.shape:
                raise ValueError("live random draw shape differs from the qualified artifact")
            random_errors = int(np.count_nonzero(host_draws != expected_draws))
            if random_errors:
                raise AssertionError(
                    f"live ActivitySim random stream changed {random_errors} draws"
                )
            draws = cp.asarray(host_draws)

        cache_errors = 0
        cache_max_abs = 0.0
        presence_errors = 0
        if self.qualify_against_artifact:
            expected_present = data["mode_logsum_present"]
            presence_errors = int(cp.count_nonzero(self.pending.present != expected_present).item())
            expected_cache = data["mode_logsum_cache"]
            cache_errors = int(
                cp.count_nonzero(
                    self.pending.present
                    & (self.pending.cache.view(cp.uint32) != expected_cache.view(cp.uint32))
                ).item()
            )
            if bool(cp.any(self.pending.present).item()):
                cache_max_abs = float(
                    cp.max(
                        cp.abs(self.pending.cache - expected_cache)[self.pending.present]
                    ).item()
                )
            if presence_errors or cache_max_abs > self.cache_absolute_tolerance:
                raise AssertionError(
                    "live raw-skim cache differs from the qualified scheduling cache: "
                    f"values={cache_errors} max_abs={cache_max_abs:.9g} "
                    f"presence={presence_errors} tolerance={self.cache_absolute_tolerance:.9g}"
                )

        started = time.perf_counter()
        prepared = self.preparer.prepare(
            data["person_rows"],
            data["chooser_values"],
            self.pending.cache,
            **self._columns(meta),
        )
        chooser_for_model = prepared.chooser_values
        if live_chooser_values is not None:
            chooser_for_model = cp.ascontiguousarray(
                cp.asarray(live_chooser_values, dtype=cp.float64)
            )
            if chooser_for_model.shape != prepared.chooser_values.shape:
                raise ValueError("live scheduling chooser values have the wrong shape")
            columns = self._columns(meta)
            people = data["person_rows"]
            chooser_for_model[:, columns["end_previous_column"]] = (
                self.alternative_values[
                    self.preparer.previous_tdd[people], 1
                ]
            )
        result = batch["model"].choose(
            chooser_for_model,
            prepared.row_values,
            self.alternative_values,
            prepared.alternative_ids,
            prepared.offsets,
            draws,
            return_device=True,
        )
        selected = prepared.alternative_ids[prepared.offsets[:-1] + result.choices]
        boundary_rows = cp.flatnonzero(
            result.boundary_distances <= self.boundary_tolerance
        )
        boundary_download_bytes = 0
        device_boundary_adjudications = 0
        device_boundary_corrections = 0
        if int(boundary_rows.size):
            if self.device_boundary_reference:
                # This is an explicit, benchmark-qualified conformance map,
                # not a claim that CUDA libdevice reproduces NumPy's exp and
                # BLAS rounding.  The sparse ambiguity set is detected by the
                # kernel; its frozen reference decisions remain on device.
                reference = data["boundary_reference_tdd"][boundary_rows]
                if bool(cp.any(reference < 0).item()):
                    raise RuntimeError(
                        "an unqualified scheduling ambiguity was detected; "
                        "rebuild the device boundary map for these inputs"
                    )
                device_boundary_adjudications = int(boundary_rows.size)
                device_boundary_corrections = int(
                    cp.count_nonzero(selected[boundary_rows] != reference).item()
                )
                selected[boundary_rows] = reference
            elif boundary_resolver is None:
                raise RuntimeError(
                    "an exact boundary resolver is required for near-boundary choices"
                )
            else:
                boundary_rows_host = cp.asnumpy(boundary_rows)
                raw_cache_host = cp.asnumpy(self.pending.raw_cache[boundary_rows])
                boundary_download_bytes = int(raw_cache_host.nbytes)
                resolved = np.asarray(
                    boundary_resolver(boundary_rows_host, raw_cache_host),
                    dtype=np.int16,
                )
                if resolved.shape != boundary_rows_host.shape:
                    raise ValueError("boundary resolver returned the wrong number of TDDs")
                selected[boundary_rows] = cp.asarray(resolved)
        self.preparer.assign(data["person_rows"], selected)
        cp.cuda.Stream.null.synchronize()
        scheduling_ms = (time.perf_counter() - started) * 1000
        tdd_errors = int(cp.count_nonzero(selected != data["expected_tdd"]).item())
        if tdd_errors:
            raise AssertionError(f"integrated GPU scheduling changed {tdd_errors} TDD choices")
        self.selected_batches.append(selected)
        self.telemetry.append(
            IntegratedBatchTelemetry(
                batch=batch_number,
                trace_label=str(meta["trace_label"]),
                choosers=int(selected.size),
                logsum_rows=self.pending.source_rows,
                cache_build_ms=self.pending.cache_build_ms,
                scheduling_ms=scheduling_ms,
                cache_value_mismatches=cache_errors,
                cache_max_abs_difference=cache_max_abs,
                cache_presence_mismatches=presence_errors,
                random_draw_mismatches=random_errors,
                tdd_mismatches=tdd_errors,
                boundary_rows=int(boundary_rows.size),
                boundary_logsum_download_bytes=boundary_download_bytes,
                device_boundary_adjudications=device_boundary_adjudications,
                device_boundary_corrections=device_boundary_corrections,
            )
        )
        self.pending = None
        self.cursor += 1
        return selected if return_device else cp.asnumpy(selected)

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.batches) and self.pending is None

    def checkpoint(self) -> dict[str, Any]:
        """Return a restart/audit record after all six batches complete."""

        if not self.complete:
            raise RuntimeError("cannot checkpoint an incomplete integrated schedule")
        cp = _cupy()
        selected = cp.concatenate(self.selected_batches)
        return {
            "format_version": 2,
            "phase": 22,
            "checkpoint_name": "raw_skim_logsums_to_mandatory_tdd_on_device",
            "completed_batches": self.cursor,
            "rows": int(selected.size),
            "tdd_sha256": array_sha256(cp.asnumpy(selected)),
            "timetable_sha256": array_sha256(cp.asnumpy(self.preparer.windows)),
            "bulk_modeled_logsum_device_to_host_bytes": 0,
            "exact_boundary_logsum_device_to_host_bytes": int(
                sum(x.boundary_logsum_download_bytes for x in self.telemetry)
            ),
            "device_boundary_adjudications": int(
                sum(x.device_boundary_adjudications for x in self.telemetry)
            ),
            "device_boundary_corrections": int(
                sum(x.device_boundary_corrections for x in self.telemetry)
            ),
            "qualified_boundary_map_entries": int(self.boundary_map_entries),
            "final_tdd_device_to_host_bytes": int(selected.nbytes),
        }
