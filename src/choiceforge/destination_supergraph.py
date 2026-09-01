"""Phase 49 device handoff between destination mode logsums and final choice.

The public ActivitySim pipeline represents the mode-choice logsum as a pandas
column between two model sub-stages.  ChoiceForge already produces that vector
on CUDA and the Phase 47 final utility immediately consumes it on CUDA.  This
module replaces that redundant device-to-host-to-device round trip with a
small, fail-closed queue whose row identity is checked at the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class ResidentLogsumPacket:
    source_values: object
    final_values: object
    row_ids: np.ndarray
    producer_trace: str
    host_materialized: bool
    source_bytes: int
    created_at: float


class DestinationSupergraphBridge:
    """One-producer/one-consumer resident logsum handoff.

    The bridge intentionally has no permissive lookup or fallback.  A changed
    call order, row count, or repeated-index order is an ABI change and raises
    before a choice can be made.
    """

    version = 1

    def __init__(self, cp):
        self.cp = cp
        self._queue: list[ResidentLogsumPacket] = []
        self._active: ResidentLogsumPacket | None = None
        self._selected: list[tuple[np.ndarray, np.ndarray]] = []
        self._events: list[dict] = []

    def publish(self, values, metadata, *, host_materialized: bool) -> None:
        started = time.perf_counter()
        row_ids = np.ascontiguousarray(metadata["chooser_ids"], dtype=np.int64)
        if getattr(values, "ndim", None) != 1 or int(values.shape[0]) != len(row_ids):
            raise ValueError("Phase 49 logsum producer shape differs from row identity")
        # Phase 47 consumes float32.  Perform the same narrowing once on-device
        # that its former np.asarray(..., dtype=float32) + upload performed.
        resident = values.astype(self.cp.float32, copy=True)
        self._queue.append(
            ResidentLogsumPacket(
                source_values=values,
                final_values=resident,
                row_ids=row_ids.copy(),
                producer_trace=str(metadata.get("trace_label", "unknown")),
                host_materialized=bool(host_materialized),
                source_bytes=int(values.nbytes),
                created_at=started,
            )
        )

    def consume(self, alternatives, *, consumer_trace: str):
        if not self._queue:
            raise RuntimeError("Phase 49 final choice has no resident logsum producer")
        packet = self._queue.pop(0)
        actual_ids = np.asarray(alternatives.index, dtype=np.int64)
        if len(actual_ids) != len(packet.row_ids) or not np.array_equal(
            actual_ids, packet.row_ids
        ):
            raise ValueError(
                "Phase 49 producer/final-choice alternative order differs"
            )
        now = time.perf_counter()
        self._events.append(
            {
                "producer_trace": packet.producer_trace,
                "consumer_trace": str(consumer_trace),
                "rows": len(packet.row_ids),
                "source_float64_bytes": packet.source_bytes,
                "resident_float32_bytes": int(packet.final_values.nbytes),
                "host_materialized_for_published_output": packet.host_materialized,
                "device_to_host_bytes_avoided": (
                    0 if packet.host_materialized else packet.source_bytes
                ),
                "host_to_device_bytes_avoided": int(packet.final_values.nbytes),
                "handoff_seconds": now - packet.created_at,
                "row_identity_exact": True,
                "contract_version": self.version,
            }
        )
        self._active = packet
        return packet.final_values

    def capture_selected(self, chooser_ids, selected_rows, *, component: str) -> None:
        """Materialize only published location logsums after device choice."""
        packet = self._active
        self._active = None
        if packet is None:
            raise RuntimeError("Phase 49 selected-logsum capture has no active packet")
        if component not in {"school_location", "workplace_location"}:
            return
        selected_rows = np.asarray(selected_rows, dtype=np.int64)
        chooser_ids = np.asarray(chooser_ids, dtype=np.int64)
        if len(selected_rows) != len(chooser_ids):
            raise ValueError("Phase 49 selected-logsum row count differs from choosers")
        selected = self.cp.asnumpy(
            packet.source_values[self.cp.asarray(selected_rows)]
        )
        self._selected.append((chooser_ids.copy(), np.asarray(selected).copy()))
        self._events[-1]["selected_output_rows"] = len(chooser_ids)
        self._events[-1]["selected_output_device_to_host_bytes"] = int(selected.nbytes)

    def consume_selected(self, output_index) -> np.ndarray:
        if not self._selected:
            raise RuntimeError("Phase 49 location output has no selected logsums")
        ids = np.concatenate([item[0] for item in self._selected])
        values = np.concatenate([item[1] for item in self._selected])
        self._selected.clear()
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Phase 49 selected location chooser identity is not unique")
        order = {int(value): position for position, value in enumerate(ids)}
        try:
            positions = np.asarray([order[int(value)] for value in output_index])
        except KeyError as exc:
            raise ValueError("Phase 49 selected output identity differs") from exc
        if len(positions) != len(ids):
            raise ValueError("Phase 49 selected output chooser count differs")
        return values[positions]

    def assert_empty(self) -> None:
        if self._queue or self._active is not None or self._selected:
            raise RuntimeError(
                "Phase 49 ended with an unconsumed resident logsum or selected-output packet"
            )

    def summary(self) -> dict:
        events = list(self._events)
        return {
            "contract_version": self.version,
            "calls": len(events),
            "rows": int(sum(item["rows"] for item in events)),
            "source_device_to_host_bytes_eliminated": int(
                sum(item["device_to_host_bytes_avoided"] for item in events)
            ),
            "selected_output_device_to_host_bytes": int(sum(
                item.get("selected_output_device_to_host_bytes", 0)
                for item in events
            )),
            "device_to_host_bytes_avoided": int(sum(
                item["device_to_host_bytes_avoided"]
                - item.get("selected_output_device_to_host_bytes", 0)
                for item in events
            )),
            "host_to_device_bytes_avoided": int(
                sum(item["host_to_device_bytes_avoided"] for item in events)
            ),
            "round_trip_bytes_avoided": int(
                sum(
                    item["device_to_host_bytes_avoided"]
                    - item.get("selected_output_device_to_host_bytes", 0)
                    + item["host_to_device_bytes_avoided"]
                    for item in events
                )
            ),
            "all_row_identities_exact": all(
                item["row_identity_exact"] for item in events
            ),
            "pending_packets": len(self._queue) + int(self._active is not None),
            "pending_selected_outputs": len(self._selected),
            "events": events,
        }
