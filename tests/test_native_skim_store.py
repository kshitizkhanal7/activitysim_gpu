from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest


cp = pytest.importorskip("cupy")

from choiceforge.native_skim_store import (
    MANIFEST_NAME,
    PAYLOAD_NAME,
    NativeSkimStore,
    build_native_skim_store,
)
from choiceforge.resident_skim_cache import MTC_PERIODS


def _document(key="TIME"):
    return {
        "terms": [
            {
                "tree": {
                    "op": "skim",
                    "direction": direction,
                    "key": {"op": "const", "value": key if rank == 3 else "DIST"},
                }
            }
            for direction, rank in (
                ("od_skims", 2),
                ("od_skims_reverse", 2),
                ("odt_skims", 3),
                ("dot_skims", 3),
                ("odr_skims", 3),
                ("dor_skims", 3),
            )
        ]
    }


def _source(tmp_path):
    omx = tmp_path / "tiny.omx"
    land = tmp_path / "land_use.csv"
    with h5py.File(omx, "w") as output:
        data = output.create_group("data")
        base = np.arange(36, dtype=np.float64).reshape(6, 6)
        data["DIST"] = base + 0.25
        for number, period in enumerate(MTC_PERIODS):
            data[f"TIME__{period}"] = base + number * 100 + 0.5
    pd.DataFrame({"TAZ": [1, 3, 5]}).to_csv(land, index=False)
    return omx, land


def test_native_skim_store_round_trip_deduplicates_and_verifies(tmp_path):
    omx, land = _source(tmp_path)
    store_path = tmp_path / "store"
    manifest = build_native_skim_store(omx, land, _document(), store_path)
    assert manifest["logical_bindings"] == 6
    assert len(manifest["entries"]) == 2
    expected_bytes = 3 * 3 * 4 * 6
    assert manifest["payload_bytes"] == expected_bytes

    store = NativeSkimStore.load(
        store_path, _document(), [1, 3, 5], budget_bytes=expected_bytes
    )
    static, zones, periods, rank = store.cube(("skim", "od_skims", "DIST"))
    reverse, *_ = store.cube(("skim", "od_skims_reverse", "DIST"))
    timed, _, timed_periods, timed_rank = store.cube(
        ("skim", "dot_skims", "TIME")
    )
    assert static.data.ptr == reverse.data.ptr
    assert (zones, periods, rank) == (3, 1, 2)
    assert (timed_periods, timed_rank) == (5, 3)
    np.testing.assert_array_equal(cp.asnumpy(static), (np.arange(36).reshape(6, 6) + 0.25)[np.ix_([0, 2, 4], [0, 2, 4])].astype(np.float32))
    assert store.telemetry.verified_payload_bytes == expected_bytes
    assert int(timed.nbytes + static.nbytes) == expected_bytes


def test_native_skim_store_fails_closed_on_payload_corruption(tmp_path):
    omx, land = _source(tmp_path)
    store_path = tmp_path / "store"
    manifest = build_native_skim_store(omx, land, _document(), store_path)
    with (store_path / PAYLOAD_NAME).open("r+b") as stream:
        stream.seek(manifest["entries"][0]["offset"] + 3)
        byte = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="corrupt"):
        NativeSkimStore.load(
            store_path, _document(), [1, 3, 5], budget_bytes=10_000
        )


def test_native_skim_store_fails_closed_on_manifest_contract_and_zones(tmp_path):
    omx, land = _source(tmp_path)
    store_path = tmp_path / "store"
    build_native_skim_store(omx, land, _document(), store_path)
    with pytest.raises(ValueError, match="skim contract"):
        NativeSkimStore.load(
            store_path, _document("OTHER"), [1, 3, 5], budget_bytes=10_000
        )
    with pytest.raises(ValueError, match="zone identity"):
        NativeSkimStore.load(
            store_path, _document(), [1, 2, 3], budget_bytes=10_000
        )

    manifest_path = store_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_bytes"] += 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        NativeSkimStore.load(
            store_path, _document(), [1, 3, 5], budget_bytes=10_000
        )


def test_native_skim_store_refuses_overwrite_and_budget_overrun(tmp_path):
    omx, land = _source(tmp_path)
    store_path = tmp_path / "store"
    manifest = build_native_skim_store(omx, land, _document(), store_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_native_skim_store(omx, land, _document(), store_path)
    with pytest.raises(MemoryError, match="exceeds budget"):
        NativeSkimStore.load(
            store_path,
            _document(),
            [1, 3, 5],
            budget_bytes=manifest["payload_bytes"] - 1,
        )
