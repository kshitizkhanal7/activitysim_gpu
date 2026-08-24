from __future__ import annotations

import json

import numpy as np
import pytest

from choiceforge.cuda_backend import _cupy, cuda_available
from choiceforge.device_resident_runtime import DeviceResidentRuntime
from choiceforge.gpu_native import DeviceTable, GpuOnlyViolation


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA unavailable")


def test_resident_graph_rejects_host_results_and_commits_atomically():
    cp = _cupy()
    runtime = DeviceResidentRuntime()
    runtime.ingress_table("people", {"id": np.arange(5, dtype=np.int64)})
    runtime.seal_ingress()

    with pytest.raises(GpuOnlyViolation, match="not a CUDA array"):
        runtime.run_stage(
            "bad_host_stage",
            reads=("people",),
            writes=("choices",),
            operation=lambda _: {"choices": {"choice": np.arange(5)}},
        )
    assert "choices" not in runtime.tables

    runtime.run_stage(
        "device_stage",
        reads=("people",),
        writes=("choices",),
        operation=lambda tables: {
            "choices": {"choice": tables["people"].columns["id"].astype(cp.int16)}
        },
    )
    assert runtime.versions["choices"] == 1
    with pytest.raises(ValueError, match="overwrite"):
        runtime.run_stage(
            "duplicate",
            reads=("people",),
            writes=("choices",),
            operation=lambda _: {"choices": DeviceTable({"choice": cp.zeros(5)})},
        )
    runtime.release_tables("choices")
    assert "choices" not in runtime.tables


def test_resident_runtime_fails_closed_after_ingress():
    runtime = DeviceResidentRuntime()
    runtime.ingress_table("people", {"id": np.arange(3)})
    runtime.seal_ingress()
    with pytest.raises(GpuOnlyViolation, match="sealed"):
        runtime.ingress_table("late", {"id": np.arange(2, dtype=np.int64)})
    assert runtime.telemetry.forbidden_postseal_host_bytes == 16
    with pytest.raises(GpuOnlyViolation, match="CPU fallback"):
        runtime.cpu_fallback("legacy_expression")
    with pytest.raises(GpuOnlyViolation, match="contract failed"):
        runtime.assert_resident_contract()


def test_checkpoint_restore_is_self_contained_and_hash_verified(tmp_path):
    cp = _cupy()
    runtime = DeviceResidentRuntime()
    runtime.ingress_table("seed", {"id": np.arange(7, dtype=np.int64)})
    runtime.seal_ingress()
    runtime.random_ledger.reserve("persons", "frequency", 3)
    runtime.run_stage(
        "double",
        reads=("seed",),
        writes=("result",),
        operation=lambda tables: {
            "result": {"value": tables["seed"].columns["id"] * cp.int64(2)}
        },
    )
    manifest = runtime.checkpoint(
        tmp_path, tables=("result",), metadata={"purpose": "unit-test"}
    )
    assert manifest["completed_stages"] == ["double"]
    assert manifest["metadata"] == {"purpose": "unit-test"}

    restored = DeviceResidentRuntime.restore(tmp_path)
    restored.run_stage(
        "continue_after_restart",
        reads=("result",),
        writes=("continued",),
        operation=lambda tables: {
            "continued": {"value": tables["result"].columns["value"] + cp.int64(1)}
        },
    )
    output = restored.publish({"result": ("value",), "continued": ("value",)})
    np.testing.assert_array_equal(output["result"]["value"], np.arange(7) * 2)
    np.testing.assert_array_equal(output["continued"]["value"], np.arange(7) * 2 + 1)
    assert restored.random_ledger.snapshot() == {"persons:frequency": 3}
    assert restored.versions["result"] == 1

    manifest_path = tmp_path / "manifest.json"
    damaged = json.loads(manifest_path.read_text())
    damaged["state_archive_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="archive hash"):
        DeviceResidentRuntime.restore(tmp_path)


def test_publication_is_explicit_and_stage_versions_advance():
    cp = _cupy()
    runtime = DeviceResidentRuntime()
    runtime.ingress_table("state", {"value": np.arange(4, dtype=np.int32)})
    runtime.seal_ingress()

    for iteration in range(2):
        runtime.run_stage(
            f"scenario_{iteration}",
            reads=("state",),
            writes=("answer",),
            replace=iteration > 0,
            operation=lambda tables: {
                "answer": {"value": tables["state"].columns["value"] + cp.int32(1)}
            },
        )
    result = runtime.publish({"answer": ("value",)})
    np.testing.assert_array_equal(result["answer"]["value"], np.arange(4) + 1)
    assert runtime.versions["answer"] == 2
    assert runtime.telemetry.publication_calls == 1
    assert runtime.telemetry.publication_bytes == 16
    assert len(runtime.telemetry_dict()["stages"]) == 2
