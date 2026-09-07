import json
from pathlib import Path

import pytest

from choiceforge.phase56_skim_cache import (
    CONTRACT,
    create_phase56_manifest,
    manifest_path,
    install_sharrow_memmap_compatibility_bridge,
    restore_sharrow_memmap_compatibility_bridge,
    validate_phase56_cache,
)


def _fixture(tmp_path: Path):
    project = tmp_path / "project"
    data = tmp_path / "data"
    (project / "configs").mkdir(parents=True)
    data.mkdir()
    (data / "skims.omx").write_bytes(b"skim-source")
    (data / "land_use.csv").write_text("zone,value\n1,2\n", encoding="utf-8")
    (project / "configs" / "network_los.yaml").write_text(
        "skim_time_periods: {}\n", encoding="utf-8"
    )
    cache = tmp_path / "skim.mmap"
    cache.write_bytes(b"verified-cache-image")
    Path(str(cache) + ".meta.pkl").write_bytes(b"layout-metadata")
    return project, data, cache


def test_phase56_manifest_round_trip_and_full_digest(tmp_path):
    project, data, cache = _fixture(tmp_path)
    sealed = create_phase56_manifest(cache, project, data)
    proof = validate_phase56_cache(
        cache, project, data, verify_cache_content=True
    )

    assert sealed["contract"] == CONTRACT
    assert proof["all_checks_pass"] is True
    assert proof["full_cache_digest_verified"] is True
    assert set(proof["source_sha256"]) == {
        "skims.omx",
        "land_use.csv",
        "network_los.yaml",
    }


@pytest.mark.parametrize("target", ["source", "metadata", "cache"])
def test_phase56_validation_fails_closed_after_mutation(tmp_path, target):
    project, data, cache = _fixture(tmp_path)
    create_phase56_manifest(cache, project, data)
    if target == "source":
        (data / "land_use.csv").write_text("changed\n", encoding="utf-8")
    elif target == "metadata":
        Path(str(cache) + ".meta.pkl").write_bytes(b"changed")
    else:
        cache.write_bytes(b"changed")

    with pytest.raises(ValueError, match="cache validation failed"):
        validate_phase56_cache(cache, project, data, verify_cache_content=True)


def test_phase56_validation_rejects_manifest_digest_tampering(tmp_path):
    project, data, cache = _fixture(tmp_path)
    create_phase56_manifest(cache, project, data)
    path = manifest_path(cache)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cache"]["bytes"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_digest"):
        validate_phase56_cache(cache, project, data)


def test_phase56_sharrow_bridge_converts_only_memmap_ownership(tmp_path, monkeypatch):
    import numpy as np
    from sharrow.shared_memory import SharedMemDatasetAccessor

    observed = []
    original_descriptor = SharedMemDatasetAccessor.__dict__["from_shared_memory"]

    def recorder(cls, key, own_data=False, mode="r+"):
        observed.append((key, own_data, mode))
        return "attached"

    monkeypatch.setattr(
        SharedMemDatasetAccessor, "from_shared_memory", classmethod(recorder)
    )
    accessor, descriptor = install_sharrow_memmap_compatibility_bridge()
    backing = np.memmap(tmp_path / "tiny.mmap", mode="w+", shape=4)
    try:
        assert accessor.from_shared_memory("memmap:tiny", backing, "r") == "attached"
        assert observed == [("memmap:tiny", True, "r")]
    finally:
        restore_sharrow_memmap_compatibility_bridge(accessor, descriptor)
        del backing
        SharedMemDatasetAccessor.from_shared_memory = original_descriptor
