"""Verified file-backed skim image for the Phase 56 model-wide runtime.

ActivitySim normally expands the public OMX skims into a 6.45 GB ephemeral
shared-memory image in every fresh process.  Sharrow already supports an
equivalent NumPy memmap backing store.  This module adds the missing release
contract: source hashes, metadata and artifact digests, fail-closed runtime
validation, and an optional out-of-band full-image qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


CONTRACT = "choiceforge-phase56-verified-sharrow-memmap-v1"


def install_sharrow_memmap_compatibility_bridge():
    """Install a narrow bridge for Sharrow's NumPy-memmap truth-value bug.

    Sharrow 2.13 passes the newly created ``np.memmap`` as ``own_data`` and
    then evaluates that array as a boolean.  NumPy correctly rejects the
    ambiguous truth test.  Treating the argument as the documented ownership
    flag makes Sharrow reopen the exact same file read-only and leaves its
    layout/reconstruction code unchanged.

    Returns the accessor class and original descriptor so the caller can
    restore process-global state after ActivitySim exits.
    """
    import numpy as np
    from sharrow.shared_memory import SharedMemDatasetAccessor

    original_descriptor = SharedMemDatasetAccessor.__dict__["from_shared_memory"]
    original_function = original_descriptor.__func__

    def compatible(cls, key, own_data=False, mode="r+"):
        if isinstance(own_data, np.memmap):
            own_data = True
        return original_function(cls, key, own_data=own_data, mode=mode)

    SharedMemDatasetAccessor.from_shared_memory = classmethod(compatible)
    return SharedMemDatasetAccessor, original_descriptor


def restore_sharrow_memmap_compatibility_bridge(accessor_class, descriptor):
    accessor_class.from_shared_memory = descriptor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _source_paths(project: Path, data: Path) -> tuple[Path, ...]:
    return (
        data / "skims.omx",
        data / "land_use.csv",
        project / "configs" / "network_los.yaml",
    )


def _source_records(project: Path, data: Path) -> list[dict]:
    records = []
    for path in _source_paths(project, data):
        if not path.is_file():
            raise FileNotFoundError(f"Phase 56 source is absent: {path}")
        stat = path.stat()
        records.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(path),
            }
        )
    return records


def manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".choiceforge.json")


def create_phase56_manifest(cache_path: Path, project: Path, data: Path) -> dict:
    """Seal a Sharrow-created memmap after an explicit build run."""
    cache_path = cache_path.resolve()
    metadata_path = Path(str(cache_path) + ".meta.pkl")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Phase 56 memmap or Sharrow metadata is absent")
    started = time.perf_counter()
    cache_stat = cache_path.stat()
    payload = {
        "contract": CONTRACT,
        "cache": {
            "bytes": cache_stat.st_size,
            "mtime_ns": cache_stat.st_mtime_ns,
            "sha256": _sha256(cache_path),
        },
        "metadata": {
            "bytes": metadata_path.stat().st_size,
            "sha256": _sha256(metadata_path),
        },
        "sources": _source_records(project.resolve(), data.resolve()),
    }
    payload["artifact_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    payload["qualification_seconds"] = time.perf_counter() - started
    target = manifest_path(cache_path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return payload


def validate_phase56_cache(
    cache_path: Path,
    project: Path,
    data: Path,
    *,
    verify_cache_content: bool = False,
) -> dict:
    """Validate the sealed cache, using full source hashes in timed runs.

    The 6.45 GB cache hash is checked during artifact construction and formal
    qualification.  Timed production startup checks its exact size and mtime,
    the small Sharrow layout metadata hash, and full hashes of all source files.
    Set ``verify_cache_content`` for the out-of-band full-image proof.
    """
    cache_path = cache_path.resolve()
    target = manifest_path(cache_path)
    metadata_path = Path(str(cache_path) + ".meta.pkl")
    if not target.is_file():
        raise FileNotFoundError(f"Phase 56 cache manifest is absent: {target}")
    started = time.perf_counter()
    document = json.loads(target.read_text(encoding="utf-8"))
    stored_digest = document.get("artifact_sha256")
    qualification_seconds = document.get("qualification_seconds")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    payload.pop("qualification_seconds", None)
    checks = {
        "contract": document.get("contract") == CONTRACT,
        "artifact_digest": stored_digest
        == hashlib.sha256(_canonical(payload)).hexdigest(),
        "cache_present": cache_path.is_file(),
        "metadata_present": metadata_path.is_file(),
    }
    if checks["cache_present"]:
        stat = cache_path.stat()
        checks["cache_size"] = stat.st_size == document["cache"]["bytes"]
        checks["cache_mtime"] = stat.st_mtime_ns == document["cache"]["mtime_ns"]
    if checks["metadata_present"]:
        checks["metadata_size"] = (
            metadata_path.stat().st_size == document["metadata"]["bytes"]
        )
        checks["metadata_digest"] = (
            _sha256(metadata_path) == document["metadata"]["sha256"]
        )
    current_sources = _source_records(project.resolve(), data.resolve())
    checks["source_digests"] = current_sources == document.get("sources")
    if verify_cache_content and checks["cache_present"]:
        checks["cache_digest"] = _sha256(cache_path) == document["cache"]["sha256"]
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Phase 56 cache validation failed: {failed}")
    return {
        "contract": CONTRACT,
        "cache_path": str(cache_path),
        "cache_bytes": document["cache"]["bytes"],
        "cache_sha256": document["cache"]["sha256"],
        "metadata_sha256": document["metadata"]["sha256"],
        "artifact_sha256": stored_digest,
        "source_sha256": {
            item["name"]: item["sha256"] for item in current_sources
        },
        "full_cache_digest_verified": bool(verify_cache_content),
        "artifact_qualification_seconds": qualification_seconds,
        "runtime_validation_seconds": time.perf_counter() - started,
        "all_checks_pass": True,
    }
