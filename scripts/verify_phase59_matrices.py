"""Exact logical OMX contents, independent of HDF5 timestamps/compression."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def inventory(path):
    result = {}
    with h5py.File(path, "r") as file:
        def visit(name, item):
            if not isinstance(item, h5py.Dataset):
                return
            digest = hashlib.sha256()
            if item.ndim == 0:
                digest.update(np.asarray(item[()]).tobytes())
            else:
                for i in range(0, item.shape[0], 64):
                    digest.update(np.ascontiguousarray(item[i:i+64]).tobytes())
            result[name] = {"dtype":str(item.dtype), "shape":list(item.shape), "sha256":digest.hexdigest()}
        file.visititems(visit)
    if not result or not any(k.startswith("data/") for k in result):
        raise ValueError(f"No matrix datasets: {path}")
    if not any(k.startswith("lookup/") for k in result):
        raise ValueError(f"No zone mapping: {path}")
    return result


def verify(reference, candidate):
    names = sorted(p.name for p in reference.glob("*.omx"))
    if not names or names != sorted(p.name for p in candidate.glob("*.omx")):
        raise ValueError("Matrix file inventories differ or are empty")
    files = {}
    for name in names:
        left, right = inventory(reference / name), inventory(candidate / name)
        if left != right:
            differing = [k for k in left.keys() | right.keys() if left.get(k) != right.get(k)]
            raise ValueError(f"{name} differs in {differing}")
        files[name] = right
    return {"exact":True, "files":files, "policy":"all logical dataset bytes, shapes, dtypes and zone mappings exact; container metadata excluded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reference", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Verified {len(result['files'])} OMX files exactly")
