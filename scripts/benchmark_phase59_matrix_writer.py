"""Balanced writer-only comparison using actual published public OMX values."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import time
import h5py
import numpy as np
import pandas as pd
from activitysim.abm.models.trip_matrices import write_matrices as upstream
from choiceforge.phase59_sparse_matrices import write_matrices as sparse
from verify_phase59_matrices import inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    if args.output_root.exists() or args.report.exists():
        raise FileExistsError("Writer benchmark requires fresh output paths")
    args.output_root.mkdir(parents=True)
    with h5py.File(args.reference, "r") as file:
        names = sorted(file["data"])
        values = {name:file["data"][name][:] for name in names}
        zone_name = next(iter(file["lookup"]))
        zones = pd.Index(file["lookup"][zone_name][:], name=zone_name)
    nonzero = np.zeros((len(zones),len(zones)), bool)
    for value in values.values():
        nonzero |= (value != 0) | np.signbit(value)
    orig, dest = np.nonzero(nonzero)
    aggregate = pd.DataFrame({name:value[orig, dest] for name,value in values.items()})
    del values
    settings = SimpleNamespace(HH_EXPANSION_WEIGHT_COL=None,
        MATRICES=[SimpleNamespace(file_name=args.reference.name,
                                 tables=[SimpleNamespace(name=name, data_field=name) for name in names])])
    reference_digest = inventory(args.reference)
    runs = []
    for trial in range(args.repetitions):
        order = ("upstream", *(f"sparse{size}" for size in args.chunks))
        if trial % 2:
            order = order[::-1]
        for position, mode in enumerate(order):
            directory = args.output_root / f"{trial+1}-{mode}"
            directory.mkdir()
            state = SimpleNamespace(get_output_file_path=lambda name:directory/name)
            frame = aggregate.copy()
            started = time.perf_counter()
            event = upstream(state, frame, zones, orig, dest, settings) if mode == "upstream" else sparse(
                state, frame, zones, orig, dest, settings, chunk_size=int(mode[6:]), workers=4)
            elapsed = time.perf_counter()-started
            if inventory(directory/args.reference.name) != reference_digest:
                raise ValueError(f"Writer {mode} changed logical matrix data")
            runs.append({"trial":trial+1, "position":position, "mode":mode,
                         "seconds":elapsed, "exact":True, "event":event,
                         "file_bytes":(directory/args.reference.name).stat().st_size})
    result = {"scope":"writer-only: identical actual public matrix values; annotation and aggregation excluded",
        "input_sha256":hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "runs":runs, "medians_seconds":{mode:statistics.median(r["seconds"] for r in runs if r["mode"] == mode)
                                        for mode in {r["mode"] for r in runs}}}
    args.report.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["medians_seconds"], indent=2))


if __name__ == "__main__":
    main()
