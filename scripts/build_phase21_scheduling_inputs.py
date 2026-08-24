"""Build the compact Phase 21 scheduling-preparation boundary.

The source is the exact Phase 20 ActivitySim capture.  This transformation is
lossless and aggressively checked: 190 TDD-row logsums are factorized into the
five-by-five skim-period cache that ActivitySim itself uses internally.  The
large feasible-row arrays and seven timetable columns are intentionally not
copied; Phase 21 must regenerate them from the device timetable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from choiceforge.gpu_scheduling_pipeline import compress_mode_choice_logsums


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark-results" / "phase20-scheduling-replay"
OUTPUT = ROOT / "benchmark-results" / "phase21-scheduling-inputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((args.source / "manifest.json").read_text())
    source_batches = []
    all_person_ids = []
    common_alternatives = None
    for meta in source_manifest["batches"]:
        with np.load(args.source / meta["file"]) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        person_ids = data["chooser_ids"] // np.int64(41)
        all_person_ids.append(person_ids)
        alternatives = data["alternative_values"]
        if common_alternatives is None:
            common_alternatives = alternatives.copy()
        elif not np.array_equal(common_alternatives, alternatives):
            raise RuntimeError("scheduling batches do not share one TDD table")
        source_batches.append((meta, data, person_ids))

    person_ids = np.unique(np.concatenate(all_person_ids)).astype(np.int64)
    np.savez_compressed(
        args.output / "common.npz",
        person_ids=person_ids,
        alternative_values=common_alternatives,
    )

    manifest = {
        "format_version": 1,
        "phase": 21,
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "person_count": int(person_ids.size),
        "alternative_count": int(common_alternatives.shape[0]),
        "skim_period_count": 5,
        "logsum_slots_per_chooser": 25,
        "valid_logsum_slots_per_first_tour": 15,
        "common_file": "common.npz",
        "batches": [],
    }
    for number, (meta, data, batch_person_ids) in enumerate(source_batches):
        cache, present = compress_mode_choice_logsums(
            data["offsets"],
            data["alternative_ids"],
            data["alternative_values"],
            data["row_values"],
        )
        person_rows = np.searchsorted(person_ids, batch_person_ids).astype(np.int32)
        if not np.array_equal(person_ids[person_rows], batch_person_ids):
            raise RuntimeError("failed to create exact person-row mapping")
        selected_rows = data["offsets"][:-1] + data["positions"]
        expected_tdd = data["alternative_ids"][selected_rows]
        stem = f"batch_{number:03d}"
        np.savez_compressed(
            args.output / f"{stem}.npz",
            coefficients=data["coefficients"],
            chooser_ids=data["chooser_ids"],
            person_rows=person_rows,
            chooser_values=data["chooser_values"],
            mode_logsum_cache=cache,
            mode_logsum_present=present,
            draws=data["draws"],
            expected_tdd=expected_tdd,
        )
        manifest["batches"].append(
            {
                "file": f"{stem}.npz",
                "trace_label": meta["trace_label"],
                "choosers": meta["choosers"],
                "expected_interaction_rows": meta["interaction_rows"],
                "expressions": meta["compact_expressions"],
                "coefficients_sha256": array_sha256(data["coefficients"]),
                "chooser_columns": meta["chooser_columns"],
                "row_columns": meta["row_columns"],
                "alternative_columns": meta["alternative_columns"],
                "source_alternative_ids_sha256": array_sha256(data["alternative_ids"]),
                "source_offsets_sha256": array_sha256(data["offsets"]),
                "source_row_values_sha256": array_sha256(data["row_values"]),
                "expected_tdd_sha256": array_sha256(expected_tdd),
                "present_logsum_slots": int(present.sum()),
            }
        )

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(path.stat().st_size for path in args.output.glob("*"))
    print(
        f"wrote {len(manifest['batches'])} Phase 21 batches, "
        f"{person_ids.size} timetable rows, {total / 1_000_000:.3f} MB"
    )


if __name__ == "__main__":
    main()
