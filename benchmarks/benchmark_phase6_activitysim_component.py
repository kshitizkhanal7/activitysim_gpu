"""Summarize the interleaved Phase 6 ActivitySim A/B experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


COMPONENT = re.compile(
    r"time to execute run\.trip_destination\s*: ([0-9.]+) seconds",
    re.IGNORECASE,
)
ALL_MODELS = re.compile(r"Time to execute all models\s*: ([0-9.]+) seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(path: Path) -> dict:
    text = (path / "activitysim.log").read_text(encoding="utf-8", errors="replace")
    component = COMPONENT.findall(text)
    all_models = ALL_MODELS.findall(text)
    if len(component) != 1 or len(all_models) != 1:
        raise RuntimeError(f"expected one component and whole-model timing in {path}")
    return {
        "trip_destination_seconds": float(component[0]),
        "all_models_seconds": float(all_models[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark-data/prototype_mtc/prototype_mtc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/phase6-activitysim-summary.json"),
    )
    args = parser.parse_args()

    baseline_dirs = [args.root / f"output_phase6_ab_a{i}" for i in (1, 2, 3)]
    optimized_dirs = [args.root / f"output_phase6_ab_b{i}" for i in (1, 2, 3)]
    baseline = [parse_run(path) for path in baseline_dirs]
    optimized = [parse_run(path) for path in optimized_dirs]
    baseline_component = [run["trip_destination_seconds"] for run in baseline]
    optimized_component = [run["trip_destination_seconds"] for run in optimized]
    baseline_all = [run["all_models_seconds"] for run in baseline]
    optimized_all = [run["all_models_seconds"] for run in optimized]

    reference_dir = args.root / "output_phase5_final3"
    files = sorted(
        path.name
        for path in reference_dir.glob("final_*.csv")
        if path.name != "final_checkpoints.csv"
    )
    hashes = {name: sha256(reference_dir / name) for name in files}
    checked_dirs = baseline_dirs + optimized_dirs
    mismatches = {
        path.name: [
            name for name, expected in hashes.items() if sha256(path / name) != expected
        ]
        for path in checked_dirs
    }
    if any(mismatches.values()):
        raise RuntimeError(f"substantive output mismatch: {mismatches}")

    baseline_median = statistics.median(baseline_component)
    optimized_median = statistics.median(optimized_component)
    baseline_all_median = statistics.median(baseline_all)
    optimized_all_median = statistics.median(optimized_all)
    summary = {
        "phase": "6E",
        "date": "2026-08-11",
        "hardware": "NVIDIA RTX A4000 16 GB",
        "software": {"activitysim": "1.4.0", "sharrow": "required"},
        "design": "interleaved A1/B1/A2/B2/A3/B3 fresh-process trials",
        "baseline": "Phase 5 ChoiceForge scheduling plus ActivitySim destination logsums",
        "optimized": "Phase 5 plus ChoiceForge combined-direction destination logsums",
        "trip_destination": {
            "baseline_seconds": baseline_component,
            "optimized_seconds": optimized_component,
            "baseline_median_seconds": baseline_median,
            "optimized_median_seconds": optimized_median,
            "median_seconds_saved": baseline_median - optimized_median,
            "speedup": baseline_median / optimized_median,
            "paired_seconds_saved": [
                a - b for a, b in zip(baseline_component, optimized_component)
            ],
            "all_optimized_faster_than_all_baseline": max(optimized_component)
            < min(baseline_component),
            "worst_optimized_vs_best_baseline_speedup": min(baseline_component)
            / max(optimized_component),
        },
        "all_models": {
            "baseline_seconds": baseline_all,
            "optimized_seconds": optimized_all,
            "baseline_median_seconds": baseline_all_median,
            "optimized_median_seconds": optimized_all_median,
            "median_seconds_saved": baseline_all_median - optimized_all_median,
            "speedup": baseline_all_median / optimized_all_median,
        },
        "correctness": {
            "reference": reference_dir.name,
            "checked_trials": [path.name for path in checked_dirs],
            "byte_identical_substantive_files": files,
            "file_count": len(files),
            "sha256": hashes,
            "checkpoint_csv_excluded_reason": "contains run timing metadata",
        },
        "notes": [
            "A and B use identical data and configs except the explicit destination-logsum backend overlay.",
            "ActivitySim's own component and all-model timers are used; no setup cost is subtracted.",
            "The separate kernel replay proves the batched CUDA boundary; the integrated gain comes from eliminating duplicated directional preprocessing and utility setup.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
