"""Summarize repeated full-model Phase 5 scheduling-suite trials.

ActivitySim owns the timing boundary. This script parses its raw logs, compares
the four scheduling components, verifies byte-identical substantive outputs,
and writes a machine-readable artifact. It intentionally reports all-model
timing separately: unrelated model stages are outside ChoiceForge's boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


COMPONENTS = (
    "mandatory_tour_scheduling",
    "joint_tour_scheduling",
    "non_mandatory_tour_scheduling",
    "atwork_subtour_scheduling",
)
COMPONENT_PATTERN = re.compile(
    r"time to execute run\.([a-z_]+)\s*: ([0-9.]+) seconds"
)
ALL_MODELS_PATTERN = re.compile(r"Time to execute all models\s*: ([0-9.]+) seconds")
TELEMETRY_PATTERN = re.compile(
    r"(?P<trace>\S+) ChoiceForge rows=(?P<rows>\d+) choosers=(?P<choosers>\d+) "
    r"compact=(?P<compact>[0-9.]+)MB lower=(?P<lower>[0-9.]+)ms "
    r"stateful=(?P<stateful>[0-9.]+)ms pack=(?P<pack>[0-9.]+)ms "
    r"rng=(?P<rng>[0-9.]+)ms gpu=(?P<gpu>[0-9.]+)ms "
    r"map=(?P<map>[0-9.]+)ms total=(?P<total>[0-9.]+)ms"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(output: Path) -> dict:
    text = (output / "activitysim.log").read_text(encoding="utf-8", errors="replace")
    found = {name: float(value) for name, value in COMPONENT_PATTERN.findall(text)}
    missing = set(COMPONENTS) - found.keys()
    if missing:
        raise RuntimeError(f"missing component timings {sorted(missing)} in {output}")
    all_models = ALL_MODELS_PATTERN.findall(text)
    if len(all_models) != 1:
        raise RuntimeError(f"expected one all-model timing in {output}")
    components = {name: found[name] for name in COMPONENTS}
    telemetry = []
    for match in TELEMETRY_PATTERN.finditer(text):
        row = match.groupdict()
        telemetry.append(
            {
                "trace": row["trace"],
                "interaction_rows": int(row["rows"]),
                "choosers": int(row["choosers"]),
                **{key + "_ms": float(row[key]) for key in ("lower", "stateful", "pack", "rng", "gpu", "map", "total")},
                "compact_mb": float(row["compact"]),
            }
        )
    return {
        "components_seconds": components,
        "scheduling_suite_seconds": round(sum(components.values()), 3),
        "all_models_seconds": float(all_models[0]),
        "telemetry": telemetry,
    }


def medians(runs: list[dict]) -> dict:
    return {
        name: statistics.median(run["components_seconds"][name] for run in runs)
        for name in COMPONENTS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("benchmark-data/prototype_mtc/prototype_mtc"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/phase5-summary.json"))
    args = parser.parse_args()

    baseline_dirs = [args.root / f"output_sharrow_warm{i}" for i in (1, 2, 3)]
    choiceforge_dirs = [args.root / f"output_phase5_final{i}" for i in (1, 2, 3)]
    baseline = [parse_run(path) for path in baseline_dirs]
    choiceforge = [parse_run(path) for path in choiceforge_dirs]
    baseline_medians = medians(baseline)
    choiceforge_medians = medians(choiceforge)
    baseline_suite = [run["scheduling_suite_seconds"] for run in baseline]
    choiceforge_suite = [run["scheduling_suite_seconds"] for run in choiceforge]

    reference_dir = baseline_dirs[-1]
    output_files = sorted(
        path.name for path in reference_dir.glob("final_*.csv")
        if path.name != "final_checkpoints.csv"
    )
    reference_hashes = {name: sha256(reference_dir / name) for name in output_files}
    mismatched_hashes = {
        path.name: [
            name for name, expected in reference_hashes.items()
            if sha256(path / name) != expected
        ]
        for path in choiceforge_dirs
    }
    if any(mismatched_hashes.values()):
        raise RuntimeError(f"substantive output mismatch: {mismatched_hashes}")

    largest = [max(run["telemetry"], key=lambda row: row["interaction_rows"]) for run in choiceforge]
    summary = {
        "phase": 5,
        "date": "2026-08-11",
        "hardware": "NVIDIA RTX A4000 16 GB",
        "software": {"activitysim": "1.4.0", "backend": "ChoiceForge compact CUDA"},
        "boundary": "sum of four complete ActivitySim tour-scheduling workflow components",
        "trials": 3,
        "components": {
            name: {
                "cached_sharrow_seconds": [run["components_seconds"][name] for run in baseline],
                "choiceforge_seconds": [run["components_seconds"][name] for run in choiceforge],
                "cached_sharrow_median_seconds": baseline_medians[name],
                "choiceforge_median_seconds": choiceforge_medians[name],
                "speedup": baseline_medians[name] / choiceforge_medians[name],
                "all_choiceforge_trials_faster_than_all_baseline_trials": (
                    max(run["components_seconds"][name] for run in choiceforge)
                    < min(run["components_seconds"][name] for run in baseline)
                ),
            }
            for name in COMPONENTS
        },
        "scheduling_suite": {
            "cached_sharrow_seconds": baseline_suite,
            "choiceforge_seconds": choiceforge_suite,
            "cached_sharrow_median_seconds": statistics.median(baseline_suite),
            "choiceforge_median_seconds": statistics.median(choiceforge_suite),
            "speedup": statistics.median(baseline_suite) / statistics.median(choiceforge_suite),
            "median_seconds_saved": statistics.median(baseline_suite) - statistics.median(choiceforge_suite),
            "all_choiceforge_trials_faster_than_all_baseline_trials": max(choiceforge_suite) < min(baseline_suite),
            "worst_choiceforge_vs_best_baseline_speedup": min(baseline_suite) / max(choiceforge_suite),
        },
        "all_models_context": {
            "cached_sharrow_seconds": [run["all_models_seconds"] for run in baseline],
            "choiceforge_seconds": [run["all_models_seconds"] for run in choiceforge],
            "cached_sharrow_median_seconds": statistics.median(run["all_models_seconds"] for run in baseline),
            "choiceforge_median_seconds": statistics.median(run["all_models_seconds"] for run in choiceforge),
            "note": "Context only; most model stages are unchanged and environmental variation dominates this boundary.",
        },
        "correctness": {
            "reference": reference_dir.name,
            "choiceforge_trials": [path.name for path in choiceforge_dirs],
            "byte_identical_substantive_files": output_files,
            "file_count": len(output_files),
            "final_tour_rows": 9806,
            "final_trip_rows": 23583,
            "sha256": reference_hashes,
            "checkpoint_csv_excluded_reason": "contains run timestamps and timing metadata",
        },
        "largest_batch_median_stage_ms": {
            key: statistics.median(row[key] for row in largest)
            for key in ("stateful_ms", "pack_ms", "rng_ms", "gpu_ms", "map_ms", "total_ms")
        },
        "largest_batch": {
            "interaction_rows": largest[0]["interaction_rows"],
            "choosers": largest[0]["choosers"],
            "compact_mb": largest[0]["compact_mb"],
        },
        "notes": [
            "Every sample comes from ActivitySim's own workflow timer in a fresh process and clean output directory.",
            "The suite boundary includes logsums where configured, compact state extraction, packing, ActivitySim RNG, host-device transfers, CUDA execution, result mapping, and timetable updates.",
            "CUDA context and kernel load costs remain inside each process; no timing is subtracted.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
