"""Summarize Phase 8's current-ActivitySim 50k-household A/B experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "benchmark-data" / "phase8-mtc-mini" / "prototype_mtc_sf"
OUT = REPO / "benchmark-results" / "phase8-activitysim-summary.json"
ALL_MODELS = re.compile(r"Time to execute all models\s*: ([0-9.]+) seconds")
SCHEDULING_TELEMETRY = re.compile(
    r"ChoiceForge rows=(\d+) choosers=(\d+) compact=([0-9.]+)MB .*?"
    r"gpu=([0-9.]+)ms .*?total=([0-9.]+)ms"
)
DESTINATION_TELEMETRY = re.compile(
    r"ChoiceForge trip-number batch purposes=(\d+) rows=(\d+) "
    r"preprocessor=([0-9.]+)ms total=([0-9.]+)ms"
)
SCHEDULING = (
    "mandatory_tour_scheduling",
    "joint_tour_scheduling",
    "non_mandatory_tour_scheduling",
    "atwork_subtour_scheduling",
)
COMPONENTS = SCHEDULING + ("trip_destination",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def parse_run(condition: str, trial: int) -> dict:
    name = f"phase8-ab-{condition}{trial}"
    output = ROOT / f"output-{name}"
    with (output / "timing_log.csv").open(newline="", encoding="utf-8") as stream:
        timings = {
            row["model_name"]: float(row["seconds"])
            for row in csv.DictReader(stream)
        }
    text = (ROOT / f"{name}.stdout.log").read_text(
        encoding="utf-8", errors="replace"
    )
    total = ALL_MODELS.findall(text)
    if len(total) != 1:
        raise RuntimeError(f"missing unique all-model timer for {name}")
    scheduling_telemetry = [
        {
            "rows": int(rows),
            "choosers": int(choosers),
            "compact_mb": float(compact),
            "gpu_ms": float(gpu),
            "total_ms": float(elapsed),
        }
        for rows, choosers, compact, gpu, elapsed in SCHEDULING_TELEMETRY.findall(text)
    ]
    destination_telemetry = [
        {
            "purposes": int(purposes),
            "rows": int(rows),
            "preprocessor_ms": float(preprocessor),
            "total_ms": float(elapsed),
        }
        for purposes, rows, preprocessor, elapsed in DESTINATION_TELEMETRY.findall(text)
    ]
    return {
        "name": name,
        "output": output,
        "all_models": float(total[0]),
        "scheduling_suite": sum(timings[item] for item in SCHEDULING),
        "scheduling_telemetry": scheduling_telemetry,
        "destination_telemetry": destination_telemetry,
        **{item: timings[item] for item in COMPONENTS},
    }


def metrics(baseline: list[float], optimized: list[float]) -> dict:
    a = statistics.median(baseline)
    b = statistics.median(optimized)
    return {
        "baseline_seconds": baseline,
        "optimized_seconds": optimized,
        "baseline_median_seconds": a,
        "optimized_median_seconds": b,
        "median_seconds_saved": a - b,
        "speedup": a / b,
        "paired_seconds_saved": [x - y for x, y in zip(baseline, optimized)],
        "all_optimized_faster_than_all_baseline": max(optimized) < min(baseline),
        "worst_optimized_vs_best_baseline_speedup": min(baseline) / max(optimized),
    }


def main() -> None:
    baseline = [parse_run("a", trial) for trial in (1, 2, 3)]
    optimized = [parse_run("b", trial) for trial in (1, 2, 3)]
    reference = baseline[0]["output"]
    files = sorted(
        path.name
        for path in reference.glob("final_*.csv")
        if path.name != "final_checkpoints.csv"
    )
    hashes = {name: sha256(reference / name) for name in files}
    checked = baseline + optimized
    mismatches = {
        run["name"]: [
            name
            for name, expected in hashes.items()
            if sha256(run["output"] / name) != expected
        ]
        for run in checked
    }
    if any(mismatches.values()):
        raise RuntimeError(f"substantive output mismatch: {mismatches}")

    keys = ("all_models", "scheduling_suite") + COMPONENTS
    summary = {
        "phase": "8A",
        "date": "2026-08-11",
        "hardware": {
            "gpu": "NVIDIA RTX A4000 16 GB",
            "nvidia_driver": "571.59",
            "logical_cpus": 48,
            "ram_gb": 63.9,
            "operating_system": "Windows",
        },
        "software": {
            "activitysim_commit": "16ab11180a26912987eb902daf945e268f3efc11",
            "activitysim_version": "1000.dev1+g16ab11180",
            "python": "3.11.14",
            "numpy": "2.4.6",
            "pandas": "2.3.3",
            "cupy": "14.1.1",
            "cuda_runtime": "12.9",
        },
        "benchmark": {
            "workflow": "ActivitySim sharrow-contrast/mtc_mini",
            "dataset": "public prototype_mtc_sf",
            "households": 50000,
            "zones": 190,
            "design": "interleaved fresh-process A1/B1/A2/B2/A3/B3",
            "baseline": "pinned current ActivitySim; Sharrow required",
            "optimized": "baseline plus explicit ChoiceForge scheduling and destination overlays",
            "openblas_threads": 24,
        },
        "metrics": {
            key: metrics(
                [run[key] for run in baseline],
                [run[key] for run in optimized],
            )
            for key in keys
        },
        "correctness": {
            "reference": baseline[0]["name"],
            "checked_trials": [run["name"] for run in checked],
            "byte_identical_substantive_files": files,
            "file_count": len(files),
            "sha256": hashes,
            "rows": {
                name: row_count(reference / name)
                for name in (
                    "final_households.csv",
                    "final_persons.csv",
                    "final_tours.csv",
                    "final_trips.csv",
                )
            },
            "checkpoint_csv_excluded_reason": "contains run timing metadata",
        },
        "real_boundary_telemetry": {
            run["name"]: {
                "scheduling_calls": len(run["scheduling_telemetry"]),
                "scheduling_rows": sum(
                    item["rows"] for item in run["scheduling_telemetry"]
                ),
                "largest_scheduling_call": max(
                    run["scheduling_telemetry"],
                    key=lambda item: item["rows"],
                ),
                "destination_batches": run["destination_telemetry"],
            }
            for run in optimized
        },
        "notes": [
            "ActivitySim's own component and whole-model timers are used; no setup cost is subtracted.",
            "The GPU timings include host preparation, transfers, kernel execution, result mapping, and conservative fallbacks.",
            "The official 500-household compile pass is reported separately and excluded from warmed A/B timings.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
