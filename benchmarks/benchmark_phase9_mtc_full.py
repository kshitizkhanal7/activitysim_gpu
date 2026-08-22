"""Summarize one or more Phase 9 full-geography MTC A/B runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import random
import statistics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "benchmark-results" / "phase9-mtc-full-destination-only-runs.json"
DEFAULT_OUTPUT = REPO / "benchmark-results" / "phase9-mtc-full-destination-only-summary.json"
ALL_MODELS = re.compile(r"Time to execute all models\s*: ([0-9.]+) seconds")
HIGH_WATER_RSS = re.compile(r"MainProcess high water mark rss: ([0-9_]+)")
HIGH_WATER_USS = re.compile(r"MainProcess high water mark uss: ([0-9_]+)")
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


def parse_run(run: dict) -> dict:
    output = Path(run["output"])
    with (output / "timing_log.csv").open(newline="", encoding="utf-8") as stream:
        timers = {row["model_name"]: float(row["seconds"]) for row in csv.DictReader(stream)}
    stdout = Path(run["stdout"]).read_text(encoding="utf-8", errors="replace")
    stderr = Path(run["stderr"]).read_text(encoding="utf-8", errors="replace")
    activitysim_log = (output / "activitysim.log").read_text(
        encoding="utf-8", errors="replace"
    )
    all_models = ALL_MODELS.findall(stdout)
    if len(all_models) != 1:
        raise RuntimeError(f"missing unique all-model timer for {run['name']}")
    all_logs = "\n".join((stdout, stderr, activitysim_log))
    rss = HIGH_WATER_RSS.findall(all_logs)
    uss = HIGH_WATER_USS.findall(all_logs)
    return {
        **run,
        "all_models": float(all_models[0]),
        "activitysim_high_water_rss_bytes": int(rss[-1].replace("_", "")) if rss else None,
        "activitysim_high_water_uss_bytes": int(uss[-1].replace("_", "")) if uss else None,
        "scheduling_suite": sum(timers[item] for item in SCHEDULING),
        **{item: timers[item] for item in COMPONENTS},
    }


def speed_metrics(baseline: list[float], optimized: list[float]) -> dict:
    a, b = statistics.median(baseline), statistics.median(optimized)
    result = {
        "baseline_seconds": baseline,
        "optimized_seconds": optimized,
        "baseline_median_seconds": a,
        "optimized_median_seconds": b,
        "median_seconds_saved": a - b,
        "speedup": a / b,
    }
    if len(baseline) >= 2 and len(optimized) >= 2:
        result["all_optimized_faster_than_all_baseline"] = max(optimized) < min(baseline)
        result["worst_optimized_vs_best_baseline_speedup"] = min(baseline) / max(optimized)
        # Deterministic nonparametric interval for the median paired saving.
        # Three repetitions remain a small experiment, so this is reported as
        # an uncertainty descriptor rather than a population guarantee.
        generator = random.Random(20260817)
        paired_savings = [a - b for a, b in zip(baseline, optimized)]
        bootstrap = sorted(
            statistics.median(
                [paired_savings[generator.randrange(len(paired_savings))] for _ in paired_savings]
            )
            for _ in range(10_000)
        )
        result["paired_seconds_saved"] = paired_savings
        result["bootstrap_median_seconds_saved_95pct"] = [
            bootstrap[int(0.025 * len(bootstrap))],
            bootstrap[int(0.975 * len(bootstrap)) - 1],
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    runs = [parse_run(run) for run in manifest["runs"]]
    baseline = [run for run in runs if run["condition"] == "baseline"]
    optimized = [run for run in runs if run["condition"] == "choiceforge"]
    if len(baseline) != len(optimized) or not baseline:
        raise RuntimeError("need equal nonzero baseline and ChoiceForge run counts")

    reference = Path(baseline[0]["output"])
    files = sorted(
        path.name
        for path in reference.glob("final_*.csv")
        if path.name != "final_checkpoints.csv"
    )
    expected = {name: sha256(reference / name) for name in files}
    mismatches = {
        run["name"]: [name for name, digest in expected.items() if sha256(Path(run["output"]) / name) != digest]
        for run in runs
    }
    if any(mismatches.values()):
        raise RuntimeError(f"substantive output mismatch: {mismatches}")

    keys = ("all_models", "scheduling_suite") + COMPONENTS
    summary = {
        "phase": manifest.get("phase", "9A"),
        "manifest": str(args.manifest),
        "benchmark": {
            **{
                key: manifest[key]
                for key in ("benchmark", "full_population_households", "zones", "data_sha256", "design")
            },
            "reproducibility": manifest.get("reproducibility"),
        },
        "runs": [{key: value for key, value in run.items() if key not in {"output", "stdout", "stderr"}} for run in runs],
        "metrics": {key: speed_metrics([run[key] for run in baseline], [run[key] for run in optimized]) for key in keys},
        "correctness": {
            "reference": baseline[0]["name"],
            "checked_runs": [run["name"] for run in runs],
            "byte_identical_substantive_files": files,
            "sha256": expected,
            "rows": {name: row_count(reference / name) for name in ("final_households.csv", "final_persons.csv", "final_tours.csv", "final_trips.csv")},
            "checkpoint_csv_excluded_reason": "contains run timing metadata",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
