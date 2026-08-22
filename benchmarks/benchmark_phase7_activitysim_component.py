"""Summarize the interleaved Phase 7 full-model A/B experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path("benchmark-data/prototype_mtc/prototype_mtc")
OUT = Path("benchmark-results/phase7-activitysim-summary.json")
COMPONENT = re.compile(r"time to execute run\.trip_destination\s*: ([0-9.]+) seconds", re.I)
ALL_MODELS = re.compile(r"Time to execute all models\s*: ([0-9.]+) seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(path: Path) -> dict[str, float]:
    text = (path / "activitysim.log").read_text(encoding="utf-8", errors="replace")
    component, all_models = COMPONENT.findall(text), ALL_MODELS.findall(text)
    if len(component) != 1 or len(all_models) != 1:
        raise RuntimeError(f"missing unique timers in {path}")
    return {"component": float(component[0]), "all_models": float(all_models[0])}


def metrics(baseline: list[float], optimized: list[float]) -> dict:
    a, b = statistics.median(baseline), statistics.median(optimized)
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
    baseline_dirs = [ROOT / f"output_phase7_ab_a{i}" for i in (1, 2, 3)]
    optimized_dirs = [ROOT / f"output_phase7_ab_b{i}" for i in (1, 2, 3)]
    baseline, optimized = map(lambda paths: [parse_run(path) for path in paths], (baseline_dirs, optimized_dirs))

    reference = ROOT / "output_phase5_final3"
    files = sorted(p.name for p in reference.glob("final_*.csv") if p.name != "final_checkpoints.csv")
    hashes = {name: sha256(reference / name) for name in files}
    mismatches = {
        path.name: [name for name, expected in hashes.items() if sha256(path / name) != expected]
        for path in baseline_dirs + optimized_dirs
    }
    if any(mismatches.values()):
        raise RuntimeError(f"substantive output mismatch: {mismatches}")

    summary = {
        "phase": "7E",
        "date": "2026-08-11",
        "hardware": "NVIDIA RTX A4000 16 GB",
        "software": {"activitysim": "1.4.0", "sharrow": "required", "cupy": "13.6.0"},
        "design": "interleaved A1/B1/A2/B2/A3/B3 fresh-process trials",
        "baseline": "Phase 5 scheduling plus Phase 6 combined-direction destination logsums",
        "optimized": "Phase 5 plus trip-number preprocessing batches and transfer-inclusive CUDA MTC21 nested-logit reduction",
        "trip_destination": metrics([x["component"] for x in baseline], [x["component"] for x in optimized]),
        "all_models": metrics([x["all_models"] for x in baseline], [x["all_models"] for x in optimized]),
        "correctness": {
            "reference": reference.name,
            "checked_trials": [path.name for path in baseline_dirs + optimized_dirs],
            "byte_identical_substantive_files": files,
            "file_count": len(files),
            "sha256": hashes,
            "checkpoint_csv_excluded_reason": "contains run timing metadata",
        },
        "notes": [
            "A and B use identical public benchmark data and differ only in the explicit destination backend overlay.",
            "ActivitySim's own component and all-model timers include setup; no time is subtracted.",
            "The CUDA kernel has a CPU fallback over the already-evaluated utilities and does not repeat random draws.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
