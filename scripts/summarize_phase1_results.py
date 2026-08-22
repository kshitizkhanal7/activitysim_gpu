"""Create a machine-readable Phase 1 summary from raw benchmark artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = [
    ROOT / "benchmark-results" / "phase1-a4000-32x16.json",
    ROOT / "benchmark-results" / "phase1-a4000-scheduling-shape.json",
]
MTC = ROOT / "benchmark-data" / "prototype_mtc" / "prototype_mtc"
OUTPUT = ROOT / "benchmark-results" / "phase1-summary.json"


def bootstrap_median_ratio(
    numerator: list[float], denominator: list[float], seed: int = 20260810
) -> list[float]:
    rng = np.random.default_rng(seed)
    top = np.asarray(numerator)
    bottom = np.asarray(denominator)
    ratios = np.empty(20_000)
    for i in range(len(ratios)):
        a = rng.choice(top, len(top), replace=True)
        b = rng.choice(bottom, len(bottom), replace=True)
        ratios[i] = np.median(a) / np.median(b)
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def benchmark_summary(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in raw["results"]:
        cpu_methods = {
            name: method
            for name, method in result["methods"].items()
            if not name.startswith("gpu_")
        }
        best_cpu_name, best_cpu = min(
            cpu_methods.items(), key=lambda item: item[1]["median_seconds"]
        )
        for gpu_name in ("gpu_transfer_inclusive", "gpu_resident"):
            gpu = result["methods"][gpu_name]
            check = result["correctness"][gpu_name]
            speedup = best_cpu["median_seconds"] / gpu["median_seconds"]
            rows.append(
                {
                    "choosers": result["choosers"],
                    "alternatives": result["alternatives"],
                    "features": result["features"],
                    "best_cpu_method": best_cpu_name,
                    "best_cpu_median_ms": best_cpu["median_seconds"] * 1000,
                    "gpu_method": gpu_name,
                    "gpu_median_ms": gpu["median_seconds"] * 1000,
                    "speedup_vs_best_cpu": speedup,
                    "speedup_bootstrap_95_percent_interval": bootstrap_median_ratio(
                        best_cpu["seconds"], gpu["seconds"], seed=result["choosers"]
                    ),
                    "choice_mismatches": check["choice_mismatches"],
                    "choice_mismatch_rate": check["choice_mismatches"] / result["choosers"],
                    "max_logsum_absolute_error": check["max_logsum_absolute_error"],
                    "p999_logsum_absolute_error": check["p999_logsum_absolute_error"],
                    "passes_2x_speed_gate": speedup >= 2.0,
                    "passes_zero_choice_mismatch_gate": check["choice_mismatches"] == 0,
                }
            )
    return {
        "source": str(path.relative_to(ROOT)),
        "settings": raw["settings"],
        "environment": raw["environment"],
        "comparisons": rows,
    }


def warm_sharrow_summary() -> dict:
    totals = []
    components: dict[str, list[float]] = {}
    hashes: dict[str, list[str]] = {}
    output_dirs = [MTC / f"output_sharrow_warm{i}" for i in range(1, 4)]
    total_pattern = re.compile(r"Time to execute all models : ([0-9.]+) seconds")
    for output_dir in output_dirs:
        log = (output_dir / "activitysim.log").read_text(encoding="utf-8", errors="replace")
        matches = total_pattern.findall(log)
        if not matches:
            raise RuntimeError(f"No total runtime found in {output_dir}")
        totals.append(float(matches[-1]))
        with (output_dir / "timing_log.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                components.setdefault(row["model_name"], []).append(float(row["seconds"]))
        for filename in (
            "final_households.csv",
            "final_persons.csv",
            "final_tours.csv",
            "final_trips.csv",
        ):
            digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
            hashes.setdefault(filename, []).append(digest)
    component_rows = [
        {
            "component": name,
            "seconds": values,
            "median_seconds": statistics.median(values),
            "fraction_of_median_total": statistics.median(values) / statistics.median(totals),
        }
        for name, values in components.items()
    ]
    component_rows.sort(key=lambda item: item["median_seconds"], reverse=True)
    return {
        "activitysim": "1.4.0",
        "sharrow": "2.16.2",
        "mode": "require, single process, one thread for numerical libraries",
        "total_seconds": totals,
        "median_total_seconds": statistics.median(totals),
        "components": component_rows,
        "final_output_hashes": {
            name: {"identical_across_runs": len(set(values)) == 1, "sha256": values[0]}
            for name, values in hashes.items()
        },
    }


def main() -> None:
    summary = {
        "schema_version": 1,
        "phase": "Phase 1 - strong CPU baselines and warm Sharrow profile",
        "warm_sharrow": warm_sharrow_summary(),
        "linear_choice_benchmarks": [benchmark_summary(path) for path in BENCHMARKS],
        "known_boundary_rows": [
            {
                "workload": "100000 choosers, 190 alternatives, 69 features",
                "row": 53360,
                "numpy_choice": 130,
                "numba_and_gpu_choice": 131,
                "distance_to_numpy_cdf_boundary": 1.1920928955078125e-7,
            },
            {
                "workload": "100000 choosers, 190 alternatives, 69 features",
                "row": 73539,
                "numpy_choice": 116,
                "numba_and_gpu_choice": 117,
                "distance_to_numpy_cdf_boundary": 2.9802322387695312e-7,
            },
        ],
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
