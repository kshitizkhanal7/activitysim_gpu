"""Summarize matched ActivitySim Phase 4 component trials.

The runs themselves are intentionally performed by ActivitySim so its workflow
timer owns the benchmark boundary. This script parses those logs, verifies the
checkpointed tour tables, and writes one machine-readable result artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np
import pandas as pd


COMPONENT = re.compile(r"time to execute run\.mandatory_tour_scheduling\s*: ([0-9.]+) seconds")
TELEMETRY = re.compile(
    r"ChoiceForge rows=(\d+) choosers=(\d+) compact=([0-9.]+)MB "
    r"lower=([0-9.]+)ms stateful=([0-9.]+)ms pack=([0-9.]+)ms "
    r"rng=([0-9.]+)ms gpu=([0-9.]+)ms map=([0-9.]+)ms total=([0-9.]+)ms"
)


def component_seconds(output: Path) -> float:
    text = (output / "activitysim.log").read_text(encoding="utf-8", errors="replace")
    matches = COMPONENT.findall(text)
    if len(matches) != 1:
        raise RuntimeError(f"expected one successful component timing in {output}")
    return float(matches[0])


def telemetry(output: Path):
    text = (output / "activitysim.log").read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in TELEMETRY.findall(text):
        values = list(map(float, match))
        rows.append(
            dict(
                interaction_rows=int(values[0]),
                choosers=int(values[1]),
                compact_mb=values[2],
                lower_ms=values[3],
                stateful_ms=values[4],
                pack_ms=values[5],
                rng_ms=values[6],
                gpu_ms=values[7],
                map_ms=values[8],
                total_ms=values[9],
            )
        )
    if len(rows) != 6:
        raise RuntimeError(f"expected six ChoiceForge batches in {output}")
    return rows


def tour_table(output: Path) -> pd.DataFrame:
    return pd.read_parquet(
        output
        / "pipeline.parquetpipeline"
        / "tours"
        / "mandatory_tour_scheduling.parquet"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("benchmark-data/prototype_mtc/prototype_mtc"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/phase4-summary.json"))
    args = parser.parse_args()
    baseline_dirs = [args.root / f"output_phase4_baseline{i}" for i in (1, 2, 3)]
    choiceforge_dirs = [args.root / f"output_phase4_optimized{i}" for i in (3, 4, 5)]
    baseline = [component_seconds(x) for x in baseline_dirs]
    choiceforge = [component_seconds(x) for x in choiceforge_dirs]
    baseline_median = statistics.median(baseline)
    choiceforge_median = statistics.median(choiceforge)

    reference = tour_table(baseline_dirs[1])
    mismatches = {}
    for column in ("tdd", "start", "end", "duration"):
        mismatches[column] = max(
            int(np.count_nonzero(reference[column].to_numpy() != tour_table(x)[column].to_numpy()))
            for x in choiceforge_dirs
        )

    batch_runs = [telemetry(x) for x in choiceforge_dirs]
    largest = []
    for run in batch_runs:
        largest.append(max(run, key=lambda x: x["interaction_rows"]))
    largest_medians = {
        key: statistics.median(x[key] for x in largest)
        for key in ("stateful_ms", "pack_ms", "rng_ms", "gpu_ms", "map_ms", "total_ms")
    }
    full_baseline_dirs = [args.root / f"output_sharrow_warm{i}" for i in (1, 2, 3)]
    full_choiceforge_dirs = [
        args.root / "output_phase4_final",
        args.root / "output_phase4_final2",
        args.root / "output_phase4_final3",
    ]
    full_baseline = [component_seconds(x) for x in full_baseline_dirs]
    full_choiceforge = [component_seconds(x) for x in full_choiceforge_dirs]
    full_reference = pd.read_csv(full_baseline_dirs[2] / "final_tours.csv", index_col=0)
    full_mismatches = {}
    for column in ("tdd", "start", "end", "duration", "destination_logsum", "mode_choice_logsum"):
        full_mismatches[column] = max(
            int(np.count_nonzero(~np.isclose(
                full_reference[column].to_numpy(),
                pd.read_csv(x / "final_tours.csv", index_col=0)[column].to_numpy(),
                rtol=0,
                atol=0,
                equal_nan=True,
            )))
            for x in full_choiceforge_dirs
        )
    summary = {
        "phase": 4,
        "date": "2026-08-11",
        "hardware": "NVIDIA RTX A4000 16 GB",
        "software": {"activitysim": "1.4.0", "backend": "ChoiceForge compact CUDA"},
        "boundary": "ActivitySim mandatory_tour_scheduling workflow component",
        "trials": 3,
        "baseline_cached_sharrow_seconds": baseline,
        "choiceforge_seconds": choiceforge,
        "baseline_median_seconds": baseline_median,
        "choiceforge_median_seconds": choiceforge_median,
        "whole_component_speedup": baseline_median / choiceforge_median,
        "normal_full_run_trials": {
            "cached_sharrow_seconds": full_baseline,
            "choiceforge_seconds": full_choiceforge,
            "cached_sharrow_median_seconds": statistics.median(full_baseline),
            "choiceforge_median_seconds": statistics.median(full_choiceforge),
            "whole_component_speedup": statistics.median(full_baseline) / statistics.median(full_choiceforge),
            "final_tour_rows": len(full_reference),
            "max_mismatches": full_mismatches,
        },
        "correctness": {
            "mandatory_tours": len(reference),
            "max_mismatches_across_choiceforge_trials": mismatches,
        },
        "largest_batch_median_stage_ms": largest_medians,
        "largest_batch": {
            "choosers": largest[0]["choosers"],
            "interaction_rows": largest[0]["interaction_rows"],
            "compact_mb": largest[0]["compact_mb"],
        },
        "notes": [
            "Each trial restores the identical mandatory_tour_frequency checkpoint and cached Sharrow assets.",
            "The timer includes mode-choice logsums, compact state extraction, packing, ActivitySim RNG, host-device transfers, GPU execution, pandas mapping, and timetable updates.",
            "ChoiceForge compilation/context initialization remains inside each fresh-process component trial; it is not hidden.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
