"""Qualify Phase 28 formulas on changed synthetic populations and skims."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class Invocation:
    rows: int
    float_inputs: object
    int_inputs: object
    skim_arguments: tuple
    logical_skim_bindings: int
    dense_input_bytes: int
    skim_coordinate_bytes: int
    float_input_sources: tuple
    int_input_sources: tuple
    skim_input_sources: tuple
    skim_input_ranks: tuple
    skim_input_groups: tuple


LABELS = (
    "name:sovtoll_available", "name:hov2toll_available",
    "name:walk_local_available", "name:walk_commuter_available",
    "name:walk_express_available", "name:walk_heavyrail_available",
    "name:walk_lrf_available", "column:walk_ferry_available",
    "name:drive_local_available", "name:drive_commuter_available",
    "name:drive_express_available", "name:drive_heavyrail_available",
    "name:drive_lrf_available", "column:drive_ferry_available",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from choiceforge.device_input_expansion import ResidentSemanticInputPlan
    from choiceforge.semantic_input_generation import _availability_expression

    required = []
    for label in LABELS:
        _availability_expression(
            label,
            lambda source: required.append(source) or "value",
            0,
        )
    sources = tuple(dict.fromkeys(required))
    results = []
    covered = set()
    for seed in (2801, 2802, 2803, 2804, 2805):
        invocation, metadata, expected = scenario(cp, seed, sources)
        plan = ResidentSemanticInputPlan.compile(invocation, metadata)
        plan.execute()
        cp.cuda.Stream.null.synchronize()
        manifest = plan.semantic_program.manifest()
        covered.update(item["source"] for item in manifest["expressions"])
        exact = bool(
            cp.array_equal(plan.invocation.float_inputs, invocation.float_inputs)
            and cp.array_equal(plan.invocation.int_inputs, invocation.int_inputs)
        )
        results.append(
            {
                "seed": seed,
                "rows": invocation.rows,
                "exact": exact,
                "generated_float_columns": manifest["generated_float_columns"],
                "generated_int_columns": manifest["generated_int_columns"],
                "anonymous_response_pattern_columns": manifest[
                    "anonymous_response_pattern_columns"
                ],
                "expected_sha256": hashlib.sha256(expected).hexdigest(),
            }
        )
    expected_sources = {"column:daily_parking_cost", *LABELS}
    report = {
        "phase": 28,
        "scope": (
            "five independently generated changed populations, parking rates, "
            "alternative times, zone coordinates, and raw skim cubes"
        ),
        "scenarios": results,
        "covered_semantic_sources": sorted(covered),
        "proof_gates": {
            "five_changed_scenarios": len(results) == 5,
            "all_scenarios_bit_exact": all(item["exact"] for item in results),
            "no_anonymous_patterns": all(
                item["anonymous_response_pattern_columns"] == 0 for item in results
            ),
            "all_fifteen_formulas_exercised": expected_sources <= covered,
            "different_scenario_outputs": len({item["expected_sha256"] for item in results}) == 5,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(report["proof_gates"].values()):
        raise SystemExit("Phase 28 changed-scenario qualification failed")
    print(json.dumps(report, indent=2))


def scenario(cp, seed, sources):
    rng = np.random.default_rng(seed)
    owners_count, alternatives, zones, periods = 64, 25, 7, 5
    rows = owners_count * alternatives
    owner = np.repeat(np.arange(owners_count, dtype=np.int32), alternatives)
    local = np.tile(np.arange(alternatives, dtype=np.int32), owners_count)
    start_values = np.array((5, 8, 11, 15, 19), dtype=np.int16)
    end_values = np.array((6, 9, 13, 17, 23), dtype=np.int16)
    start = start_values[local // 5]
    end = end_values[local % 5]
    duration = end.astype(np.int64) - start.astype(np.int64)
    chooser_ids = np.repeat(np.arange(seed * 1000, seed * 1000 + owners_count), alternatives)
    archetype = owner % 8
    auto = (archetype % 4).astype(np.int64)

    group_coordinates = {}
    for direction, group in (("odt_skims", 0), ("dot_skims", 1)):
        base_o = (archetype + group) % zones
        base_d = (archetype * 2 + 1 + group) % zones
        origin = base_o.astype(np.int64)
        destination = base_d.astype(np.int64)
        time_index = (local % periods).astype(np.int64)
        group_coordinates[direction] = (origin, destination, time_index)

    cubes = {}
    gathered = {}
    for source in sources:
        cube = rng.choice(
            np.array((-2.0, 0.0, 0.0, 1.0, 3.0), dtype=np.float32),
            size=(zones, zones, periods),
        ).astype(np.float32)
        cubes[source] = cube
        direction = source[1]
        origin, destination, time_index = group_coordinates[direction]
        gathered[source] = cube[origin, destination, time_index]

    def v(direction, key):
        return gathered[("skim", direction, key)]

    def scaled(array):
        return array / np.float32(100.0)

    values = {}
    values["name:sovtoll_available"] = (
        (v("odt_skims", "SOVTOLL_VTOLL") > 0)
        | (v("dot_skims", "SOVTOLL_VTOLL") > 0)
    )
    values["name:hov2toll_available"] = (
        v("odt_skims", "HOV2TOLL_VTOLL")
        + v("dot_skims", "HOV2TOLL_VTOLL") > 0
    )
    for mode, suffix in (("local", "LOC"), ("commuter", "COM"),
                         ("express", "EXP"), ("heavyrail", "HVY"), ("lrf", "LRF")):
        walk = (
            (scaled(v("odt_skims", f"WLK_{suffix}_WLK_TOTIVT")) > 0)
            & (scaled(v("dot_skims", f"WLK_{suffix}_WLK_TOTIVT")) > 0)
        )
        drive = (
            (auto[owner] > 0)
            & (scaled(v("odt_skims", f"DRV_{suffix}_WLK_TOTIVT")) > 0)
            & (scaled(v("dot_skims", f"WLK_{suffix}_DRV_TOTIVT")) > 0)
        )
        if suffix != "LOC":
            walk &= (
                scaled(v("odt_skims", f"WLK_{suffix}_WLK_KEYIVT"))
                + scaled(v("dot_skims", f"WLK_{suffix}_WLK_KEYIVT")) > 0
            )
            drive &= (
                scaled(v("odt_skims", f"DRV_{suffix}_WLK_KEYIVT"))
                + scaled(v("dot_skims", f"WLK_{suffix}_DRV_KEYIVT")) > 0
            )
        values[f"name:walk_{mode}_available"] = walk
        values[f"name:drive_{mode}_available"] = drive
    values["column:walk_ferry_available"] = (
        values["name:walk_lrf_available"]
        & (scaled(v("odt_skims", "WLK_LRF_WLK_FERRYIVT"))
           + scaled(v("dot_skims", "WLK_LRF_WLK_FERRYIVT")) > 0)
    )
    values["column:drive_ferry_available"] = (
        values["name:drive_lrf_available"]
        & (scaled(v("odt_skims", "DRV_LRF_WLK_FERRYIVT"))
           + scaled(v("dot_skims", "WLK_LRF_WLK_FERRYIVT")) > 0)
    )

    rates = rng.uniform(0.05, 8.0, size=8)
    parking = (rates[archetype][owner] * duration).astype(np.float32)
    float_host = parking.reshape(-1, 1)
    int_host = np.column_stack((auto[owner], *(values[label] for label in LABELS))).astype(np.int64)
    float_inputs = cp.asarray(float_host)
    int_inputs = cp.asarray(int_host)

    skim_arguments = [cp.asarray(cubes[source].reshape(-1)) for source in sources]
    coordinate_bytes = 0
    for direction in ("odt_skims", "dot_skims"):
        for array in group_coordinates[direction]:
            device = cp.asarray(array, dtype=cp.int64)
            skim_arguments.append(device)
            coordinate_bytes += int(device.nbytes)
        skim_arguments.extend((np.int64(zones), np.int64(periods)))
    groups = tuple(0 if source[1] == "odt_skims" else 1 for source in sources)
    invocation = Invocation(
        rows=rows,
        float_inputs=float_inputs,
        int_inputs=int_inputs,
        skim_arguments=tuple(skim_arguments),
        logical_skim_bindings=len(sources),
        dense_input_bytes=int(float_inputs.nbytes + int_inputs.nbytes),
        skim_coordinate_bytes=coordinate_bytes,
        float_input_sources=(("column", "daily_parking_cost"),),
        int_input_sources=(("name", "auto_ownership"),) + tuple(
            tuple(label.split(":", 1)) for label in LABELS
        ),
        skim_input_sources=sources,
        skim_input_ranks=(3,) * len(sources),
        skim_input_groups=groups,
    )
    metadata = {"chooser_ids": chooser_ids, "start": start, "end": end}
    expected = np.ascontiguousarray(float_host).tobytes() + np.ascontiguousarray(int_host).tobytes()
    return invocation, metadata, expected


if __name__ == "__main__":
    main()
