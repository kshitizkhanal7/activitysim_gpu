"""Qualify Phase 29 on changed raw tables and changed resident skim cubes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


LABELS18 = (
    "name:sov_available", "name:sovtoll_available",
    "name:hov2_available", "name:hov2toll_available",
    "name:hov3_available", "name:hov3toll_available",
    "name:walk_local_available", "name:walk_commuter_available",
    "name:walk_express_available", "name:walk_heavyrail_available",
    "name:walk_lrf_available", "column:walk_ferry_available",
    "name:drive_local_available", "name:drive_commuter_available",
    "name:drive_express_available", "name:drive_heavyrail_available",
    "name:drive_lrf_available", "column:drive_ferry_available",
)


def constants():
    return {
        "shortWalk": 0.333,
        "walkSpeed": 3.0,
        "min_waitTime": 0.0,
        "max_waitTime": 50.0,
        "Taxi_waitTime_mean": {1: 5.5, 2: 9.5, 3: 13.3, 4: 17.3, 5: 26.5},
        "Taxi_waitTime_sd": {1: 0.2, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.5},
        "TNC_single_waitTime_mean": {1: 3.0, 2: 6.3, 3: 8.4, 4: 8.5, 5: 10.3},
        "TNC_single_waitTime_sd": {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.9, 5: 1.1},
        "TNC_shared_waitTime_mean": {1: 5.0, 2: 8.0, 3: 11.0, 4: 15.0, 5: 15.0},
        "TNC_shared_waitTime_sd": {1: 0.2, 2: 0.5, 3: 0.9, 4: 1.4, 5: 1.8},
    }


def density_band(land, zones):
    measure = (
        land.TOTPOP.reindex(zones).to_numpy()
        + land.TOTEMP.reindex(zones).to_numpy()
    ) / (land.TOTACRE.reindex(zones).to_numpy() / 640.0)
    return np.asarray(pd.cut(
        measure, [-np.inf, 500, 2000, 5000, 15000, np.inf],
        labels=[5, 4, 3, 2, 1],
    ).astype(int))


def independent_waits(raw, origin, destination, c):
    draws = raw["standard_normal_draws"]
    origin_band = density_band(raw["land_use"], origin)
    destination_band = density_band(raw["land_use"], destination)
    outputs = []
    for family, prefix in enumerate(("Taxi", "TNC_single", "TNC_shared")):
        waits = []
        for side, bands in enumerate((origin_band, destination_band)):
            mean = np.asarray([c[f"{prefix}_waitTime_mean"][int(x)] for x in bands])
            sd = np.asarray([c[f"{prefix}_waitTime_sd"][int(x)] for x in bands])
            transform = 1.0 + sd * sd / (mean * mean)
            mu = np.log(mean / np.sqrt(transform))
            sigma = np.sqrt(np.log(transform))
            waits.append(np.exp(draws[:, family * 2 + side] * sigma + mu).clip(0, 50))
        outputs.append(waits[0] + waits[1])
    return outputs


def raw_scenario(seed, count=2000):
    rng = np.random.default_rng(seed)
    zones = np.arange(1, 31, dtype=np.int64)
    land = pd.DataFrame(
        {
            "TOTPOP": rng.integers(50, 50000, len(zones)),
            "TOTEMP": rng.integers(50, 80000, len(zones)),
            "TOTACRE": rng.uniform(50, 6000, len(zones)),
            "PRKCST": rng.uniform(0, 12, len(zones)),
            "area_type": rng.integers(1, 7, len(zones)),
            "TOPOLOGY": rng.integers(1, 5, len(zones)),
            "TERMINAL": rng.uniform(0, 20, len(zones)),
            "density_index": rng.uniform(0, 50, len(zones)),
        },
        index=pd.Index(zones, name="zone_id"),
    )
    ids = np.arange(seed * 10000, seed * 10000 + count, dtype=np.int64)
    origin = rng.choice(zones, count)
    destination = rng.choice(zones, count)
    purpose = "work" if seed % 2 else "school"
    destination_column = "workplace_zone_id" if purpose == "work" else "school_zone_id"
    tours = pd.DataFrame(
        {
            "home_zone_id": origin,
            destination_column: destination,
            "value_of_time": rng.uniform(0.25, 4.0, count),
            "tour_type": purpose,
            "tour_category": "mandatory",
            "number_of_participants": 1,
            "free_parking_at_work": rng.random(count) < 0.35,
            "auto_ownership": rng.integers(0, 5, count),
            "age": rng.integers(5, 90, count),
            "hhsize": rng.integers(1, 8, count),
            "num_workers": rng.integers(0, 5, count),
            "density_index": land.density_index.reindex(origin).to_numpy(),
        },
        index=pd.Index(ids, name="tour_id"),
    )
    raw = {
        "tours": tours,
        "land_use": land,
        "tour_purpose": purpose,
        "constants": constants(),
        "cbd_threshold": 2,
        "standard_normal_draws": rng.normal(size=(count, 6)),
    }
    return raw, ids, origin, destination


def qualify_raw(seed):
    from choiceforge.raw_table_input_generation import _owner_sources

    raw, ids, origin, destination = raw_scenario(seed)
    actual, actual_origin, actual_destination, actual_parking = _owner_sources(
        raw, ids, raw["constants"]
    )
    c = raw["constants"]
    tours = raw["tours"]
    land = raw["land_use"]
    waits = independent_waits(raw, origin, destination, c)
    expected = {
        "column:terminal_time": land.TERMINAL.reindex(destination).to_numpy(),
        "column:ivot": 1.0 / tours.value_of_time.to_numpy(),
        "column:density_index": tours.density_index.to_numpy(),
        "column:origin_walk_time": c["shortWalk"] * 60.0 / c["walkSpeed"],
        "column:destination_walk_time": c["shortWalk"] * 60.0 / c["walkSpeed"],
        "column:dest_density_index": land.density_index.reindex(destination).to_numpy(),
        "column:totalWaitTaxi": waits[0],
        "column:totalWaitSingleTNC": waits[1],
        "column:totalWaitSharedTNC": waits[2],
        "name:auto_ownership": tours.auto_ownership.to_numpy(),
        "name:age": tours.age.to_numpy(),
        "name:is_joint": np.zeros(len(tours), dtype=bool),
        "name:is_atwork_subtour": np.zeros(len(tours), dtype=bool),
        "name:work_tour_is_SOV": np.zeros(len(tours), dtype=bool),
        "name:number_of_participants": np.ones(len(tours), dtype=np.int64),
        "column:hhsize": tours.hhsize.to_numpy(),
        "column:dest_topology": land.TOPOLOGY.reindex(destination).to_numpy(),
        "name:work_tour_is_bike": np.zeros(len(tours), dtype=bool),
        "column:is_indiv": np.ones(len(tours), dtype=bool),
        "column:num_workers": tours.num_workers.to_numpy(),
        "column:destination_in_cbd": land.area_type.reindex(destination).to_numpy() < 2,
        "name:is_escort": np.zeros(len(tours), dtype=bool),
    }
    exact = np.array_equal(actual_origin, origin) and np.array_equal(actual_destination, destination)
    for key, value in expected.items():
        exact &= np.array_equal(np.asarray(actual[key]), np.asarray(value))
    free = tours.free_parking_at_work.to_numpy() if raw["tour_purpose"] == "work" else np.zeros(len(tours), bool)
    expected_parking = np.where(
        free, 0.0, land.PRKCST.reindex(destination).to_numpy()
    )
    exact &= np.array_equal(actual_parking, expected_parking)
    payload = b"".join(
        np.ascontiguousarray(np.asarray(actual[key])).tobytes() for key in sorted(actual)
    ) + np.ascontiguousarray(actual_parking).tobytes()
    return {
        "seed": seed,
        "raw_tours": len(ids),
        "exact": bool(exact),
        "source_columns_checked": len(expected) + 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def qualify_cuda(seed):
    import cupy as cp
    from choiceforge.device_input_expansion import ResidentSemanticInputPlan
    from choiceforge.semantic_input_generation import _availability_expression
    from qualify_phase28_semantic_scenarios import scenario

    required = []
    for label in LABELS18:
        _availability_expression(
            label, lambda source: required.append(source) or "value", 0
        )
    sources = tuple(dict.fromkeys(required))
    invocation, metadata, expected = scenario(cp, seed, sources, LABELS18)
    plan = ResidentSemanticInputPlan.compile(invocation, metadata)
    plan.execute()
    cp.cuda.Stream.null.synchronize()
    manifest = plan.semantic_program.manifest()
    return {
        "seed": seed,
        "rows": invocation.rows,
        "exact": bool(
            cp.array_equal(plan.invocation.float_inputs, invocation.float_inputs)
            and cp.array_equal(plan.invocation.int_inputs, invocation.int_inputs)
        ),
        "generated_availability_formulas": manifest["generated_int_columns"],
        "generated_sources": [
            item["source"] for item in manifest["expressions"]
            if item["source"] != "column:daily_parking_cost"
        ],
        "anonymous_response_pattern_columns": manifest["anonymous_response_pattern_columns"],
        "sha256": hashlib.sha256(expected).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (2901, 2902, 2903, 2904, 2905)
    raw_results = [qualify_raw(seed) for seed in seeds]
    cuda_results = [qualify_cuda(seed) for seed in seeds]
    report = {
        "phase": 29,
        "scope": (
            "five changed raw-table populations plus five changed raw-skim CUDA "
            "worlds, including all direct owner formulas and all 18 availability rules"
        ),
        "raw_table_scenarios": raw_results,
        "cuda_scenarios": cuda_results,
        "proof_gates": {
            "five_changed_raw_table_scenarios": len(raw_results) == 5,
            "ten_thousand_raw_tours": sum(x["raw_tours"] for x in raw_results) == 10000,
            "all_raw_table_formulas_exact": all(x["exact"] for x in raw_results),
            "five_changed_cuda_scenarios": len(cuda_results) == 5,
            "all_eighteen_availability_formulas_generated": (
                set(LABELS18) == {
                    source for result in cuda_results
                    for source in result["generated_sources"]
                }
            ),
            "all_cuda_rows_exact": all(x["exact"] for x in cuda_results),
            "zero_anonymous_patterns": all(
                x["anonymous_response_pattern_columns"] == 0 for x in cuda_results
            ),
            "all_outputs_scenario_responsive": (
                len({x["sha256"] for x in raw_results}) == 5
                and len({x["sha256"] for x in cuda_results}) == 5
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(report["proof_gates"].values()):
        raise SystemExit("Phase 29 changed-scenario qualification failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
