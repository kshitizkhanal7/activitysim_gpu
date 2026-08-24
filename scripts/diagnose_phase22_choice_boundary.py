"""Compare arithmetic policies for the Phase 22 near-boundary work tour."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numba

from choiceforge.gpu_scheduling_pipeline import CompiledCpuSchedulingPreparer
from choiceforge.scheduling_compiler import compile_python_expression


@numba.njit(fastmath=True)
def sharrow_style_dot(features, coefficients):
    result = np.empty(features.shape[0], dtype=np.float32)
    for row in range(features.shape[0]):
        result[row] = np.dot(features[row], coefficients)
    return result


@numba.njit(fastmath=False)
def sharrow_exact_dot(features, coefficients):
    result = np.empty(features.shape[0], dtype=np.float32)
    for row in range(features.shape[0]):
        result[row] = np.dot(features[row], coefficients)
    return result


def period_code(values):
    values = np.asarray(values)
    return np.where(
        values <= 5,
        0,
        np.where(values <= 9, 1, np.where(values <= 14, 2, np.where(values <= 18, 3, 4))),
    )


def select(utilities, draw, probability_dtype, *, overflow_protection=True):
    dtype = np.dtype(probability_dtype).type
    values = np.asarray(utilities, dtype=dtype)
    weights = np.exp(
        values - values.max() if overflow_protection else values
    ).astype(dtype)
    weights /= weights.sum(dtype=dtype)
    remainder = np.float64(draw)
    for position, probability in enumerate(weights):
        remainder -= np.float64(probability)
        if remainder <= 0:
            return position, remainder, weights
    return len(weights) - 1, remainder, weights


def lane_dot(features, coefficients, lanes, *, tree=False):
    products = np.asarray(features, dtype=np.float32) * np.asarray(
        coefficients, dtype=np.float32
    )
    accumulators = np.zeros((products.shape[0], lanes), dtype=np.float32)
    for column in range(products.shape[1]):
        lane = column % lanes
        accumulators[:, lane] = np.float32(
            accumulators[:, lane] + products[:, column]
        )
    if tree:
        while accumulators.shape[1] > 1:
            if accumulators.shape[1] % 2:
                accumulators = np.column_stack(
                    (accumulators, np.zeros(products.shape[0], dtype=np.float32))
                )
            accumulators = np.float32(
                accumulators[:, 0::2] + accumulators[:, 1::2]
            )
        return accumulators[:, 0]
    result = np.zeros(products.shape[0], dtype=np.float32)
    for lane in range(lanes):
        result = np.float32(result + accumulators[:, lane])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", type=Path, default=Path("benchmark-results/phase21-scheduling-inputs")
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("benchmark-results/phase22-integrated-diagnostic-capture/batch_000.npz"),
    )
    parser.add_argument("--chooser-id", type=int, default=13282973)
    parser.add_argument(
        "--scheduling-capture",
        type=Path,
        default=Path("benchmark-results/p22dc3/scheduling_boundary.npz"),
    )
    args = parser.parse_args()

    manifest = json.loads((args.inputs / "manifest.json").read_text())
    common = np.load(args.inputs / manifest["common_file"])
    batch = np.load(args.inputs / "batch_000.npz")
    raw = np.load(args.capture)
    metadata = manifest["batches"][0]
    ids = raw["chooser_ids"]
    first = np.r_[True, ids[1:] != ids[:-1]]
    owners = np.cumsum(first, dtype=np.int32) - 1
    slots = period_code(raw["start"]) * 5 + period_code(raw["end"])
    cache = np.zeros((batch["chooser_ids"].size, 25), dtype=np.float32)
    cache[owners, slots] = raw["logsums"].astype(np.float32)

    preparer = CompiledCpuSchedulingPreparer(
        manifest["person_count"], common["alternative_values"]
    )
    names = metadata["chooser_columns"]
    prepared = preparer.prepare(
        batch["person_rows"],
        batch["chooser_values"],
        cache,
        end_previous_column=names.index("end_previous"),
        tour_count_column=names.index("tour_count"),
        tour_num_column=names.index("tour_num"),
    )
    chooser = int(np.flatnonzero(batch["chooser_ids"] == args.chooser_id)[0])
    begin, end = prepared.offsets[chooser : chooser + 2]
    alternative_ids = prepared.alternative_ids[begin:end]
    expressions = [compile_python_expression(x) for x in metadata["expressions"]]
    features64 = np.empty((end - begin, len(expressions)), dtype=np.float64)
    for local_row, row in enumerate(range(begin, end)):
        environment = {
            name: np.float64(
                27.8
                if name == "income_in_thousands"
                else prepared.chooser_values[chooser, column]
            )
            for column, name in enumerate(metadata["chooser_columns"])
        }
        environment.update(
            {
                name: np.float64(prepared.row_values[row, column])
                for column, name in enumerate(metadata["row_columns"])
            }
        )
        environment.update(
            {
                name: np.float64(
                    common["alternative_values"][alternative_ids[local_row], column]
                )
                for column, name in enumerate(metadata["alternative_columns"])
            }
        )
        features64[local_row] = [
            eval(expression, {"__builtins__": {}}, environment)
            for expression in expressions
        ]

    coefficients32 = batch["coefficients"].astype(np.float32)
    policies = {
        "float32_dot": np.dot(features64.astype(np.float32), coefficients32),
        "numba_fastmath_float32_dot": sharrow_style_dot(
            features64.astype(np.float32), coefficients32
        ),
        "numba_exact_float32_dot": sharrow_exact_dot(
            features64.astype(np.float32), coefficients32
        ),
        "float64_dot_from_float32": np.dot(
            features64.astype(np.float32).astype(np.float64),
            coefficients32.astype(np.float64),
        ),
        "float64_dot": np.dot(features64, coefficients32.astype(np.float64)),
    }
    sharrow_features = np.insert(features64.astype(np.float32), (27, 27), 0.0, axis=1)
    sharrow_coefficients = np.insert(coefficients32, (27, 27), 0.0)
    for lanes in (2, 4, 8, 16):
        policies[f"lane{lanes}_sequential"] = lane_dot(
            sharrow_features, sharrow_coefficients, lanes
        )
        policies[f"lane{lanes}_tree"] = lane_dot(
            sharrow_features, sharrow_coefficients, lanes, tree=True
        )
    draw = float(batch["draws"][chooser])
    expected = int(batch["expected_tdd"][chooser])
    output = {"chooser_id": args.chooser_id, "draw": draw, "expected_tdd": expected}
    if args.scheduling_capture.exists():
        sharrow_utility = np.load(args.scheduling_capture)["utility"]
        output["sharrow_utility_dtype"] = str(sharrow_utility.dtype)
        output["utility_comparisons"] = {
            name: {
                "bit_mismatches": int(
                    np.count_nonzero(
                        np.asarray(values, dtype=np.float32).view(np.uint32)
                        != sharrow_utility.view(np.uint32)
                    )
                ),
                "max_abs_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(values, dtype=np.float32) - sharrow_utility
                        )
                    )
                ),
                "tdd_168_difference": float(
                    np.asarray(values, dtype=np.float32)[
                        int(np.flatnonzero(alternative_ids == 168)[0])
                    ]
                    - sharrow_utility[
                        int(np.flatnonzero(alternative_ids == 168)[0])
                    ]
                ),
            }
            for name, values in policies.items()
        }
    for utility_name, utilities in policies.items():
        for probability_dtype in (np.float32, np.float64):
            for overflow_protection in (True, False):
                position, remainder, probabilities = select(
                    utilities,
                    draw,
                    probability_dtype,
                    overflow_protection=overflow_protection,
                )
                selected = int(alternative_ids[position])
                expected_position = int(np.flatnonzero(alternative_ids == expected)[0])
                prior_position = max(0, expected_position - 1)
                output[
                    f"{utility_name}_{np.dtype(probability_dtype).name}_"
                    f"{'shifted' if overflow_protection else 'unshifted'}"
                ] = {
                    "selected_tdd": selected,
                    "remainder": float(remainder),
                    "cumulative_before_expected": float(
                        probabilities[: prior_position + 1].sum(dtype=np.float64)
                    ),
                    "cumulative_through_expected": float(
                        probabilities[: expected_position + 1].sum(dtype=np.float64)
                    ),
                }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
