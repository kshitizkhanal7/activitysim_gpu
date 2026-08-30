"""Identify the float32 reduction schedule behind Sharrow's 15-by-1 dot.

This is a deliberately small, deterministic probe.  Sharrow emits
``np.dot(features, coefficients, out=one_element_array)`` inside a Numba
function.  Numba lowers that shape to BLAS SGEMV, so a CUDA compatibility
kernel must reproduce the SGEMV reduction schedule rather than assume a
generic sequential dot product.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=False)
def blas_reference(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    result = np.empty(features.shape[0], dtype=np.float32)
    coefficient_column = coefficients.reshape(15, 1)
    for row in range(features.shape[0]):
        output = np.empty(1, dtype=np.float32)
        np.dot(features[row], coefficient_column, out=output)
        result[row] = output[0]
    return result


@nb.njit(cache=True, fastmath=False)
def candidate_reductions(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    result = np.empty((features.shape[0], 6), dtype=np.float32)
    for row in range(features.shape[0]):
        # 0: ordinary source-order accumulation.
        sequential = np.float32(0.0)
        for position in range(15):
            sequential = np.float32(
                sequential
                + np.float32(features[row, position] * coefficients[position])
            )
        result[row, 0] = sequential

        # OpenBLAS' m=1 SGEMV fallback consumes four terms at a time.  These
        # variants make the possible compiler association explicit.
        group_left = np.float32(0.0)
        group_tree = np.float32(0.0)
        four_lanes = np.zeros(4, dtype=np.float32)
        for base in range(0, 12, 4):
            p0 = np.float32(features[row, base] * coefficients[base])
            p1 = np.float32(features[row, base + 1] * coefficients[base + 1])
            p2 = np.float32(features[row, base + 2] * coefficients[base + 2])
            p3 = np.float32(features[row, base + 3] * coefficients[base + 3])
            left = np.float32(np.float32(np.float32(p0 + p1) + p2) + p3)
            tree = np.float32(np.float32(p0 + p1) + np.float32(p2 + p3))
            group_left = np.float32(group_left + left)
            group_tree = np.float32(group_tree + tree)
            four_lanes[0] = np.float32(four_lanes[0] + p0)
            four_lanes[1] = np.float32(four_lanes[1] + p1)
            four_lanes[2] = np.float32(four_lanes[2] + p2)
            four_lanes[3] = np.float32(four_lanes[3] + p3)
        for position in range(12, 15):
            product = np.float32(features[row, position] * coefficients[position])
            group_left = np.float32(group_left + product)
            group_tree = np.float32(group_tree + product)
            four_lanes[position % 4] = np.float32(
                four_lanes[position % 4] + product
            )
        result[row, 1] = group_left
        result[row, 2] = group_tree
        result[row, 3] = np.float32(
            np.float32(four_lanes[0] + four_lanes[1])
            + np.float32(four_lanes[2] + four_lanes[3])
        )
        result[row, 4] = np.float32(
            np.float32(
                np.float32(four_lanes[0] + four_lanes[1]) + four_lanes[2]
            )
            + four_lanes[3]
        )

        # The literal OpenBLAS fallback expression as parsed by C:
        # temp += p0 + p1 + p2 + p3, followed by scalar tail terms.
        literal = np.float32(0.0)
        for base in range(0, 12, 4):
            literal = np.float32(
                literal
                + np.float32(
                    np.float32(
                        np.float32(
                            features[row, base] * coefficients[base]
                            + features[row, base + 1] * coefficients[base + 1]
                        )
                        + features[row, base + 2] * coefficients[base + 2]
                    )
                    + features[row, base + 3] * coefficients[base + 3]
                )
            )
        for position in range(12, 15):
            literal = np.float32(
                literal + features[row, position] * coefficients[position]
            )
        result[row, 5] = literal
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(410031)
    features = rng.normal(0.0, 40.0, size=(200_000, 15)).astype(np.float32)
    coefficients = rng.normal(0.0, 8.0, size=15).astype(np.float32)
    expected = blas_reference(features, coefficients)
    candidates = candidate_reductions(features, coefficients)
    names = (
        "sequential",
        "group4_left",
        "group4_tree",
        "lane4_tree",
        "lane4_left",
        "openblas_literal",
    )
    comparisons = {}
    expected_bits = expected.view(np.uint32)
    for column, name in enumerate(names):
        values = candidates[:, column]
        comparisons[name] = {
            "bit_mismatches": int(np.count_nonzero(values.view(np.uint32) != expected_bits)),
            "max_abs_difference": float(np.max(np.abs(values - expected))),
        }
    document = {"rows": len(features), "comparisons": comparisons}
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
