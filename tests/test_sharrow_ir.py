import csv
import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from choiceforge.activitysim_expression import parse_activitysim_expression
from choiceforge.sharrow_ir import (
    compare_strict_to_sharrow,
    evaluate_ir,
    evaluate_strict_cpu,
    expression_ir,
    ir_sha256,
    ordered_float32_utilities,
    specification_ir,
)


def test_expression_ir_preserves_skim_direction_and_operation_order():
    tree = expression_ir("@walktimelong_multiplier * (od_skims['DISTWALK'] - walkThresh).clip(lower=0) * 60/walkSpeed")
    assert tree["op"] == "div"
    assert tree["left"]["op"] == "mul"
    assert "od_skims" in str(tree)


def test_public_mtc_spec_generates_deterministic_hashable_ir():
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    spec = pd.read_csv(path, comment="#")
    document = specification_ir(spec)
    assert len(document["terms"]) == 379
    assert len(document["alternatives"]) == 21
    assert document["sha256"] == ir_sha256(document)


def test_strict_cpu_executes_every_public_mtc_term_and_alternative():
    """Canonical coverage gate: all 379 terms feed all 21 alternatives."""
    import numpy as np
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    spec = pd.read_csv(path, comment="#")
    document = specification_ir(spec)
    rows = 5
    values = lambda: np.full(rows, 2, dtype=np.int64)
    names = set()
    for term in document["terms"]:
        names.update(
            node.id
            for node in ast.walk(parse_activitysim_expression(term["expression"]))
            if isinstance(node, ast.Name)
        )
    mappings = {"df", "od_skims", "odt_skims", "dot_skims"}
    environment = {name: values() for name in names - mappings - {"np"}}
    environment.update({name: defaultdict(values) for name in mappings})
    symbols = {
        value["symbol"]: 1.0
        for term in document["terms"]
        for value in term["coefficients"].values()
        if isinstance(value, dict)
    }
    result = evaluate_strict_cpu(
        document, environment, coefficient_environment=symbols
    )
    assert result.features.shape == (rows, 379)
    assert result.utilities.shape == (rows, 21)
    assert np.isfinite(result.features).all()
    assert np.isfinite(result.utilities).all()


def test_ir_evaluator_runs_source_order_expression():
    import numpy as np
    tree = expression_ir("@x * (od_skims['DIST'] - threshold).clip(lower=0) / speed")
    env = {"x": 2.0, "threshold": 1.0, "speed": 2.0, "od_skims": {"DIST": np.array([1.0, 3.0])}, "df": {}}
    np.testing.assert_allclose(evaluate_ir(tree, env, np), [0.0, 2.0])


def test_ir_supports_activitysim_reverse_and_bidirectional_max_skims():
    import numpy as np

    reverse_tree = expression_ir("od_skims.reverse('DIST')")
    maximum_tree = expression_ir("od_skims.max('DIST')")
    env = {
        "od_skims": {"DIST": np.array([1.0, 5.0], dtype=np.float32)},
        "od_skims_reverse": {
            "DIST": np.array([3.0, 2.0], dtype=np.float32)
        },
    }
    np.testing.assert_array_equal(
        evaluate_ir(reverse_tree, env, np), np.array([3.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        evaluate_ir(maximum_tree, env, np), np.array([3.0, 5.0], dtype=np.float32)
    )


def test_ir_supports_round_trip_skim_names_used_by_tour_mode_choice():
    import numpy as np

    tree = expression_ir("odr_skims['BRIDGETOLL'] + dor_skims['BRIDGETOLL']")
    env = {
        "odr_skims": {"BRIDGETOLL": np.array([1.25], dtype=np.float32)},
        "dor_skims": {"BRIDGETOLL": np.array([2.50], dtype=np.float32)},
    }
    np.testing.assert_array_equal(
        evaluate_ir(tree, env, np), np.array([3.75], dtype=np.float32)
    )


def _strict_spec():
    return pd.DataFrame({
        "Label": ["scaled", "available", "constant"],
        "Expression": [
            "@scale * (od_skims['DIST'] - threshold).clip(lower=0)",
            "df.flag & (df.x > 0)",
            "@constant",
        ],
        "A": [0.25, -999.0, 1.0],
        "B": [-0.5, 0.0, 2.0],
    })


def _strict_environment(rows=2):
    import numpy as np
    return {
        "scale": 2.0,
        "threshold": 1.0,
        "constant": 0.125,
        "df": {
            "flag": np.array([True, False])[:rows],
            "x": np.array([1, -1])[:rows],
        },
        "od_skims": {
            "DIST": np.array([1.0, 3.0], dtype=np.float32)[:rows],
        },
    }


def test_strict_cpu_evaluator_enforces_declared_feature_and_utility_dtypes():
    import numpy as np
    result = evaluate_strict_cpu(specification_ir(_strict_spec()), _strict_environment())
    assert result.features.dtype == np.float32
    assert result.coefficients.dtype == np.float32
    assert result.utilities.dtype == np.float32
    np.testing.assert_array_equal(
        result.features,
        np.array([[0.0, 1.0, 0.125], [4.0, 0.0, 0.125]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.utilities,
        ordered_float32_utilities(result.features, result.coefficients),
    )


def test_strict_cpu_rejects_changed_policy_or_hash():
    import copy
    import pytest
    document = specification_ir(_strict_spec())
    changed = copy.deepcopy(document)
    changed["numeric_policy"]["allow_fastmath"] = True
    changed["sha256"] = ir_sha256(changed)
    with pytest.raises(ValueError, match="numeric policy"):
        evaluate_strict_cpu(changed, _strict_environment())
    changed = copy.deepcopy(document)
    changed["terms"][0]["label"] = "tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_strict_cpu(changed, _strict_environment())


def test_strict_cpu_resolves_symbols_and_fails_closed_when_missing():
    import numpy as np
    import pytest
    document = specification_ir(
        pd.DataFrame({"Expression": ["df.x"], "A": ["beta"]})
    )
    environment = {"df": {"x": np.array([2.0])}}
    result = evaluate_strict_cpu(
        document, environment, coefficient_environment={"beta": 3.0}
    )
    np.testing.assert_array_equal(result.utilities, [[6.0]])
    with pytest.raises(ValueError, match="unresolved coefficient"):
        evaluate_strict_cpu(document, environment)


def test_comparison_gate_splits_expression_and_accumulation_differences():
    import numpy as np
    strict = evaluate_strict_cpu(
        specification_ir(_strict_spec()), _strict_environment()
    )
    sharrow_features = strict.features.copy()
    sharrow_features[0, 0] = np.nextafter(
        sharrow_features[0, 0], np.float32(1)
    )
    sharrow_utilities = ordered_float32_utilities(
        sharrow_features, strict.coefficients
    )
    sharrow_utilities[1, 1] = np.nextafter(
        sharrow_utilities[1, 1], np.float32(np.inf)
    )
    report = compare_strict_to_sharrow(
        strict, sharrow_features, sharrow_utilities, row_labels=[101, 102]
    )
    assert not report["exact_gate_passed"]
    assert report["feature_comparison"]["divergent_terms"] == 1
    assert report["feature_comparison"]["first_divergence"]["row_label"] == 101
    assert report["classification"]["expression_policy_cells"] == 1
    assert report["classification"]["utility_cells_with_accumulation_policy_difference"] >= 1


def test_comparison_gate_passes_exact_strict_arrays():
    strict = evaluate_strict_cpu(
        specification_ir(_strict_spec()), _strict_environment(rows=1)
    )
    report = compare_strict_to_sharrow(
        strict, strict.features, strict.utilities
    )
    assert report["exact_gate_passed"]
    assert report["feature_comparison"]["divergent_terms"] == 0
    assert report["utility_comparison"]["divergent_alternatives"] == 0


def test_ordered_float32_accumulator_matches_independent_scalar_oracle():
    import numpy as np
    rng = np.random.default_rng(1301)
    features = rng.normal(size=(7, 11)).astype(np.float32)
    coefficients = rng.normal(size=(11, 4)).astype(np.float32)
    expected = np.zeros((7, 4), dtype=np.float32)
    for row in range(7):
        for alternative in range(4):
            value = np.float32(0)
            for term in range(11):
                product = np.multiply(
                    features[row, term], coefficients[term, alternative],
                    dtype=np.float32,
                )
                value = np.add(value, product, dtype=np.float32)
            expected[row, alternative] = value
    np.testing.assert_array_equal(
        ordered_float32_utilities(features, coefficients), expected
    )
