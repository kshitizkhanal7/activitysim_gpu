import csv
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from choiceforge.activitysim_expression import (
    ExpressionUnsupported, evaluate_activitysim_expression, lower_activitysim_utility_spec,
    parse_activitysim_expression,
)


def _environment(xp=np):
    return {
        "df": {"age": xp.asarray([9, 16, 19, 22]), "ivot": xp.asarray([1.0, 1.2, 0.8, 1.1]), "density_index": xp.asarray([2, 3, 4, 5])},
        "od_skims": {"DISTWALK": xp.asarray([1.0, 2.5, 3.0, 0.5])},
        "odt_skims": {"SOV_TIME": xp.asarray([10.0, 20.0, 30.0, 40.0])},
        "walkThresh": 2.0, "walkSpeed": 3.0, "coef_walktimeshort_multiplier": 1.5,
        "TRANSIT_SCALE_FACTOR": 1.0,
    }


def test_real_mtc_expression_subset_parses_completely():
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    with path.open(newline="", encoding="utf8") as stream:
        expressions = {row["Expression"].strip() for row in csv.DictReader(stream) if row["Expression"].strip()}
    assert len(expressions) == 253
    for expression in expressions:
        parse_activitysim_expression(expression)


def test_every_real_mtc_expression_evaluates_in_reviewed_subset():
    """Exercise every source expression with generic nonzero numeric columns.

    This is a syntax/operation-coverage gate, not a behavioral substitute for
    captured ActivitySim rows (which are checked before production enablement).
    """
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    with path.open(newline="", encoding="utf8") as stream:
        expressions = {row["Expression"].strip() for row in csv.DictReader(stream) if row["Expression"].strip()}
    rows = 5
    values = lambda: np.full(rows, 2, dtype=np.int64)
    names = set()
    for expression in expressions:
        names.update(node.id for node in ast.walk(parse_activitysim_expression(expression)) if isinstance(node, ast.Name))
    mappings = {"df", "od_skims", "odt_skims", "dot_skims"}
    env = {name: values() for name in names - mappings - {"np"}}
    env.update({name: defaultdict(values) for name in mappings})
    for expression in expressions:
        result = evaluate_activitysim_expression(expression, env, np)
        assert np.asarray(result).shape in {(), (rows,)}


def test_every_real_mtc_expression_runs_on_gpu():
    cp = pytest.importorskip("cupy")
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    with path.open(newline="", encoding="utf8") as stream:
        expressions = {row["Expression"].strip() for row in csv.DictReader(stream) if row["Expression"].strip()}
    rows = 5
    values = lambda: cp.full(rows, 2, dtype=cp.int64)
    names = set()
    for expression in expressions:
        names.update(node.id for node in ast.walk(parse_activitysim_expression(expression)) if isinstance(node, ast.Name))
    mappings = {"df", "od_skims", "odt_skims", "dot_skims"}
    env = {name: values() for name in names - mappings - {"np"}}
    env.update({name: defaultdict(values) for name in mappings})
    for expression in expressions:
        result = evaluate_activitysim_expression(expression, env, cp)
        assert cp.asarray(result).shape in {(), (rows,)}


def test_cpu_expression_semantics_cover_skim_clip_and_availability():
    env = _environment()
    np.testing.assert_allclose(
        evaluate_activitysim_expression("@coef_walktimeshort_multiplier * od_skims['DISTWALK'].clip(upper=walkThresh) * 60/walkSpeed", env, np),
        [30.0, 60.0, 60.0, 15.0],
    )
    np.testing.assert_array_equal(
        evaluate_activitysim_expression("@(df.age >= 16) & (df.age <= 19)", env, np), [False, True, True, False],
    )


def test_calibrated_household_clip_and_where_semantics():
    env = {
        "df": {
            "income": np.asarray([-2.0, 10.0, 50.0]),
            "workers": np.asarray([0, 1, 2]),
            "ratio": np.asarray([np.nan, 8.0, 18.0]),
        }
    }
    np.testing.assert_array_equal(
        evaluate_activitysim_expression("df.income.clip(0, 30)", env, np),
        [0.0, 10.0, 30.0],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = evaluate_activitysim_expression(
            "np.where(df.workers > 0, df.ratio / df.workers, 0)", env, np
        )
    np.testing.assert_array_equal(result, [0.0, 8.0, 9.0])


def test_gpu_expression_matches_cpu_for_real_expression():
    cp = pytest.importorskip("cupy")
    expression = "@coef_walktimeshort_multiplier * od_skims['DISTWALK'].clip(upper=walkThresh) * 60/walkSpeed"
    actual = evaluate_activitysim_expression(expression, _environment(cp), cp)
    np.testing.assert_allclose(cp.asnumpy(actual), evaluate_activitysim_expression(expression, _environment(), np))


def test_unknown_syntax_fails_closed():
    with pytest.raises(ExpressionUnsupported, match="only np.maximum"):
        evaluate_activitysim_expression("np.exp(df.age)", _environment(), np)


def test_lowering_resolved_spec_matches_direct_cpu_utility():
    pd = pytest.importorskip("pandas")
    spec = pd.DataFrame({
        "Label": ["walk_distance", "sov_time", "age_16p"],
        "Expression": ["od_skims['DISTWALK']", "odt_skims['SOV_TIME']", "@(df.age >= 16)"],
        "WALK": [-0.3, 0.0, 0.1], "DRIVE": [0.0, -0.04, 0.2],
    })
    model, features = lower_activitysim_utility_spec(spec, _environment(), np)
    expected = np.column_stack((
        -0.3 * _environment()["od_skims"]["DISTWALK"] + 0.1 * (_environment()["df"]["age"] >= 16),
        -0.04 * _environment()["odt_skims"]["SOV_TIME"] + 0.2 * (_environment()["df"]["age"] >= 16),
    ))
    np.testing.assert_allclose(model.cpu_reference(features), expected)
