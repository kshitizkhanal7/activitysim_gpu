import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choiceforge.phase63_plans import ExpressionPlans


def build(directory, values, expression="x * 2 + 1", constant=3):
    import sharrow as sh
    tree = sh.DataTree(df=pd.DataFrame({"x":values}))
    tree.extra_vars = {"constant":constant}
    return tree.setup_flow({"value":expression}, cache_dir=str(directory))


def test_real_sharrow_plan_reuse_live_values_and_definitions(tmp_path):
    # First upstream build creates its module; second saves metadata; third hits.
    original = build(tmp_path,[1.,2.])
    expected = original.load()
    plans = ExpressionPlans()
    with plans.scope():
        np.testing.assert_array_equal(build(tmp_path,[1.,2.]).load(),expected)
        changed = build(tmp_path,[5.,6.,7.])
        np.testing.assert_array_equal(changed.load().reshape(-1),[11.,13.,15.])
        assert changed.defs == {"value":"x * 2 + 1"}
        different = build(tmp_path,[1.,2.],"x * 3 + 1")
        np.testing.assert_array_equal(different.load().reshape(-1),[4.,7.])
    assert plans.hits >= 1 and plans.misses >= 2


def test_constant_changes_and_corrupt_metadata_rebuild(tmp_path):
    build(tmp_path,[1.],"x + constant")
    plans = ExpressionPlans()
    with plans.scope():
        build(tmp_path,[1.],"x + constant")
        for path in (tmp_path/"choiceforge_phase63_plans").glob("*.json"):
            path.write_text('{"payload":{}}')
        np.testing.assert_array_equal(build(tmp_path,[2.],"x + constant").load(),[[5.]])
        np.testing.assert_array_equal(build(tmp_path,[2.],"x + constant",constant=9).load(),[[11.]])
    assert plans.rejected == 1


def test_scope_restored_on_exception():
    from sharrow.flows import Flow
    original = Flow.init_sub_funcs
    with pytest.raises(RuntimeError):
        with ExpressionPlans().scope():
            raise RuntimeError("deliberate")
    assert Flow.init_sub_funcs is original


def test_generated_source_mutation_invalidates_metadata(tmp_path):
    first = build(tmp_path,[1.])
    plans = ExpressionPlans()
    with plans.scope():
        build(tmp_path,[2.])
        source = tmp_path/("flow_"+first.flow_hash)/"__init__.py"
        source.write_bytes(source.read_bytes()+b"\n# benign source mutation\n")
        np.testing.assert_array_equal(build(tmp_path,[3.]).load(),[[7.]])
    assert plans.rejected==1 and plans.hits==0
