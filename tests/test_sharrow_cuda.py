from collections import defaultdict
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from choiceforge.activitysim_expression import parse_activitysim_expression
from choiceforge.sharrow_cuda import (
    InputBinding,
    _bindings,
    clear_strict_cuda_cache,
    compare_strict_cpu_cuda,
    evaluate_strict_cuda,
    generate_cuda_source,
    mtc21_logsums_from_strict_ir_cuda,
)
from choiceforge.sharrow_ir import evaluate_strict_cpu, specification_ir
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from choiceforge.cuda_skims import CudaDatasetSkimBinding
from test_nested_logit import NEST


def _spec():
    return pd.DataFrame({
        "Label": ["arithmetic", "mask", "range", "maximum", "integer"],
        "Expression": [
            "@scale * (od_skims['DIST'] - threshold).clip(lower=0, upper=5) / denom",
            "df.flag & ~df.other",
            "@lower <= df.x <= upper",
            "np.maximum(odt_skims['TIME'], dot_skims['TIME'])",
            "df.count",
        ],
        "A": [0.25, -999.0, 1.5, -0.03, 2.0],
        "B": [-0.5, 0.0, -2.0, 0.07, -0.25],
    })


def _environment():
    return {
        "scale": 2.0,
        "threshold": 1.0,
        "denom": np.array([2.0, 4.0, 8.0, 3.0]),
        "lower": -1.0,
        "upper": 2.0,
        "df": {
            "flag": np.array([True, True, False, False]),
            "other": np.array([False, True, False, True]),
            "x": np.array([-1.0, 0.5, 3.0, 2.0]),
            "count": np.array([1, 2, 3, 4], dtype=np.int64),
        },
        "od_skims": {"DIST": np.array([0.5, 3.0, 9.0, 4.0], dtype=np.float32)},
        "odt_skims": {"TIME": np.array([10.0, 40.0, 30.0, 8.0], dtype=np.float32)},
        "dot_skims": {"TIME": np.array([12.0, 20.0, 35.0, 8.0], dtype=np.float64)},
    }


def test_generated_cuda_matches_strict_cpu_for_every_supported_operation():
    pytest.importorskip("cupy")
    document = specification_ir(_spec())
    cpu = evaluate_strict_cpu(document, _environment())
    cuda = evaluate_strict_cuda(document, _environment())
    report = compare_strict_cpu_cuda(cpu, cuda)
    assert report["exact_gate_passed"]
    np.testing.assert_array_equal(cuda.features, cpu.features)
    np.testing.assert_array_equal(cuda.utilities, cpu.utilities)


def test_generated_cuda_preserves_nonfinite_edge_cases_exactly():
    pytest.importorskip("cupy")
    spec = pd.DataFrame({
        "Expression": ["df.x / df.y", "np.maximum(df.x, df.z)", "df.subnormal"],
        "A": [1.0, 0.5, 1.0],
    })
    environment = {
        "df": {
            "x": np.array([1.0, 0.0, np.nan, np.inf]),
            "y": np.array([0.0, 0.0, 1.0, np.inf]),
            "z": np.array([2.0, np.nan, 3.0, -np.inf]),
            "subnormal": np.full(
                4, np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32
            ),
        }
    }
    document = specification_ir(spec)
    cpu = evaluate_strict_cpu(
        document, environment, expression_dtype="float32"
    )
    cuda = evaluate_strict_cuda(
        document, environment, sparse_zero_coefficients=True
    )
    assert compare_strict_cpu_cuda(cpu, cuda)["exact_gate_passed"]
    assert cuda.telemetry.sparse_zero_coefficients
    assert cuda.telemetry.zero_coefficient_ops_skipped_per_row == 0


def test_float32_expression_policy_compiles_and_reports_explicitly():
    pytest.importorskip("cupy")
    spec = pd.DataFrame({
        "Expression": ["df.x * scale + df.y", "df.x <= df.y"],
        "A": [0.5, 2.0],
        "B": [-0.25, 0.0],
    })
    environment = {
        "df": {
            "x": np.array([1, 2, 3, 4], dtype=np.float32),
            "y": np.array([8, 6, 4, 2], dtype=np.float32),
        },
        "scale": np.float32(2.0),
    }
    document = specification_ir(spec)
    cpu = evaluate_strict_cpu(document, environment)
    cuda = evaluate_strict_cuda(
        document, environment, expression_float32=True
    )
    assert compare_strict_cpu_cuda(cpu, cuda)["exact_gate_passed"]
    assert cuda.telemetry.expression_dtype == "float32"
    assert cuda.telemetry.input_bytes < evaluate_strict_cuda(
        document, environment
    ).telemetry.input_bytes


def test_float32_expression_policy_has_a_separate_cpu_oracle():
    pytest.importorskip("cupy")
    spec = pd.DataFrame({
        "Expression": ["(df.x + 1.0) - df.x"],
        "A": [1.0],
    })
    environment = {"df": {"x": np.array([1.0e8], dtype=np.float64)}}
    document = specification_ir(spec)
    strict64 = evaluate_strict_cpu(document, environment)
    reference32 = evaluate_strict_cpu(
        document, environment, expression_dtype="float32"
    )
    cuda32 = evaluate_strict_cuda(
        document, environment, expression_float32=True
    )
    assert strict64.features[0, 0] == 1.0
    assert reference32.features[0, 0] == 0.0
    assert compare_strict_cpu_cuda(reference32, cuda32)["exact_gate_passed"]


def test_strict_cuda_cache_reuses_ir_and_typed_schema():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(_spec())
    first = evaluate_strict_cuda(document, _environment())
    second = evaluate_strict_cuda(document, _environment())
    assert first.telemetry.compiled_this_call
    assert not second.telemetry.compiled_this_call
    assert first.telemetry.cache_key == second.telemetry.cache_key
    assert first.telemetry.cache_key.startswith(document["sha256"])


def test_strict_cuda_reuses_device_coefficients_and_reports_split_transfer_timing():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(_spec())
    first = evaluate_strict_cuda(document, _environment())
    second = evaluate_strict_cuda(document, _environment())
    assert not first.telemetry.coefficient_cache_hit
    assert second.telemetry.coefficient_cache_hit
    assert second.telemetry.coefficient_upload_ms == 0
    assert second.telemetry.host_to_device_ms == second.telemetry.input_upload_ms
    assert second.telemetry.host_pack_ms >= 0


def test_persistent_plan_reuses_compiled_schema_and_reports_build_cost_once():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(_spec())
    first = evaluate_strict_cuda(
        document, _environment(), persistent_plan=True, compact_inputs=True
    )
    second = evaluate_strict_cuda(
        document, _environment(), persistent_plan=True, compact_inputs=True
    )
    assert first.telemetry.persistent_plan
    assert not first.telemetry.plan_cache_hit
    assert first.telemetry.plan_build_ms > 0
    assert second.telemetry.plan_cache_hit
    assert second.telemetry.plan_build_ms == 0
    assert first.telemetry.cache_key == second.telemetry.cache_key


def test_persistent_plan_refuses_changed_compact_alias_layout():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(pd.DataFrame({
        "Expression": ["df.x + df.y"],
        "A": [1.0],
    }))
    shared = np.arange(8, dtype=np.float32)
    shared_environment = {"df": {"x": shared, "y": shared}}
    split_environment = {
        "df": {
            "x": np.arange(8, dtype=np.float32),
            "y": np.arange(8, dtype=np.float32) + 10,
        }
    }
    first = evaluate_strict_cuda(
        document, shared_environment, compact_inputs=True, persistent_plan=True
    )
    second = evaluate_strict_cuda(
        document, split_environment, compact_inputs=True, persistent_plan=True
    )
    third = evaluate_strict_cuda(
        document, split_environment, compact_inputs=True, persistent_plan=True
    )
    assert not first.telemetry.plan_cache_hit
    assert not second.telemetry.plan_cache_hit
    assert third.telemetry.plan_cache_hit
    assert first.telemetry.dense_row_inputs == 1
    assert second.telemetry.dense_row_inputs == 2
    expected = evaluate_strict_cpu(document, split_environment)
    assert compare_strict_cpu_cuda(expected, third)["exact_gate_passed"]


def test_persistent_plan_uses_stable_slots_when_scalar_values_change():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(pd.DataFrame({
        "Expression": ["df.x * scale + offset"],
        "A": [1.0],
    }))
    first_environment = {
        "df": {"x": np.arange(8, dtype=np.float32)},
        "scale": 2.0,
        "offset": 2.0,
    }
    second_environment = {
        "df": {"x": np.arange(8, dtype=np.float32)},
        "scale": 3.0,
        "offset": 7.0,
    }
    first = evaluate_strict_cuda(
        document, first_environment, compact_inputs=True, persistent_plan=True
    )
    second = evaluate_strict_cuda(
        document, second_environment, compact_inputs=True, persistent_plan=True
    )
    assert not first.telemetry.plan_cache_hit
    assert second.telemetry.plan_cache_hit
    assert first.telemetry.scalar_inputs == second.telemetry.scalar_inputs == 2
    expected = evaluate_strict_cpu(document, second_environment)
    assert compare_strict_cpu_cuda(expected, second)["exact_gate_passed"]


def test_persistent_plan_reuses_device_workspace_without_changing_results():
    pytest.importorskip("cupy")
    clear_strict_cuda_cache()
    document = specification_ir(_spec())
    first = evaluate_strict_cuda(
        document,
        _environment(),
        compact_inputs=True,
        persistent_plan=True,
        reuse_buffers=True,
    )
    expected_first = first.utilities.copy()
    second = evaluate_strict_cuda(
        document,
        _environment(),
        compact_inputs=True,
        persistent_plan=True,
        reuse_buffers=True,
    )
    assert first.telemetry.reusable_workspace
    assert not first.telemetry.workspace_cache_hit
    assert second.telemetry.workspace_cache_hit
    np.testing.assert_array_equal(expected_first, second.utilities)
    expected = evaluate_strict_cpu(document, _environment())
    assert compare_strict_cpu_cuda(expected, second)["exact_gate_passed"]


def test_canonical_generator_parallelizes_features_without_reordering_utilities():
    path = Path("benchmark-data/phase9-mtc-full/prototype_mtc_extended/configs/trip_mode_choice.csv")
    spec = pd.read_csv(path, comment="#")
    document = specification_ir(spec)
    rows = 2
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
    bindings, _ = _bindings(document, environment)
    source, _ = generate_cuda_source(document, bindings, capture_features=False)
    assert "if (threadIdx.x < 256)" in source
    assert "case 255:" in source
    assert "#pragma unroll 1" in source
    assert "__fmul_rn" in source and "__fadd_rn" in source
    fused_source, _ = generate_cuda_source(
        document,
        bindings,
        capture_features=False,
        fused_utility_accumulation=True,
    )
    assert "fmaf(shared_features[term]" in fused_source
    assert "__fmul_rn(shared_features[term]" not in fused_source


def test_direct_generator_can_fuse_row_sources_and_skim_coordinates():
    document = specification_ir(pd.DataFrame({
        "Expression": ["df.x + od_skims['DIST']"],
        "A": [1.0],
    }))
    bindings = [
        InputBinding(("column", "x"), "float", "float64", 0),
        InputBinding(("skim", "od_skims", "DIST"), "float", "skim", 0, 2, 0),
    ]
    source, _ = generate_cuda_source(
        document,
        bindings,
        capture_features=False,
        group_skim_indices=True,
        row_source_references={("column", "x"): "phase37_x"},
        group_coordinate_references={0: ("phase37_origin", "phase37_destination", None)},
        extra_kernel_parameters=("    const float* phase37_packet",),
        row_prelude=(
            "    const float phase37_x = phase37_packet[row];\n"
            "    const long long phase37_origin = row;\n"
            "    const long long phase37_destination = row + 1;"
        ),
    )
    assert "const float* phase37_packet" in source
    assert "const float phase37_x = phase37_packet[row]" in source
    assert "(phase37_origin * skim_group_0_dest_count + phase37_destination)" in source
    assert "float_inputs[row *" not in source
    assert "skim_group_0_orig[row]" not in source
    assert "skim_group_0_dest[row]" not in source

    block_source, _ = generate_cuda_source(
        document,
        bindings,
        capture_features=False,
        group_skim_indices=True,
        row_source_references={("column", "x"): "phase51_x"},
        group_coordinate_references={0: ("phase51_origin", "phase51_destination", None)},
        extra_kernel_parameters=("    const float* phase51_packet",),
        block_prelude="    __shared__ float phase51_values[1];",
        row_prelude=(
            "    const float phase51_x = phase51_packet[row];\n"
            "    const long long phase51_origin = row;\n"
            "    const long long phase51_destination = row;"
        ),
    )
    assert "const float phase51_x = phase51_packet[row]" in block_source
    assert "__shared__ float phase51_values[1]" in block_source

    with pytest.raises(ValueError, match="unknown bindings"):
        generate_cuda_source(
            document,
            bindings,
            row_source_references={("column", "missing"): "missing"},
        )
    tiled_source, _ = generate_cuda_source(
        document,
        bindings,
        locality_tile_rows=2,
        group_skim_indices=True,
        row_source_references={("column", "x"): "phase52_x[tile_row]"},
        group_coordinate_references={
            0: ("phase52_origin[gather_row]", "phase52_destination[gather_row]", None)
        },
        block_prelude=(
            "    __shared__ float phase52_x[2];\n"
            "    __shared__ long long phase52_origin[2];\n"
            "    __shared__ long long phase52_destination[2];"
        ),
        row_prelude="    phase52_x[tile_row] = phase51_packet[row];",
        extra_kernel_parameters=("    const float* phase51_packet",),
    )
    assert "constexpr int TILE_ROWS = 2" in tiled_source
    assert "phase52_x[tile_row] = phase51_packet[row]" in tiled_source
    assert "phase52_origin[gather_row]" in tiled_source
    assert "float_inputs[row *" not in tiled_source
    with pytest.raises(ValueError, match="unknown skim groups"):
        generate_cuda_source(
            document,
            bindings,
            group_skim_indices=True,
            group_coordinate_references={7: ("origin", "destination", None)},
        )

def test_generated_cuda_matches_strict_cpu_for_canonical_mtc_ir():
    pytest.importorskip("cupy")
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
    cpu = evaluate_strict_cpu(
        document, environment, coefficient_environment=symbols
    )
    cuda = evaluate_strict_cuda(
        document,
        environment,
        coefficient_environment=symbols,
        sparse_zero_coefficients=True,
    )
    report = compare_strict_cpu_cuda(cpu, cuda)
    assert report["exact_gate_passed"]
    assert report["terms"] == 379
    assert report["alternatives"] == 21
    assert cuda.telemetry.zero_coefficient_ops_skipped_per_row > 0


def test_generated_strict_utilities_stay_on_device_through_nested_logsum():
    pytest.importorskip("cupy")
    alternatives = list(MTC21_ALTERNATIVES)
    rng = np.random.default_rng(1402)
    spec = pd.DataFrame({
        "Expression": ["df.x", "df.y", "df.x * df.y"],
        **{
            alternative: rng.normal(size=3)
            for alternative in alternatives
        },
    })
    environment = {
        "df": {
            "x": rng.normal(size=257),
            "y": rng.normal(size=257),
        }
    }
    document = specification_ir(spec)
    cpu = evaluate_strict_cpu(document, environment)
    from choiceforge.nested_logit import mtc21_nested_logsums_cuda

    expected = mtc21_nested_logsums_cuda(
        cpu.utilities, NEST, alternatives
    )
    actual, telemetry = mtc21_logsums_from_strict_ir_cuda(
        document, environment, NEST, return_telemetry=True
    )
    np.testing.assert_array_equal(actual, expected)
    assert telemetry.utility.device_to_host_ms == 0
    assert telemetry.nested_logsum.host_to_device_ms == 0


def test_resident_invocation_snapshots_dense_inputs_and_skim_coordinates():
    cp = pytest.importorskip("cupy")
    rows = 7
    cube = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    orig = np.arange(rows, dtype=np.int64) % 4
    dest = (np.arange(rows, dtype=np.int64) * 2) % 5
    period = np.arange(rows, dtype=np.int64) % 3
    dense = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    spec = pd.DataFrame({
        "Expression": ["odt_skims['TIME']", "df.x"],
        "A": [0.5, 2.0],
        "B": [-1.0, 0.25],
    })
    environment = {
        "df": {"x": dense},
        "odt_skims": {
            "TIME": CudaDatasetSkimBinding(
                data=cp.asarray(cube),
                orig=cp.asarray(orig),
                dest=cp.asarray(dest),
                time=cp.asarray(period),
                dest_count=5,
                time_count=3,
            )
        },
    }
    result = evaluate_strict_cuda(
        specification_ir(spec),
        environment,
        return_device=True,
        capture_features=False,
        compact_inputs=True,
        group_skim_indices=True,
        capture_resident_invocation=True,
    )
    invocation = result.resident_invocation
    expected = cp.asnumpy(result.utilities)
    dense[:] = 99
    environment["odt_skims"]["TIME"].orig[:] = 3
    actual = cp.asnumpy(invocation.execute())
    np.testing.assert_array_equal(actual, expected)
    assert invocation.logical_skim_bindings == 1
    assert invocation.unique_skim_arrays == 1
    assert invocation.shared_skim_data_bytes == cube.nbytes
    assert invocation.skim_coordinate_bytes == rows * 3 * np.dtype(np.int64).itemsize


@pytest.mark.parametrize(
    "tile_rows,cooperative", [(1, False), (1, True), (2, True), (4, True), (8, True)]
)
def test_tiled_strict_cuda_cooperatively_reuses_skims_and_scalars_exactly(
    tile_rows, cooperative
):
    cp = pytest.importorskip("cupy")
    rows = 17
    cube = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    orig = np.arange(rows, dtype=np.int64) % 4
    dest = (np.arange(rows, dtype=np.int64) * 3) % 5
    period = np.arange(rows, dtype=np.int64) % 3
    gathered = cube[orig, dest, period]
    spec = pd.DataFrame({
        "Expression": [
            "odt_skims['TIME'] * scale + df.x",
            "odt_skims['TIME'] / denom",
            "df.flag & active",
        ],
        "A": [0.25, -0.5, 2.0],
        "B": [-0.75, 0.125, -3.0],
    })
    document = specification_ir(spec)
    common = {
        "df": {
            "x": np.linspace(-2, 3, rows),
            "flag": np.arange(rows) % 2 == 0,
        },
        "scale": 2.5,
        "denom": 3.0,
        "active": True,
    }
    cpu_environment = {**common, "odt_skims": {"TIME": gathered}}
    cuda_environment = {
        **common,
        "odt_skims": {
            "TIME": CudaDatasetSkimBinding(
                data=cp.asarray(cube),
                orig=cp.asarray(orig),
                dest=cp.asarray(dest),
                time=cp.asarray(period),
                dest_count=5,
                time_count=3,
            )
        },
    }
    cpu = evaluate_strict_cpu(document, cpu_environment)
    cuda = evaluate_strict_cuda(
        document,
        cuda_environment,
        locality_tile_rows=tile_rows,
        locality_optimized=cooperative,
        compact_inputs=True,
        group_skim_indices=True,
        sparse_zero_coefficients=True,
    )
    assert compare_strict_cpu_cuda(cpu, cuda)["exact_gate_passed"]
    assert cuda.telemetry.tile_rows == tile_rows
    assert cuda.telemetry.dense_row_inputs == 2
    assert cuda.telemetry.scalar_inputs == 3
    assert cuda.telemetry.unique_skim_bindings == 1
    assert cuda.telemetry.skim_reference_uses == 2
    assert cuda.telemetry.skim_loads_avoided_per_row == (1 if cooperative else 0)
    assert cuda.telemetry.grouped_skim_indices
    assert cuda.telemetry.skim_index_groups == 1
    assert cuda.telemetry.sparse_zero_coefficients == (not cooperative)
