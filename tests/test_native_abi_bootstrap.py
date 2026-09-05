import pandas as pd
import pytest


cp = pytest.importorskip("cupy")

from choiceforge.cuda_skims import CudaDatasetSkimBinding
from choiceforge.native_abi_bootstrap import (
    NativeSkimCube,
    compile_native_strict_abi,
)
from choiceforge.sharrow_cuda import evaluate_strict_cuda
from choiceforge.sharrow_ir import specification_ir


def _document(expression="df.terminal_time + auto_ownership + scale + od_skims['DIST']"):
    return specification_ir(
        pd.DataFrame(
            {
                "Expression": [expression, "df.terminal_time * scale"],
                "A": [0.5, -0.25],
                "B": [-1.0, 2.0],
            }
        )
    )


def test_native_abi_matches_environment_discovered_strict_cuda_exactly():
    import numpy as np

    document = _document()
    rows = 4
    cube = cp.asarray(
        np.arange(9, dtype=np.float32).reshape(3, 3)
    )
    origin = cp.asarray([0, 1, 2, 1], dtype=cp.int64)
    destination = cp.asarray([2, 0, 1, 2], dtype=cp.int64)
    terminal = np.asarray([1.5, 2.0, 3.25, 4.5], dtype=np.float32)
    ownership = np.asarray([0, 1, 2, 3], dtype=np.int64)
    environment = {
        "df": {"terminal_time": terminal},
        "auto_ownership": ownership,
        "scale": 1.25,
        "od_skims": {
            "DIST": CudaDatasetSkimBinding(
                cube, origin, destination, None, 3, 1
            )
        },
    }
    # The public evaluator resolves a mapping skim through __getitem__; use a
    # tiny adapter with the same strict-binding return contract.
    class Wrapper(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    environment["od_skims"] = Wrapper(environment["od_skims"])
    environment["od_skims"]["DIST"].choiceforge_device_skim_binding
    legacy = evaluate_strict_cuda(
        document,
        environment,
        rows=rows,
        return_device=True,
        capture_features=False,
        compact_inputs=True,
        group_skim_indices=True,
        expression_float32=True,
        fused_utility_accumulation=True,
    )

    native = compile_native_strict_abi(
        document,
        {"scale": 1.25},
        lambda source: NativeSkimCube(cube, 3, 1, 2),
        rows=rows,
    )
    invocation = native.invocation
    invocation.float_inputs[:, 0] = cp.asarray(terminal)
    invocation.int_inputs[:, 0] = cp.asarray(ownership)
    invocation.skim_arguments[1][:] = origin
    invocation.skim_arguments[2][:] = destination
    actual = invocation.execute()
    cp.cuda.Stream.null.synchronize()
    assert cp.array_equal(actual, legacy.utilities)
    assert native.manifest["dense_preprocessor_rows_read"] == 0
    assert native.manifest["dense_preprocessor_values_read"] == 0
    assert native.manifest["float_row_sources"] == 1
    assert native.manifest["int_row_sources"] == 1
    assert native.manifest["skim_coordinate_groups"] == 1


def test_native_abi_minimal_row_state_allocates_only_one_contract_row():
    import numpy as np

    document = _document()
    cube = cp.asarray(np.arange(9, dtype=np.float32).reshape(3, 3))
    native = compile_native_strict_abi(
        document,
        {"scale": 1.25},
        lambda source: NativeSkimCube(cube, 3, 1, 2),
        rows=50_000,
        minimal_row_state=True,
    )
    invocation = native.invocation
    assert invocation.rows == 50_000
    assert invocation.float_inputs.shape[0] == 1
    assert invocation.int_inputs.shape[0] == 1
    assert invocation.skim_coordinate_bytes == 16
    assert native.manifest["codegen"]["minimal_row_state"] is True


def test_native_abi_can_prepare_contract_without_compiling_unused_kernel():
    import numpy as np

    document = _document()
    cube = cp.asarray(np.arange(9, dtype=np.float32).reshape(3, 3))
    native = compile_native_strict_abi(
        document,
        {"scale": 1.25},
        lambda source: NativeSkimCube(cube, 3, 1, 2),
        rows=16,
        minimal_row_state=True,
        minimal_output_state=True,
        compile_kernel=False,
    )
    assert native.invocation.kernel is None
    assert native.invocation.utilities.shape[0] == 1
    assert native.manifest["compiled_this_call"] is False
    assert native.manifest["codegen"]["kernel_compiled"] is False
    assert native.manifest["codegen"]["minimal_output_state"] is True


def test_native_abi_fails_closed_for_unknown_row_source():
    document = _document("df.unknown + scale")
    with pytest.raises(ValueError, match="neither declared row state"):
        compile_native_strict_abi(
            document,
            {"scale": 1.0},
            lambda source: None,
            rows=2,
        )


def test_native_abi_fails_closed_for_wrong_skim_rank():
    document = _document("odt_skims['TIME'] + scale")
    cube = cp.zeros((2, 2), dtype=cp.float32)
    with pytest.raises(ValueError, match="rank 2 violates"):
        compile_native_strict_abi(
            document,
            {"scale": 1.0},
            lambda source: NativeSkimCube(cube, 2, 1, 2),
            rows=2,
        )
