#!/usr/bin/env python3
"""Exhaustively qualify Phase 48's float32 exponential ABI.

The scan visits every one of the 2**32 possible float32 bit patterns, compares
the CUDA approximation used by ChoiceForge with this machine's NumPy float32
``exp``, and proves that the checked-in correction table is complete for the
declared [-80, 80] destination-utility domain.  NaNs and infinities are not
part of the finite input domain.  This is deliberately a qualification tool,
not a benchmark shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from choiceforge.arithmetic_abi import numpy_float32_choice_cuda_helpers
from choiceforge.cuda_backend import _cupy
from choiceforge.modelwide_graph import (
    CONTRACT,
    _EXP_CORRECTIONS,
    _EXP_CORRECTION_SHA256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/phase48-exp-domain-scan.json"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1 << 24,
        help="Number of uint32 bit patterns per chunk (default: 16,777,216).",
    )
    return parser


def _compile_kernel(cp):
    source = numpy_float32_choice_cuda_helpers(1) + r'''
extern "C" __global__ void phase48_scan_expf(
    const float* values,
    float* results,
    long long count)
{
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) results[index] = numpy_avx2_expf(values[index]);
}
'''
    kernel = cp.RawKernel(
        source,
        "phase48_scan_expf",
        options=("--std=c++11", "--fmad=false", "--prec-div=true", "--ftz=false"),
    )
    kernel.compile()
    return kernel


def main() -> int:
    args = _parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    cp = _cupy()
    kernel = _compile_kernel(cp)
    checked_table = np.asarray(_EXP_CORRECTIONS, dtype=np.uint32)
    observed_domain_pairs: list[tuple[int, int]] = []
    finite_inputs = 0
    all_finite_mismatches = 0
    domain_inputs = 0
    started = time.perf_counter()

    limit = 1 << 32
    for start in range(0, limit, args.chunk_size):
        count = min(args.chunk_size, limit - start)
        bits64 = np.arange(start, start + count, dtype=np.uint64)
        bits = bits64.astype(np.uint32, copy=False)
        finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
        finite_bits = bits[finite]
        values = finite_bits.view(np.float32)
        finite_inputs += int(values.size)

        device_values = cp.asarray(values)
        device_results = cp.empty(values.size, dtype=cp.float32)
        blocks = (values.size + 255) // 256
        kernel((blocks,), (256,), (device_values, device_results, np.int64(values.size)))
        gpu_bits = cp.asnumpy(device_results).view(np.uint32)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            cpu_bits = np.exp(values).astype(np.float32, copy=False).view(np.uint32)

        mismatch = gpu_bits != cpu_bits
        all_finite_mismatches += int(np.count_nonzero(mismatch))
        in_domain = (values >= CONTRACT.exponential_domain[0]) & (
            values <= CONTRACT.exponential_domain[1]
        )
        domain_inputs += int(np.count_nonzero(in_domain))
        domain_mismatch = mismatch & in_domain
        if np.any(domain_mismatch):
            observed_domain_pairs.extend(
                zip(
                    finite_bits[domain_mismatch].astype(np.uint32).tolist(),
                    cpu_bits[domain_mismatch].astype(np.uint32).tolist(),
                )
            )

    observed = np.asarray(sorted(observed_domain_pairs), dtype=np.uint32).reshape(-1, 2)
    observed_hash = hashlib.sha256(observed.astype("<u4", copy=False).tobytes()).hexdigest()
    table_exact = np.array_equal(observed, checked_table)
    success = (
        finite_inputs == 4_278_190_080
        and table_exact
        and observed_hash == _EXP_CORRECTION_SHA256
    )
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = device["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    report = {
        "phase": 48,
        "purpose": "exhaustive float32 exponential ABI qualification",
        "success": success,
        "input_bit_patterns_scanned": limit,
        "finite_input_patterns": finite_inputs,
        "all_finite_mismatches_before_correction": all_finite_mismatches,
        "declared_domain": list(CONTRACT.exponential_domain),
        "finite_domain_input_patterns": domain_inputs,
        "domain_mismatches_before_correction": int(observed.shape[0]),
        "domain_correction_entries": int(checked_table.shape[0]),
        "domain_correction_table_exact": table_exact,
        "observed_correction_sha256": observed_hash,
        "contract_correction_sha256": _EXP_CORRECTION_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
        "chunk_size": args.chunk_size,
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu": device_name,
        "proof_boundary": (
            "all 2**32 float32 bit patterns; finite values compared globally; "
            "correction completeness required only inside the fail-closed [-80, 80] ABI"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
