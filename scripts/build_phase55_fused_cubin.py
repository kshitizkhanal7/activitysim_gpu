"""Build the reviewed Phase 55 sm_86 destination kernel binary.

This developer-only command turns the checked-in Phase 52 CUDA source into a
device-specific cubin.  Production never invokes NVRTC: it verifies both
source and binary hashes before loading the reviewed binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cupy.cuda import compiler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/choiceforge/kernels/phase52_public_destination_tile4.cu"
DEFAULT_OUTPUT = ROOT / "src/choiceforge/kernels/phase55_public_destination_sm86.cubin"
DEFAULT_MANIFEST = ROOT / "src/choiceforge/kernels/phase55_public_destination_sm86.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="86")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    options = ("--std=c++11", "--fmad=true", "--prec-div=true", "--ftz=true")
    compiled, _mapping = compiler.compile_using_nvrtc(
        source, options=options, arch=str(args.arch), filename=SOURCE.name
    )
    if not compiled.startswith(b"\x7fELF"):
        raise RuntimeError("NVRTC did not return a CUDA ELF cubin")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compiled)
    manifest = {
        "contract": "choiceforge-phase55-reviewed-fused-cubin-v1",
        "architecture": f"sm_{args.arch}",
        "kernel": "choiceforge_strict_ir_v3",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(compiled).hexdigest(),
        "bytes": len(compiled),
        "options": list(options),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
