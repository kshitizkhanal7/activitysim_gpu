"""Build the public Phase 31 native skim artifact once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omx", type=Path, required=True)
    parser.add_argument("--land-use", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from choiceforge.native_skim_store import build_native_skim_store
    from choiceforge.sharrow_ir import specification_ir

    document = specification_ir(pd.read_csv(args.spec, comment="#"))
    manifest = build_native_skim_store(
        args.omx, args.land_use, document, args.output
    )
    report = {
        "phase": 31,
        "scope": "one-time immutable public native skim-store build",
        "artifact": str(args.output.resolve()),
        "manifest": manifest,
        "proof_gates": {
            "all_209_logical_bindings_declared": manifest["logical_bindings"] == 209,
            "all_149_physical_cubes_packed": len(manifest["entries"]) == 149,
            "public_zone_count": manifest["zone_count"] == 1454,
            "payload_is_float32_hot_set": manifest["payload_bytes"] == 6_198_588_112,
            "payload_hash_recorded": len(manifest["payload_sha256"]) == 64,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": report["artifact"],
        "payload_bytes": manifest["payload_bytes"],
        "payload_sha256": manifest["payload_sha256"],
        "proof_gates": report["proof_gates"],
    }, indent=2))
    if not all(report["proof_gates"].values()):
        raise SystemExit("Phase 31 store-build proof gate failed")


if __name__ == "__main__":
    main()
