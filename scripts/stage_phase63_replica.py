"""Stage current source in a NEW directory for a clean local reconstruction.

No original environment, cache, input data or model answers are copied. The
staged bootstrap must retrieve inputs and build its own numerical environment.
This is a same-machine reconstruction, never independent hardware replication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError("Replica staging never overwrites an existing directory")
    destination.mkdir(parents=True)
    for name in ("src","scripts","tests","benchmarks","integration","reproducibility"):
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns(
            "__pycache__","*.pyc","phase57_postprocess_cache"))
    for name in ("pyproject.toml","README.md","requirements-phase63-lock.txt"):
        shutil.copy2(ROOT/name,destination/name)
    for name in ("configs_phase33_choiceforge","configs_phase59_seed17","configs_phase59_seed991","configs_phase63_coeff"):
        shutil.copytree(ROOT/"benchmark-data"/name,destination/"benchmark-data"/name)
    (destination/"benchmark-results").mkdir()
    records = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination)
            source = ROOT/relative
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f"Copy differs: {relative}")
            records[relative.as_posix()] = expected
    receipt = dict(complete=True,same_machine=True,copied_model_answers=False,copied_environment=False,
                   copied_input_data=False,copied_runtime_program_cache=False,
                   shipped_source_keyed_kernel_binaries=True,source_sha256=records)
    (destination/"staging-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(destination=str(destination),verified_files=len(records))))


if __name__=="__main__":
    main()
