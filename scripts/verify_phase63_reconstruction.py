"""Record exact same-machine reconstruction and executed pytest XML evidence.

Run after all measurements: input/installation hashing is not benchmark work.
The receipt proves present-state equality, not a second-machine speed claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from run_phase58_comparison import ROOT, source_fingerprint


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def test_result(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    count = lambda key: sum(int(s.get(key, "0")) for s in suites)
    skipped = count("skipped")
    failed = count("failures") + count("errors")
    total = count("tests")
    require(total > 0 and failed == 0, f"Unsuccessful test report: {path}")
    return dict(passed=total - skipped - failed, skipped=skipped, failed=failed, exit_code=0,
                junit=str(path), seconds=sum(float(s.get("time", "0")) for s in suites),
                skip_reasons=[node.get("message", "") for node in root.iter("skipped")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--main-junit", type=Path, required=True)
    parser.add_argument("--replica-junit", type=Path, required=True)
    args = parser.parse_args()
    replica = args.replica.resolve()
    source = source_fingerprint()
    differences = [name for name, value in source.items() if digest(replica / name) != value]
    require(not differences, f"Replica production differs: {differences}")
    result_path = ROOT / "benchmark-results/phase63-reconstruction-verification.json"
    tests_path = ROOT / "benchmark-results/phase63-test-verification.json"
    require(not result_path.exists() and not tests_path.exists(), "Existing receipts must not be overwritten")
    command = [str(replica / ".venv-phase8/Scripts/python.exe"),
               str(replica / "scripts/prepare_phase63_replication.py"), "check"]
    completed = subprocess.run(command, cwd=replica, text=True, capture_output=True, check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    check = json.loads(completed.stdout)
    require(check["prepared_workspace_verified"] and check["packages"] == 98, "Replica check incomplete")
    manifest = ROOT / "reproducibility/phase63-inputs.json"
    inputs = json.loads(manifest.read_text())
    project = replica / "benchmark-data/phase9-mtc-full/prototype_mtc_extended"
    archive = project / "downloads/data_full.tar.zst"
    require(digest(archive) == inputs["public_archive"]["sha256"], "Downloaded archive differs")
    evidence = [manifest, ROOT / "reproducibility/phase63-upstream-source.json",
                ROOT / "integration/activitysim-phase63.patch", ROOT / "requirements-phase63-lock.txt",
                replica / "staging-receipt.json", Path(__file__),
                replica / "scripts/bootstrap_phase63.ps1", replica / "scripts/prepare_phase63_replication.py"]
    for name in inputs["data"]:
        evidence.append(project / "data_full" / name)
    result = dict(status="passed", same_machine=True, independent_hardware_replication=False,
                  replica=str(replica), source_differences=differences, matched_production_files=len(source),
                  check_command=command, check_result=check, archive_bytes=archive.stat().st_size,
                  archive_sha256=digest(archive), public_archive_url=inputs["public_archive"]["url"],
                  cache_scope="Replica-local CuPy/Numba caches; tests and preparation warm them before timing.",
                  staging_note="Initial receipt precedes source-only benchmark-script repair and final fingerprint expansion. No original model answers were staged.",
                  source_sha256=source, evidence_sha256={str(p): digest(p) for p in evidence})
    main_tests = test_result(args.main_junit.resolve())
    clean_tests = test_result(args.replica_junit.resolve())
    require(main_tests["passed"] >= 602 and main_tests["skipped"] == 0, "Unexpected main tests")
    require(clean_tests["passed"] == main_tests["passed"] - 1 and clean_tests["skipped"] == 1,
            "Unexpected clean-test count")
    require("public MTC checkpoint not staged" in " ".join(clean_tests["skip_reasons"]),
            "Unexpected clean-test skip reason")
    test_files = sorted((ROOT / "tests").rglob("*.py"))
    require(all(digest(replica / p.relative_to(ROOT)) == digest(p) for p in test_files),
            "Test sources differ between main and replica")
    test_evidence = [args.main_junit.resolve(), args.replica_junit.resolve(), *test_files]
    tests = dict(status="passed", main=main_tests, clean_replica=clean_tests,
                 evidence_sha256={str(p): digest(p) for p in test_evidence})
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    tests_path.write_text(json.dumps(tests, indent=2) + "\n")
    print(json.dumps(dict(reconstruction="passed", main=main_tests, clean_replica=clean_tests), indent=2))


if __name__ == "__main__":
    main()
