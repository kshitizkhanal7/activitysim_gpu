"""Exact summary values; only declared departure-key decimal spelling may vary."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import io
from pathlib import Path


DEPARTURE_REPORTS = frozenset({
    "non_mandatory_tours_tod_count.csv", "work_tours_tod_count.csv",
    "trip_purpose_by_time_of_day.csv", "school_tours_tod_count.csv",
})


def representation_differences(name, left, right):
    """No tolerance: preserve inventory, headers, row order and all other cells."""
    if name not in DEPARTURE_REPORTS:
        raise ValueError(f"Published report bytes differ: {name}")
    a = list(csv.reader(io.StringIO(left.decode("utf-8"))))
    b = list(csv.reader(io.StringIO(right.decode("utf-8"))))
    if not a or not b or a[0] != b[0] or a[0].count("depart") != 1 or len(a) != len(b):
        raise ValueError(f"Published report schema/rows differ: {name}")
    column = a[0].index("depart")
    changes = []
    for row_number, (x, y) in enumerate(zip(a[1:], b[1:]), start=2):
        if len(x) != len(a[0]) or len(y) != len(a[0]):
            raise ValueError(f"Published report row shape differs: {name}:{row_number}")
        for index, (old, new) in enumerate(zip(x, y)):
            if old == new:
                continue
            equal = False
            if index == column:
                try:
                    old_value, new_value = Decimal(old), Decimal(new)
                    equal = old_value.is_finite() and new_value.is_finite() and old_value == new_value
                except InvalidOperation:
                    pass
            if not equal:
                raise ValueError(f"Published report cell differs: {name}:{row_number}:{index+1}")
            changes.append({"row":row_number, "column":"depart", "reference":old, "candidate":new})
    # Do not silently normalize unrelated CSV quoting or newline changes.
    if not changes:
        raise ValueError(f"Undeclared report byte difference: {name}")
    return changes


def verify(reference, candidate):
    def inventory(directory):
        root = directory / "summarize"
        paths = sorted(p for p in root.rglob("*") if p.is_file())
        if not paths:
            raise ValueError("No published summary reports")
        return {p.relative_to(root).as_posix():p for p in paths}
    left, right = inventory(reference), inventory(candidate)
    if left.keys() != right.keys():
        raise ValueError(f"Published report inventory differs: {sorted(left.keys() ^ right.keys())}")
    hashes = {"reference":{}, "candidate":{}}
    changes = {}
    for name in left:
        a, b = left[name].read_bytes(), right[name].read_bytes()
        hashes["reference"][name] = hashlib.sha256(a).hexdigest()
        hashes["candidate"][name] = hashlib.sha256(b).hexdigest()
        if a != b:
            changes[name] = representation_differences(name, a, b)
    return {"exact":True, "contract":"exact-values-declared-departure-spelling-v1",
            "report_count":len(right), "byte_identical":not changes,
            "sha256":hashes, "representation_only_differences":changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reference", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Verified {result['report_count']} summary reports exactly")
