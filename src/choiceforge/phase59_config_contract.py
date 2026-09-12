"""Permit seed/sample-only settings overlays without weakening compiled specs."""
from collections import defaultdict
import hashlib
from pathlib import Path
import yaml

SCENARIO_KEYS = {"inherit_settings", "rng_base_seed", "households_sample_size", "households_sample_seed"}


def compatible_config(expected, current, directories):
    def grouped(items):
        result = defaultdict(list)
        for key, digest in sorted(items.items(), key=lambda item:int(item[0].split(":", 1)[0])):
            result[key.split(":", 1)[1]].append(digest)
        return dict(result)
    expected_groups = grouped(expected)
    kept = {}
    for key, digest in current.items():
        number, name = key.split(":", 1)
        if name == "settings.yaml" and digest not in expected_groups.get(name, []):
            raw = (Path(directories[int(number)]) / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                return False
            settings = yaml.safe_load(raw)
            if (not isinstance(settings, dict) or settings.get("inherit_settings") is not True
                    or not set(settings) <= SCENARIO_KEYS):
                return False
            # Only this additional, non-arithmetic overlay is omitted. Existing
            # settings, coefficients, model spec, nesting and their order remain
            # byte-hash-validated against the reviewed compiler atlas.
            continue
        kept[key] = digest
    return grouped(kept) == expected_groups
