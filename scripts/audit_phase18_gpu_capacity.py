"""Write an evidence-backed Phase 18 capacity audit for the local GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark-data" / "phase9-mtc-full" / "prototype_mtc_extended" / "data_full"
OUTPUT = ROOT / "benchmark-results" / "phase18-capacity-audit.json"
PHASE17 = ROOT / "benchmark-results" / "phase9-mtc-full-p17modeproof16-runs.json"
PHASE18 = ROOT / "benchmark-results" / "phase18-gpu-native-full-households.json"
GIB = 1024**3


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in stream) - 1


def main() -> None:
    smi_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    gpu_name, total_mib, free_mib, driver = [item.strip() for item in smi_text.split(",")]
    total_bytes = int(total_mib) * 1024**2

    skim_path = DATA / "skims.omx"
    dataset_count = 0
    skim_raw_bytes = 0
    largest = []
    with h5py.File(skim_path, "r") as store:
        def visit(name, obj):
            nonlocal dataset_count, skim_raw_bytes
            if isinstance(obj, h5py.Dataset):
                dataset_count += 1
                raw = int(obj.size * obj.dtype.itemsize)
                skim_raw_bytes += raw
                largest.append((raw, name, list(obj.shape), str(obj.dtype)))
        store.visititems(visit)

    phase17 = json.loads(PHASE17.read_text(encoding="utf-8"))
    p17_gpu_peaks = [
        float(run["gpu_peak_memory_mib"])
        for run in phase17["runs"]
        if run["condition"] == "choiceforge"
    ]
    phase18 = json.loads(PHASE18.read_text(encoding="utf-8"))

    # Planning allocations are intentionally conservative. They are design
    # assumptions, not claims that Phase 18 has already filled these pools.
    reserve_bytes = 2 * GIB
    hot_skim_bytes = 4 * GIB
    persistent_state_bytes = 2 * GIB
    workspace_bytes = 3 * GIB
    committed = reserve_bytes + hot_skim_bytes + persistent_state_bytes + workspace_bytes
    remaining = total_bytes - committed

    report = {
        "phase": 18,
        "generated_from_live_machine": True,
        "gpu": {
            "name": gpu_name,
            "driver": driver,
            "reported_total_mib": int(total_mib),
            "reported_free_mib_at_audit": int(free_mib),
            "total_bytes": total_bytes,
        },
        "public_mtc": {
            "households": csv_rows(DATA / "households.csv"),
            "persons": csv_rows(DATA / "persons.csv"),
            "files_on_disk_bytes": {
                name: (DATA / name).stat().st_size
                for name in ("households.csv", "persons.csv", "land_use.csv", "skims.omx")
            },
            "skim_dataset_count": dataset_count,
            "skim_raw_uncompressed_bytes": skim_raw_bytes,
            "skim_raw_uncompressed_gib": skim_raw_bytes / GIB,
            "largest_skim_datasets": [
                {"name": name, "bytes": size, "shape": shape, "dtype": dtype}
                for size, name, shape, dtype in sorted(largest, reverse=True)[:5]
            ],
        },
        "prior_full_activitysim_evidence": {
            "scope": "50,000-household Phase 17 public MTC runs",
            "choiceforge_gpu_peak_mib_samples": p17_gpu_peaks,
            "choiceforge_gpu_peak_mib_max": max(p17_gpu_peaks),
        },
        "phase18_vertical_slice_evidence": {
            "households": phase18["input"]["households"],
            "sampled_device_active_peak_bytes": phase18["telemetry"][
                "sampled_device_active_peak_bytes"
            ],
            "gpu_compute_speedup_vs_parallel_numba": phase18["speedup"][
                "gpu_compute_vs_numba"
            ],
            "gpu_total_speedup_vs_parallel_numba": phase18["speedup"][
                "gpu_total_with_transfer_vs_numba"
            ],
            "qualified": phase18["qualified"],
        },
        "planning_budget": {
            "status": "design assumption requiring stage-by-stage high-water qualification",
            "reserve_bytes": reserve_bytes,
            "hot_skim_cache_bytes": hot_skim_bytes,
            "persistent_state_bytes": persistent_state_bytes,
            "workspace_bytes": workspace_bytes,
            "committed_bytes": committed,
            "unallocated_bytes": remaining,
        },
        "conclusions": [
            "The complete uncompressed skim collection cannot coexist with a safe runtime reserve and full model state on this 16 GB device.",
            "A hot-skim cache plus deterministic household partitions is mandatory for the eventual full GPU-native model.",
            "The Phase 18 vertical slice fits the entire public household table, but it is much smaller than a complete ActivitySim state graph.",
            "No full-model maximum partition size is claimed until every remaining model component has a measured high-water mark.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
