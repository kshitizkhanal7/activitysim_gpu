"""Equivalent compiled CPU/GPU reductions on actual captured live mode inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numba
import numpy as np
from choiceforge.cuda_backend import _cupy
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from choiceforge.phase59_cpu_algorithms import modes_cpu
from choiceforge.nested_logit import MTC21_ALTERNATIVES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    files = sorted(args.inputs.glob("batch-*.npz"))
    if not files: raise ValueError("No actual live inputs")
    cp = _cupy()
    batches = []
    identities = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            nest = json.loads(str(data["nest_json"]))
            values, draws = np.ascontiguousarray(data["utilities"]), np.ascontiguousarray(data["draws"]).reshape(-1)
            identities.extend(data["chooser_ids"].tolist())
            batches.append((values, draws, validate(nest, MTC21_ALTERNATIVES), cp.asarray(values), cp.asarray(draws), nest))
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate live chooser IDs across mode segments")

    def cpu(): return [modes_cpu(b[0], b[1], b[2]) for b in batches]
    def gpu():
        results = [reduce_modes(b[3], b[4], b[5]) for b in batches]
        cp.cuda.Stream.null.synchronize()
        return results

    cpu(); gpu()  # Explicitly outside the resident-input timing boundary.
    sweep = []
    for threads in (1, 4, 12, 24, 48):
        if threads > numba.config.NUMBA_NUM_THREADS: continue
        numba.set_num_threads(threads)
        cpu(); gpu()
        samples = {"cpu":[], "gpu":[]}
        for trial in range(args.repetitions):
            for backend in (("cpu", "gpu") if trial % 2 == 0 else ("gpu", "cpu")):
                started = time.perf_counter()
                result = cpu() if backend == "cpu" else gpu()
                samples[backend].append(time.perf_counter()-started)
        checks = []
        for expected, actual in zip(cpu(), gpu()):
            guarded = (expected[2] != 0) | (cp.asnumpy(actual[2]) != 0)
            np.testing.assert_array_equal(expected[0][~guarded], cp.asnumpy(actual[0])[~guarded])
            logsums = cp.asnumpy(actual[1])
            np.testing.assert_allclose(expected[1], logsums, atol=1e-12, rtol=1e-12)
            checks.append({"rows":len(expected[0]), "guarded_rows":int(guarded.sum()),
                           "off_boundary_choices_exact":True, "max_logsum_abs":float(np.max(np.abs(expected[1]-logsums)))})
        sweep.append({"threads":threads, "samples_seconds":samples, "checks":checks,
                      "cpu_median_seconds":statistics.median(samples["cpu"]), "gpu_median_seconds":statistics.median(samples["gpu"])})
    best = min(sweep, key=lambda r:r["cpu_median_seconds"])
    root = Path(__file__).resolve().parents[1]
    result = {"scope":"actual public live utilities, uniforms and per-segment nests; equivalent compact reducers; resident inputs and allocation included; excludes JIT, expression evaluation, RNG, packing and transfers",
        "not_full_component_speedup":True, "rows":len(identities), "segments":len(batches), "sweep":sweep,
        "best_cpu_threads":best["threads"], "best_cpu_over_gpu_ratio":best["cpu_median_seconds"]/best["gpu_median_seconds"],
        "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), root/"src/choiceforge/phase58_mode_reduction.py", root/"src/choiceforge/phase59_cpu_algorithms.py")}}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"sweep", "input_sha256", "source_sha256"}}, indent=2))


if __name__ == "__main__": main()
