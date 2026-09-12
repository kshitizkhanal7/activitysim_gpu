"""CPU thread sweep for equivalent compact algorithms, not full-model claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
import numba
from activitysim.abm.models.trip_scheduling import TripSchedulingSettings, PROBS_JOIN_COLUMNS_DEPARTURE_BASED
from activitysim.core.random import Random
from choiceforge.cuda_backend import _cupy
from choiceforge.phase58_trip_runtime import TripRuntime
from choiceforge.phase59_retry import RetrySchedulingService
from choiceforge.phase59_cpu_algorithms import retries_cpu, modes_cpu
from choiceforge.phase58_mode_reduction import reduce_modes, validate
from choiceforge.nested_logit import MTC21_ALTERNATIVES
from choiceforge.modelwide_service import Phase46DestinationService

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "benchmark-data/phase9-mtc-full/prototype_mtc_extended"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cp = _cupy()
    reference = PROJECT / "o-p17modeproof16-baseline-50000-1"
    trips = pd.read_csv(reference / "final_trips.csv", index_col="trip_id", low_memory=False)
    tours = pd.read_csv(reference / "final_tours.csv", index_col="tour_id", low_memory=False)
    spec = pd.read_csv(PROJECT / "configs/trip_scheduling_probs.csv", comment="#")
    settings = TripSchedulingSettings(logic_version=2, MAX_ITERATIONS=100)
    service = RetrySchedulingService(spec, PROBS_JOIN_COLUMNS_DEPARTURE_BASED, TripRuntime())
    frame, active, groups, host, firstout, firstin = service.pack(trips, tours, settings)
    rng = Random()
    rng.set_base_seed(17)
    rng.add_channel("trips", frame)
    rng.begin_step("trip_scheduling")
    ledger = rng.channels["trips"].row_states.loc[active.index]
    generator = Phase46DestinationService()
    started = time.perf_counter()
    draws_device = generator.generate_from_seeds(ledger.row_seed, ledger.offset, 100)
    cp.cuda.Stream.null.synchronize()
    draw_generation_seconds = time.perf_counter()-started
    draws = cp.asnumpy(draws_device)
    device = [cp.asarray(a) for a in host]
    probabilities = cp.asnumpy(service.spec_probabilities)
    cpu_args = (*host, draws, probabilities, settings.DEPART_ALT_BASE, firstout, firstin, 100)

    def gpu_retry():
        result = cp.full(len(frame), -1, cp.int32)
        consumed = cp.zeros(len(frame), cp.int32)
        counts = cp.zeros((100,4), cp.int32)
        service.retry_kernel(((groups+127)//128,), (128,), (device[0], np.int32(groups), *device[1:],
            draws_device, service.spec_probabilities, np.int32(probabilities.shape[1]),
            np.int32(settings.DEPART_ALT_BASE), np.int32(firstout), np.int32(firstin), np.int32(100),
            result, consumed, counts))
        cp.cuda.Stream.null.synchronize()
        return result, consumed, counts

    # Nesting topology is the public test fixture; utilities are explicitly synthetic.
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_nested_logit import NEST
    random = np.random.default_rng(5917)
    utilities = random.normal(-2, 4, (len(trips), 21)).astype(np.float32)
    mode_draws = random.random(len(trips))
    utility_device, mode_draw_device = cp.asarray(utilities), cp.asarray(mode_draws)
    coeffs = validate(NEST, MTC21_ALTERNATIVES)

    def gpu_mode():
        result = reduce_modes(utility_device, mode_draw_device, NEST)
        cp.cuda.Stream.null.synchronize()
        return result

    controls = {}
    for name, cpu, gpu in (("complete_departure_retries", lambda:retries_cpu(*cpu_args), gpu_retry),
                           ("nested_mode_reduction", lambda:modes_cpu(utilities, mode_draws, coeffs), gpu_mode)):
        cpu()  # JIT compilation is excluded and explicitly disclosed.
        gpu()
        sweep = []
        for threads in (1, 4, 12, 24, 48):
            if threads > numba.config.NUMBA_NUM_THREADS:
                continue
            numba.set_num_threads(threads)
            cpu()
            samples = {"cpu":[], "gpu":[]}
            for trial in range(args.repetitions):
                for backend in (("cpu", "gpu") if trial % 2 == 0 else ("gpu", "cpu")):
                    started = time.perf_counter()
                    result = cpu() if backend == "cpu" else gpu()
                    samples[backend].append(time.perf_counter()-started)
            expected, actual = cpu(), gpu()
            if name == "complete_departure_retries":
                np.testing.assert_array_equal(expected[0], cp.asnumpy(actual[0]))
                np.testing.assert_array_equal(expected[1], cp.asnumpy(actual[1]))
                assert int(expected[2].sum()) == int(cp.asnumpy(actual[2])[:,2].sum())
                correctness = {"choices_exact":True, "rng_counts_exact":True, "failure_count_exact":True}
            else:
                guards = (expected[2] != 0) | (cp.asnumpy(actual[2]) != 0)
                np.testing.assert_array_equal(expected[0][~guards], cp.asnumpy(actual[0])[~guards])
                np.testing.assert_allclose(expected[1], cp.asnumpy(actual[1]), atol=1e-12, rtol=1e-12)
                correctness = {"off_boundary_choices_exact":True, "guarded_rows":int(guards.sum()),
                               "max_logsum_abs":float(np.max(np.abs(expected[1]-cp.asnumpy(actual[1]))))}
            sweep.append({"threads":threads, "samples_seconds":samples,
                "cpu_median_seconds":statistics.median(samples["cpu"]),
                "gpu_median_seconds":statistics.median(samples["gpu"]), "correctness":correctness})
        best = min(sweep, key=lambda x:x["cpu_median_seconds"])
        controls[name] = {"sweep":sweep, "best_cpu_threads":best["threads"],
            "best_cpu_over_gpu_ratio":best["cpu_median_seconds"]/best["gpu_median_seconds"]}
    document = {"scope":"allocation plus compact computation; already resident inputs; compilation, packing, RNG generation and transfers excluded",
        "not_a_full_component_or_full_model_speedup":True,
        "scheduling_input":"public 50k final trip/tour structure, fresh live seed 17, actual probability spec; no expected choices read",
        "mode_input":"synthetic utilities and draws at the public trip row count; public nesting topology",
        "rows":len(trips), "groups":groups, "active_trips":len(active),
        "shared_gpu_random_preparation_seconds":draw_generation_seconds, "controls":controls,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src/choiceforge").glob("phase59*.py"))},
        "thread_environment":{k:os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    args.output.write_text(json.dumps(document, indent=2)+"\n")
    print(json.dumps({name:{k:v for k,v in control.items() if k != "sweep"} for name,control in controls.items()}, indent=2))


if __name__ == "__main__":
    main()
