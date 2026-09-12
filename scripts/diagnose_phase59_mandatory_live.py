"""Instrumented live scheduling audit, inputs/results only; never timing evidence.

Wraps the scheduler to observe its own calculations. No CPU expected choices,
population-specific correction rules, or reference files enter this wrapper.
"""
import argparse
from pathlib import Path
import runpy
import sys
import numpy as np
from choiceforge.cuda_backend import _cupy
from choiceforge.phase59_live_mandatory import LiveMandatoryScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--audit-dir", type=Path, required=True)
args, remaining = parser.parse_known_args()
args.audit_dir.mkdir(parents=True, exist_ok=False)
original = LiveMandatoryScheduler.choose


def observed(self, ids, draws, values, *a, **k):
    cp = _cupy()
    batch = self.cursor
    model = self.batches[batch]["model"]
    calculate = model.choose
    observed_values = {}

    def capture(*ma, **mk):
        result = calculate(*ma, **mk)
        observed_values["distances"] = cp.asnumpy(result.boundary_distances)
        observed_values["initial_tdds"] = cp.asnumpy(ma[3][ma[4][:-1]+result.choices])
        return result

    model.choose = capture
    try:
        result = original(self, ids, draws, values, *a, **k)
    finally:
        model.choose = calculate
    np.savez(args.audit_dir/f"batch-{batch}.npz", chooser_ids=np.asarray(ids),
             draws=cp.asnumpy(cp.asarray(draws)), final_tdds=cp.asnumpy(cp.asarray(result)),
             **observed_values)
    return result


LiveMandatoryScheduler.choose = observed
runner = Path(__file__).with_name("run_phase22_integrated_scheduling.py")
sys.argv = [str(runner), *remaining]
runpy.run_path(str(runner), run_name="__main__")
