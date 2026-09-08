"""Reproduce why Phase 58 does not promote Phase 54 normals to a general RNG."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from activitysim.core.random import Random
from choiceforge.modelwide_service import Phase46DestinationService


def audit():
    frame = pd.DataFrame(index=pd.Index([311,22,530,719,813,44], name="trip_id"))
    service = Phase46DestinationService()
    mismatches, compared = [], 0
    for seed in (0,17,991):
        generators = [Random(), Random()]
        for rng in generators:
            rng.set_base_seed(seed)
            rng.add_channel("trips",frame)
        reference, device = generators
        state = SimpleNamespace(get_rn_generator=lambda:device)
        for step in ("trip_destination", "trip_scheduling", "trip_mode_choice"):
            for rng in generators:
                rng.begin_step(step)
            for active in (frame,frame.iloc[[4,0,5,2]],frame.iloc[::-1]):
                for rng in generators:
                    rng.random_for_df(active, 3)
                    mu = pd.Series(np.linspace(.3,1.1,len(active)), index=active.index)
                    rng.lognormal_for_df(active,mu,0,broadcast=True,scale=True)
                    rng.normal_for_df(active,mu=1,sigma=.2,size=3)
                for size in (None,3,7):
                    for broadcast in (False,True):
                        expected = np.asarray(reference.normal_for_df(active,size=size,broadcast=broadcast))
                        actual = service.cp.asnumpy(service.normal_for_df_device(state,active,1 if size is None else size))
                        if size is None:
                            actual = actual[:,0]
                        unequal = expected != actual
                        compared += expected.size
                        if unequal.any():
                            location = tuple(np.argwhere(unequal)[0])
                            mismatches.append({"seed":seed,"step":step,"size":size,"broadcast":broadcast,
                                "different":int(unequal.sum()),"max_abs":float(np.max(np.abs(expected-actual))),
                                "example_cpu":float(expected[location]),"example_gpu":float(actual[location])})
                for rng in generators:
                    rng.random_for_df(active)
            pd.testing.assert_frame_equal(reference.channels["trips"].row_states,device.channels["trips"].row_states)
            for rng in generators:
                rng.end_step(step)
    return {"status":"rejected_general_exact_normal_rng" if mismatches else "no_mismatch_in_this_test",
            "compared":compared,"different":sum(m["different"] for m in mismatches),
            "ledger_exact":True,"mismatches":mismatches,
            "production_decision":"Phase58 ordinary normals remain on authoritative CPU; no tolerance relaxation."}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=audit()
    from pathlib import Path
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="mismatches"}))
