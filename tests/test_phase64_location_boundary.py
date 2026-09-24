import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from choiceforge.phase64_location_boundary import LocationBoundary
from choiceforge import modelwide_final


@pytest.mark.parametrize("advance_rng",[False,True])
@pytest.mark.parametrize("with_sampling",[False,True])
def test_live_cpu_boundary_borrows_normals_preserves_width_and_restores(monkeypatch,advance_rng,with_sampling):
    from activitysim.core import simulate
    before=simulate.simple_simulate_logsums
    channel=SimpleNamespace(row_states=pd.DataFrame({"offset":[36,36]},index=[10,20]))
    def forbidden(*a,**k):
        raise AssertionError("Must not consume new random numbers")
    rng=SimpleNamespace(normal_for_df=forbidden,get_channel_for_df=lambda frame:channel)
    state=SimpleNamespace(get_rn_generator=lambda:rng,settings=SimpleNamespace(skip_failed_choices=False))
    service=SimpleNamespace(cp=SimpleNamespace(asarray=np.asarray,asnumpy=np.asarray))
    sample=pd.DataFrame({"alt_dest":[1,2,3,4,5],"mode_choice_logsum":0.},index=[10,10,20,20,20])
    owners=pd.DataFrame({"feature":[1.,2.]},index=[10,20])
    controller=LocationBoundary.__new__(LocationBoundary)
    controller.events=[]
    controller.sample_context=None
    controller.context=dict(state=state,choosers=sample,owner_choosers=owners,
        normal_ids=owners.index,normal_values=np.arange(12).reshape(2,6),tour_purpose="work",
        logsum_settings=None,model_settings=None,network_los=None,chunk_tag="test",trace_label="test",
        in_period_col=None,out_period_col=None,duration_col=None)
    controller.cpu_simple=object()
    def logsums(state,frame,*a,**k):
        assert simulate.simple_simulate_logsums is controller.cpu_simple
        values=rng.normal_for_df(frame,broadcast=True,size=6)
        if advance_rng:
            channel.row_states.loc[20,"offset"]+=6
        return values.iloc[:,0]+frame.feature
    controller.cpu_logsums=logsums
    controller.final_signature=inspect.signature(modelwide_final.device_compact_interaction_sample_simulate)
    if with_sampling:
        from activitysim.core import interaction_simulate
        controller.sample_context=dict(choosers=owners,alternatives=pd.DataFrame(index=[1,2,3,4,5]),
            spec=None,locals_d=None,trace_label="test",zone_layer=None,compute_settings=None)
        monkeypatch.setattr(interaction_simulate,"eval_interaction_utilities",
            lambda **k:(SimpleNamespace(utility=np.arange(5,dtype=np.float32)),None))
    def final_utility(state,choosers,alts,*a):
        if with_sampling:
            weights=np.exp(np.arange(5,dtype=np.float32)-4)
            np.testing.assert_array_equal(alts.prob.to_numpy(),(weights/weights.sum())[2:])
        return np.asarray(alts.mode_choice_logsum+7,dtype=np.float32)
    monkeypatch.setattr(modelwide_final,"_cpu_reference_utility",final_utility)
    def final(*a,**k):
        utilities=service.phase64_boundary_utilities(np.array([1]),4)
        np.testing.assert_array_equal(utilities,[[15.,15.,15.,-999.]])
        return "done"
    controller.final_original=final
    args=(state,owners,sample,pd.DataFrame({"a":[1.]}),"alt_dest")
    if advance_rng:
        with pytest.raises(ValueError,match="advanced RNG"):
            controller.final(*args,service=service)
        assert not controller.events[0]["complete"]
        assert not controller.events[0]["rng_unchanged"]
    else:
        assert controller.final(*args,service=service)=="done"
        assert controller.events[0]["normal_values_downloaded"]==6
        assert controller.events[0]["complete"]
        assert controller.events[0]["rng_unchanged"]
    assert rng.normal_for_df is forbidden
    assert simulate.simple_simulate_logsums is before
    assert controller.context is None
    assert controller.sample_context is None
    assert not hasattr(service,"phase64_boundary_utilities")


def test_final_rejects_missing_producer():
    controller=LocationBoundary()
    with pytest.raises(ValueError,match="matching live boundary"):
        controller.final(None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),"x",service=None)
