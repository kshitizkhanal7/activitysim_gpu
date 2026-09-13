import numpy as np
import pandas as pd
import pytest
import numba
from choiceforge.phase61_normals import standard_normals, normal
from test_phase58_trip_runtime import state_for


@pytest.mark.parametrize("count",[1,2,3,6,7,16])
@pytest.mark.parametrize("threads",[1,48])
def test_normals_byte_exact_across_offsets_gaussian_cache_and_threads(count,threads):
    rng = np.random.default_rng(61017)
    seeds = np.r_[np.array([0,1,2**32-1],np.uint32),rng.integers(0,2**32,397,dtype=np.uint32)]
    offsets = rng.integers(0,700,len(seeds),dtype=np.int64)
    expected = []
    skipped = []
    for seed,offset in zip(seeds,offsets):
        state = np.random.RandomState(int(seed))
        skipped.append(sum(state.random_sample(int(offset))))
        expected.append(state.standard_normal(count))
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(min(threads,numba.config.NUMBA_NUM_THREADS))
        actual,discarded = standard_normals(seeds,offsets,count)
    finally:
        numba.set_num_threads(previous)
    np.testing.assert_array_equal(actual.view(np.uint64),np.asarray(expected).view(np.uint64))
    np.testing.assert_array_equal(discarded,np.asarray(skipped))


@pytest.mark.parametrize("seed",[0,17,991])
def test_public_random_broadcast_scaled_lognormal_and_ledger(seed,monkeypatch):
    from activitysim.core.random import SimpleChannel
    frame = pd.DataFrame(index=pd.Index([311,22,530,719,813,44],name="trip_id"))
    reference,actual = state_for(frame,seed),state_for(frame,seed)
    a,b = reference.get_rn_generator(),actual.get_rn_generator()
    a.begin_step("trip_mode_choice")
    b.begin_step("trip_mode_choice")
    original = SimpleChannel.normal_for_df
    events = []
    for size in (None,1,3,6,7):
        for broadcast in (False,True):
            rows = frame.iloc[[4,0,5,2,4]] if broadcast else frame.iloc[::-1]
            expected = a.normal_for_df(rows,size=size,broadcast=broadcast)
            with monkeypatch.context() as m:
                m.setattr(SimpleChannel,"normal_for_df",lambda *args,**kw:normal(original,events,*args,**kw))
                result = b.normal_for_df(rows,size=size,broadcast=broadcast)
            np.testing.assert_array_equal(np.asarray(result).view(np.uint64),np.asarray(expected).view(np.uint64))
    for mu,sigma in ((1.,.2),(0.,0.),(-0.,1.)):
        expected = a.lognormal_for_df(frame,mu,sigma,broadcast=True)
        with monkeypatch.context() as m:
            m.setattr(SimpleChannel,"normal_for_df",lambda *args,**kw:normal(original,events,*args,**kw))
            result = b.lognormal_for_df(frame,mu,sigma,broadcast=True)
        np.testing.assert_array_equal(np.asarray(result).view(np.uint64),np.asarray(expected).view(np.uint64))
    pd.testing.assert_frame_equal(a.channels["trips"].row_states,b.channels["trips"].row_states)
    np.testing.assert_array_equal(a.random_for_df(frame),b.random_for_df(frame))
    assert events
