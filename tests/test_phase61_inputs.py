from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from choiceforge.phase61_inputs import SharedInputs, direct_lookup


@pytest.mark.parametrize("reverse",[False,True])
@pytest.mark.parametrize("layout",["2d","3d","permuted"])
@pytest.mark.parametrize("fixed",[False,True])
def test_direct_skim_matches_actual_sharrow_lookup(reverse,layout,fixed):
    import xarray as xr
    import sharrow
    from activitysim.core.skim_dataset import DatasetWrapper
    cube = np.arange(3*3*2,dtype=np.float32).reshape(3,3,2)
    dims = ("otaz","dtaz","time_period")
    if layout == "2d": cube,dims = cube[:,:,0],dims[:2]
    if layout == "permuted": cube,dims = cube.transpose(2,1,0),dims[::-1]
    dataset = xr.Dataset({"X":(dims,cube)},coords={"time_period":["AM","PM"]})
    frame = pd.DataFrame({"o":[0,2,-1],"d":[2,1,0],"p":["AM","PM","AM"]},index=[19,19,5])
    wrapper = DatasetWrapper(dataset,"o","d",None if layout == "2d" else "p").set_df(frame)
    key = ("X","PM") if fixed else "X"
    try:
        expected = wrapper.lookup(key,reverse)
    except ValueError:
        with pytest.raises(ValueError): direct_lookup(DatasetWrapper.lookup,wrapper,key,reverse)
    else:
        pd.testing.assert_series_equal(direct_lookup(DatasetWrapper.lookup,wrapper,key,reverse),expected)


def test_skim_invalid_target_and_position_fail_closed():
    original = lambda *a:None
    wrapper = SimpleNamespace(df=None)
    with pytest.raises(ValueError,match="target"): direct_lookup(original,wrapper,"X")


def test_shared_columns_are_live_and_derived_columns_not_borrowed():
    from choiceforge.cuda_backend import _cupy
    cp = _cupy()
    store = SharedInputs()
    frame = pd.DataFrame({"x":[1.,-0.,3.],"depart":[4,5,6]},index=[10,20,30])
    state = SimpleNamespace(get_dataframe=lambda name:frame)
    store.publish(state,"trips")
    chooser = frame.loc[[30,10]].copy()
    chooser["depart"] += 1
    env = {"df":{n:chooser[n].to_numpy() for n in chooser}}
    env.update(env["df"])
    store.bind("trips",chooser,env)
    np.testing.assert_array_equal(cp.asnumpy(env["x"]),[3.,1.])
    assert isinstance(env["depart"],np.ndarray)
    assert store.events[-1]["columns"] == ["x"]
    prior = store.store.lease("trips",["x"])
    frame.loc[10,"x"] = 7.
    store.publish(state,"trips")
    with pytest.raises(ValueError,match="stale"): prior.arrays()
    with pytest.raises(ValueError,match="unknown"):
        store.bind("trips",frame.rename(index={10:99}),env)


def test_runtime_hooks_restore_after_failure():
    from activitysim.core.skim_dataset import DatasetWrapper
    from choiceforge.phase61_runtime import Runtime
    prior = DatasetWrapper.lookup
    runtime = Runtime("skims")
    with pytest.raises(RuntimeError):
        with runtime.for_step(None,"test"):
            assert DatasetWrapper.lookup is not prior
            raise RuntimeError("test")
    assert DatasetWrapper.lookup is prior
    assert runtime.events[-1]["saved_answers_read"] is False


@pytest.mark.parametrize("size",[0,1,1000])
@pytest.mark.parametrize("fmt",["{left:,.2f} - {right:,.2f}","{rank}","{mid}","{rank:02d}"])
def test_categorical_labels_preserve_original_values_and_errors(size,fmt):
    from activitysim.abm.models.summarize import construct_bin_labels
    from choiceforge.phase61_inputs import categorical_labels
    bins = pd.Series(pd.cut(np.arange(size)%15,bins=[-1,2,6,20]))
    try:
        expected = construct_bin_labels(bins,fmt)
    except (ValueError,TypeError) as error:
        with pytest.raises(type(error)): categorical_labels(construct_bin_labels,bins,fmt)
    else:
        pd.testing.assert_series_equal(categorical_labels(construct_bin_labels,bins,fmt),expected)


@pytest.mark.parametrize("seed",[0,17,991])
def test_global_uniform_extension_preserves_real_channel_ledger(seed):
    from test_phase58_trip_runtime import state_for
    from choiceforge.phase61_runtime import Runtime
    frame = pd.DataFrame(index=pd.Index([311,22,530,719,813,44],name="trip_id"))
    reference,actual = state_for(frame,seed),state_for(frame,seed)
    for step in ("cdap_simulate","mandatory_tour_frequency"):
        a,b = reference.get_rn_generator(),actual.get_rn_generator()
        a.begin_step(step)
        b.begin_step(step)
        for n in (1,3,7):
            expected = a.random_for_df(frame,n)
            with Runtime("uniforms").for_step(actual,step):
                result = b.random_for_df(frame,n)
            np.testing.assert_array_equal(result,expected)
        pd.testing.assert_frame_equal(a.channels["trips"].row_states,b.channels["trips"].row_states)
        a.end_step(step)
        b.end_step(step)
@pytest.mark.parametrize("kind",["int64","uint64","object","empty"])
def test_chain_group_identity(kind):
    from choiceforge.phase61_inputs import chain_groups
    from choiceforge.phase58_schedule_chain import chain_groups as original
    frame=pd.DataFrame({"person_id":[9,1,9,9,1,3],"tour_id":[2,4,2,7,4,2]})
    if kind == "empty":frame=frame.iloc[:0]
    else:frame=frame.astype(kind)
    if kind == "uint64":frame["person_id"] += np.uint64(2**63)
    if kind == "empty":
        with pytest.raises(TypeError):original(frame)
        with pytest.raises(TypeError):chain_groups(original,frame)
        return
    expected,count=original(frame)
    actual,n=chain_groups(original,frame)
    np.testing.assert_array_equal(expected,actual)
    assert n==count
