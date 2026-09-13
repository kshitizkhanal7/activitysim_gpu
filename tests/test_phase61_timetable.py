from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from choiceforge.phase61_timetable import Availability, available_cpu, encode


def oracle(windows,footprints,owners,alts):
    from activitysim.core.timetable import COLLISION_ARRAY
    x = footprints[alts] + (windows[owners] << 3)
    return ~np.isin(x,COLLISION_ARRAY).any(axis=1)


@pytest.mark.parametrize("periods",[1,23,32])
@pytest.mark.parametrize("backend",["cpu","cuda"])
def test_exact_collision_vocabulary_and_live_mutation(periods,backend):
    rng = np.random.default_rng(19)
    values = np.array([0,2,4,6,7],np.int8)
    windows = rng.choice(values,(37,periods))
    footprints = rng.choice(values,(41,periods))
    # Include all 25 state pairs at the high bit, not only random dense masks.
    windows[:5] = 0
    footprints[:5] = 0
    windows[:5,-1] = values
    footprints[:5,-1] = values
    owners = np.r_[np.repeat(np.arange(5),5),rng.integers(0,37,1000)]
    alts = np.r_[np.tile(np.arange(5),5),rng.integers(0,41,1000)].astype(np.int32)
    table = SimpleNamespace(windows=windows,tdd_footprints=footprints,
                            windows_df=pd.DataFrame(index=np.arange(37)*13+9))
    service = Availability(backend=backend)
    ids = table.windows_df.index.take(owners)
    np.testing.assert_array_equal(service(table,ids,alts),oracle(windows,footprints,owners,alts))
    lease = service.store.lease(str(id(table)),["state0"]) if backend == "cuda" else None
    windows[0] = 7
    np.testing.assert_array_equal(service(table,ids,alts),oracle(windows,footprints,owners,alts))
    if lease is not None:
        with pytest.raises(ValueError,match="stale"): lease.arrays()


@pytest.mark.parametrize("bad",["periods","state","owner","alt"])
def test_unsupported_timetable_inputs_rejected(bad):
    windows = np.zeros((2,33 if bad == "periods" else 23),np.int8)
    footprint = windows.copy()
    if bad == "state": windows[0,0] = 1
    table = SimpleNamespace(windows=windows,tdd_footprints=footprint,windows_df=pd.DataFrame(index=[10,20]))
    with pytest.raises(ValueError):
        Availability()(table,[99 if bad == "owner" else 10],np.array([99 if bad == "alt" else 0]))
