import numpy as np
import pytest
from choiceforge.phase59_entity_store import EntityStore
from choiceforge.cuda_backend import _cupy


def test_reuse_mutation_and_layout_invalidation():
    store = EntityStore()
    values = np.array([4, 8, 16])
    lease = store.publish("trips", [10, 20, 30], {"depart":values})
    first = lease.arrays()[0]
    store.publish("trips", [10, 20, 30], {"depart":values.copy()})
    assert lease.arrays()[0] is first
    values[0] = 7
    np.testing.assert_array_equal(_cupy().asnumpy(first), [4, 8, 16])
    newer = store.publish("trips", [10, 20, 30], {"depart":values})
    with pytest.raises(ValueError, match="stale"):
        lease.arrays()
    store.publish("trips", [30, 10], {"depart":np.array([16, 7])})
    with pytest.raises(ValueError, match="stale"):
        newer.arrays()
    np.testing.assert_array_equal(_cupy().asnumpy(store.gather("trips", [10, 30], ["depart"])[0]), [7, 16])
    with pytest.raises(ValueError, match="unknown"):
        store.gather("trips", [20], ["depart"])


def test_identity_and_schema_fail_closed():
    store = EntityStore()
    with pytest.raises(ValueError, match="unique"):
        store.publish("trips", [1, 1], {"mode":np.array([0, 1])})
    with pytest.raises(ValueError, match="aligned"):
        store.publish("trips", [1, 2], {"mode":np.array([0])})
    store.publish("trips", [1, 2], {"mode":np.array([0, 1])})
    with pytest.raises(ValueError, match="missing"):
        store.lease("trips", ["depart"])


def test_signed_zero_identity_copy_and_explicit_invalidation():
    store = EntityStore()
    ids = np.array([1, 2])
    lease = store.publish("trips", ids, {"x":np.array([0., 1.])})
    ids[0] = 99
    np.testing.assert_array_equal(store.tables["trips"]["index"], [1, 2])
    newer = store.publish("trips", [1, 2], {"x":np.array([-0., 1.])})
    with pytest.raises(ValueError, match="stale"):
        lease.arrays()
    assert np.signbit(_cupy().asnumpy(newer.arrays()[0])[0])
    store.invalidate("trips")
    with pytest.raises(ValueError, match="stale"):
        newer.arrays()
