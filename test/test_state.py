# test the low level primitive operations

from distributions.dbg.models import (
    bb as dist_bb,
    bnb as dist_bnb,
    nich as dist_nich,
)

from microscopes.models import bb, bbnc, bnb, nich, niw
from microscopes.mixture.definition import (
    model_definition,
    fixed_model_definition,
)
from microscopes.common.rng import rng

from microscopes.mixture.model import (
    initialize as cxx_initialize,
    deserialize as cxx_deserialize,
    bind,
    bind_fixed,
    initialize_fixed,
)

from microscopes.common.recarray.dataview import (
    numpy_dataview as cxx_numpy_dataview,
)

from microscopes.mixture.testutil import toy_dataset

import itertools as it
import numpy as np
import numpy.ma as ma
import pickle
import copy

#from nose.plugins.attrib import attr
from nose.tools import (
    assert_almost_equals,
    assert_equals,
    assert_is_not,
)
from distributions.tests.util import assert_close


def unset(s, data, r):
    for i, yi in enumerate(data):
        s.remove_value(i, yi, r)


def ensure_k_groups(s, k, r):
    groups = sorted(list(s.groups()))
    if len(groups) < k:
        for _ in xrange(k - len(groups)):
            s.create_group(r)
    elif len(groups) > k:
        for gid in groups[k:]:
            s.delete_group(gid)


def test_operations():
    N = 10
    R = rng(12)

    def mkrow():
        return (np.random.choice([False, True]),
                np.random.choice([False, True]),
                np.random.random(),
                np.random.choice([False, True]))
    dtype = [('', bool), ('', bool), ('', float), ('', bool)]
    # non-masked data
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)

    defn = model_definition(N, [bb, bb, nich, bb])
    init_args = {
        'defn': defn,
        'cluster_hp': {'alpha': 2.0},
        'feature_hps': [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
            dist_bb.EXAMPLES[0]['shared'],
        ],
        'r': R,
    }
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    # *_initialize() randomly assigns all entities to a group, so we'll have to
    # unset this assignment for this test
    unset(cxx_s, data, R)

    ensure_k_groups(cxx_s, 3, R)

    assert cxx_s.nentities() == N

    cxx_s.dcheck_consistency()

    assert cxx_s.ngroups() == 3 and set(cxx_s.empty_groups()) == set([0, 1, 2])

    for i, yi in enumerate(data):
        egid = i % 2
        cxx_s.add_value(egid, i, yi, R)
        cxx_s.dcheck_consistency()

    for i, yi in it.islice(enumerate(data), 2):
        cxx_s.remove_value(i, yi, R)
        cxx_s.dcheck_consistency()

    newrow = mkrow()
    newdata = np.array([newrow], dtype=dtype)

    cxx_score = cxx_s.score_value(newdata[0], R)
    assert cxx_score is not None
    cxx_s.dcheck_consistency()


def test_masked_operations():
    N = 10
    R = rng(2347785)

    dtype = [('', bool), ('', int), ('', float)]

    def randombool():
        return np.random.choice([False, True])

    def mkrow():
        return (randombool(), np.random.randint(1, 10), np.random.random())

    def mkmask():
        return (randombool(), randombool(), randombool())
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)
    mask = [mkmask() for _ in xrange(N)]
    data = ma.masked_array(data, mask=mask)

    defn = model_definition(N, [bb, bnb, nich])
    init_args = {
        'defn': defn,
        'cluster_hp': {'alpha': 10.0},
        'feature_hps': [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bnb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
        ],
        'r': R,
    }
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    # see comment above
    unset(cxx_s, data, R)
    ensure_k_groups(cxx_s, 3, R)

    for i, yi in enumerate(data):
        egid = i % 2
        cxx_s.add_value(egid, i, yi, R)
        cxx_s.dcheck_consistency()

    for i, yi in enumerate(data):
        cxx_s.remove_value(i, yi, R)
        cxx_s.dcheck_consistency()


def _test_serializer(initialize_fn, deserialize_fn, dataview):
    N = 10
    R = rng()

    dtype = [('', bool), ('', int), ('', float)]

    def randombool():
        return np.random.choice([False, True])

    def mkrow():
        return (randombool(), np.random.randint(1, 10), np.random.random())

    def mkmask():
        return (randombool(), randombool(), randombool())
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)

    defn = model_definition(N, [bb, bnb, nich])
    init_args = {
        'defn': defn,
        'data': dataview(data),
        'cluster_hp': {'alpha': 10.0},
        'feature_hps': [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bnb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
        ],
        'r': R,
    }
    state = initialize_fn(**init_args)

    raw = state.serialize()

    state1 = deserialize_fn(defn, raw)
    assert state1 is not None

    bstr = pickle.dumps(state)
    state2 = pickle.loads(bstr)
    assert state2 is not None


def test_serializer_cxx():
    _test_serializer(cxx_initialize, cxx_deserialize, cxx_numpy_dataview)


def _assert_copy(s1, s2, bind_fn, view, r):
    assert_equals(s1.nentities(), s2.nentities())
    assert_equals(s1.nfeatures(), s2.nfeatures())
    assert_equals(set(s1.groups()), set(s2.groups()))
    assert_equals(s1.assignments(), s2.assignments())
    for i in xrange(s1.nfeatures()):
        hp1 = s1.get_feature_hp(i)
        hp2 = s2.get_feature_hp(i)
        assert_close(hp1, hp2)
    for gid, fid in it.product(s1.groups(), range(s1.nfeatures())):
        ss1 = s1.get_suffstats(gid, fid)
        ss2 = s2.get_suffstats(gid, fid)
        assert_close(ss1, ss2)
    assert_almost_equals(s1.score_assignment(),
                         s2.score_assignment())
    assert_almost_equals(s1.score_data(None, None, r),
                         s2.score_data(None, None, r))
    before = list(s1.assignments())
    gid = bind_fn(s1, view).remove_value(0, r)
    assert_equals(s1.assignments()[0], -1)
    assert_equals(before, s2.assignments())
    bind_fn(s1, view).add_value(gid, 0, r)  # restore s1


def _test_copy_state(defn, initialize_fn, bind_fn):
    Y = toy_dataset(defn)
    view = cxx_numpy_dataview(Y)
    r = rng()
    state = initialize_fn(defn, view, r)
    state_shallow = copy.copy(state)
    state_deep = copy.deepcopy(state)
    assert_is_not(state, state_shallow)
    assert_is_not(state, state_deep)
    _assert_copy(state, state_shallow, bind_fn, view, r)
    _assert_copy(state, state_deep, bind_fn, view, r)


def test_copy_state():
    defn = model_definition(10, [bb, niw(3)])
    _test_copy_state(defn, cxx_initialize, bind)


def test_copy_fixed_state():
    defn = fixed_model_definition(10, 3, [bb, niw(3)])
    _test_copy_state(defn, initialize_fixed, bind_fixed)


def test_copy_state_bbnc():
    defn = model_definition(10, [bbnc])
    _test_copy_state(defn, cxx_initialize, bind)


def test_sample_post_pred():
    N = 10
    R = rng(5483932)
    D = 4

    def randombool():
        return np.random.choice([False, True])

    def mkrow():
        return tuple(randombool() for _ in xrange(D))
    dtype = [('', bool)] * D
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)

    defn = model_definition(N, [bb] * D)
    init_args = {
        'defn': defn,
        'cluster_hp': {'alpha': 2.0},
        'feature_hps': [dist_bb.EXAMPLES[0]['shared']] * D,
        'r': R,
    }
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    G = 3
    unset(cxx_s, data, R)
    ensure_k_groups(cxx_s, 3, R)

    for i, yi in enumerate(data):
        egid = i % G
        cxx_s.add_value(egid, i, yi, R)

    # sample
    y_new_data = mkrow()
    y_new_mask = tuple(randombool() for _ in xrange(D))
    y_new = ma.masked_array(
        np.array([y_new_data], dtype=dtype),
        mask=[y_new_mask])[0]

    n_samples = 1000

    cxx_samples = np.hstack(
        [cxx_s.sample_post_pred(y_new, R)[1] for _ in xrange(n_samples)])

    idmap = {C: i for i, C in enumerate(it.product([False, True], repeat=D))}

    def todist(samples):
        dist = np.zeros(len(idmap))
        for s in samples:
            dist[idmap[tuple(s)]] += 1.0
        dist /= dist.sum()
        return dist

    cxx_dist = todist(cxx_samples)
    assert cxx_dist is not None

    # XXX(stephentu):
    # when we had python models, we used to compare the posterior
    # sample distribution between the python and C++ models. now
    # we don't do anything useful with the samples
