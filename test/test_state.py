# test the low level primitive operations

from distributions.util import scores_to_probs
from distributions.dbg.models import \
    bb as dist_bb, \
    bnb as dist_bnb, \
    nich as dist_nich

from microscopes.models import bb, bnb, nich
from microscopes.mixture.definition import model_definition
from microscopes.cxx.common.rng import rng

from microscopes.py.mixture.model import \
    initialize as py_initialize, \
    deserialize as py_deserialize
from microscopes.cxx.mixture.model import \
    initialize as cxx_initialize, \
    deserialize as cxx_deserialize

from microscopes.py.common.recarray.dataview import \
    numpy_dataview as py_numpy_dataview
from microscopes.cxx.common.recarray.dataview import \
    numpy_dataview as cxx_numpy_dataview

import itertools as it
import numpy as np
import numpy.ma as ma

from nose.plugins.attrib import attr
from nose.tools import assert_almost_equals

def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5) # floats don't have much precision

def assert_1darray_almst_equals(a, b, places=5):
    assert len(a.shape) == 1
    assert a.shape[0] == b.shape[0]
    for x, y in zip(a, b):
        assert_almost_equals(x, y, places=places)

def assert_suff_stats_equal(py_s, cxx_s, features, groups):
    for fid, gid in it.product(features, groups):
        py_ss = py_s.get_suffstats(gid, fid)
        cxx_ss = cxx_s.get_suffstats(gid, fid)
        assert_dict_almost_equals(py_ss, cxx_ss)

def unset(s, data, r):
    for i, yi in enumerate(data):
        s.remove_value(i, yi, r)

def ensure_k_groups(s, k, r):
    groups = sorted(list(s.groups()))
    if len(groups) < k:
        for _ in xrange(k-len(groups)):
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
    dtype = [('',bool), ('',bool), ('',float), ('',bool)]
    # non-masked data
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)

    defn = model_definition(N, [bb, bb, nich, bb])
    init_args = {
        'defn' : defn,
        'cluster_hp' : {'alpha':2.0},
        'feature_hps' : [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
            dist_bb.EXAMPLES[0]['shared'],
        ],
        'r' : R,
    }
    py_s = py_initialize(data=py_numpy_dataview(data), **init_args)
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    # *_initialize() randomly assigns all entities to a group, so we'll have to
    # unset this assignment for this test
    unset(py_s, data, R)
    unset(cxx_s, data, R)

    ensure_k_groups(py_s, 3, R)
    ensure_k_groups(cxx_s, 3, R)

    assert py_s.nentities() == N
    assert cxx_s.nentities() == N

    py_s.dcheck_consistency()
    cxx_s.dcheck_consistency()

    assert py_s.ngroups() == 3 and set(py_s.empty_groups()) == set([0, 1, 2])
    assert cxx_s.ngroups() == 3 and set(cxx_s.empty_groups()) == set([0, 1, 2])

    for i, yi in enumerate(data):
        egid = i % 2
        py_s.add_value(egid, i, yi)
        py_s.dcheck_consistency()
        cxx_s.add_value(egid, i, yi, R)
        cxx_s.dcheck_consistency()

    assert_suff_stats_equal(py_s, cxx_s, features=range(4), groups=range(2))

    assert_almost_equals(py_s.score_joint(), cxx_s.score_joint(R), places=2)

    for i, yi in it.islice(enumerate(data), 2):
        py_s.remove_value(i, yi)
        py_s.dcheck_consistency()
        cxx_s.remove_value(i, yi, R)
        cxx_s.dcheck_consistency()

    assert_suff_stats_equal(py_s, cxx_s, features=range(4), groups=range(2))

    newrow = mkrow()
    newdata = np.array([newrow], dtype=dtype)

    py_score = py_s.score_value(newdata[0])
    cxx_score = cxx_s.score_value(newdata[0], R)

    # XXX: technically this need not be true, but it is true for our implementations
    assert py_score[0] == cxx_score[0]

    # the scores won't be that close since the python one uses double precision
    # whereas the c++ one uses single precision
    assert_1darray_almst_equals(
        scores_to_probs(py_score[1]),
        scores_to_probs(cxx_score[1]), places=2)

    py_score = py_s.score_data(None, None)
    cxx_score = cxx_s.score_data(None, None, R)
    assert_almost_equals(py_score, cxx_score, places=2)

    py_score = py_s.score_data(0, 0)
    cxx_score = cxx_s.score_data(0, 0, R)
    assert_almost_equals(py_score, cxx_score, places=2)

    py_score = py_s.score_data([0,1], [0])
    cxx_score = cxx_s.score_data([0,1], [0], R)
    assert_almost_equals(py_score, cxx_score, places=2)

    py_score = py_s.score_data(np.array([0,1], dtype=np.int)[0], [0])
    cxx_score = cxx_s.score_data(np.array([0,1], dtype=np.int)[0], [0], R)
    assert_almost_equals(py_score, cxx_score, places=2)

    py_s.dcheck_consistency()
    cxx_s.dcheck_consistency()

def test_masked_operations():
    N = 10
    R = rng(2347785)

    dtype = [('',bool), ('',int), ('',float)]
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
        'defn' : defn,
        'cluster_hp' : {'alpha':10.0},
        'feature_hps': [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bnb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
        ],
        'r' : R,
    }
    py_s = py_initialize(data=py_numpy_dataview(data), **init_args)
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    # see comment above
    unset(py_s, data, R)
    unset(cxx_s, data, R)
    ensure_k_groups(py_s, 3, R)
    ensure_k_groups(cxx_s, 3, R)

    for i, yi in enumerate(data):
        egid = i % 2
        py_s.add_value(egid, i, yi)
        cxx_s.add_value(egid, i, yi, R)
        py_s.dcheck_consistency()
        cxx_s.dcheck_consistency()

    assert_suff_stats_equal(py_s, cxx_s, features=range(3), groups=range(3))

    for i, yi in enumerate(data):
        py_s.remove_value(i, yi)
        cxx_s.remove_value(i, yi, R)
        py_s.dcheck_consistency()
        cxx_s.dcheck_consistency()

    assert_suff_stats_equal(py_s, cxx_s, features=range(3), groups=range(3))

def _test_serializer(initialize_fn, deserialize_fn, dataview):
    N = 10
    R = rng()

    dtype = [('',bool), ('',int), ('',float)]
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
        'defn' : defn,
        'data' : dataview(data),
        'cluster_hp' : {'alpha':10.0},
        'feature_hps': [
            dist_bb.EXAMPLES[0]['shared'],
            dist_bnb.EXAMPLES[0]['shared'],
            dist_nich.EXAMPLES[0]['shared'],
        ],
        'r' : R,
    }
    state = initialize_fn(**init_args)

    raw = state.serialize()

    state1 = deserialize_fn(defn, raw)


@attr('wip')
def test_serializer_py():
    _test_serializer(py_initialize, py_deserialize, py_numpy_dataview)

@attr('wip')
def test_serializer_cxx():
    _test_serializer(cxx_initialize, cxx_deserialize, cxx_numpy_dataview)

def test_sample_post_pred():
    N = 10
    R = rng(5483932)
    D = 4

    def randombool():
        return np.random.choice([False, True])
    def mkrow():
        return tuple(randombool() for _ in xrange(D))
    dtype = [('',bool)]*D
    data = [mkrow() for _ in xrange(N)]
    data = np.array(data, dtype=dtype)

    defn = model_definition(N, [bb]*D)
    init_args = {
        'defn' : defn,
        'cluster_hp' : {'alpha':2.0},
        'feature_hps': [dist_bb.EXAMPLES[0]['shared']]*D,
        'r' : R,
    }
    py_s = py_initialize(data=py_numpy_dataview(data), **init_args)
    cxx_s = cxx_initialize(data=cxx_numpy_dataview(data), **init_args)

    G = 3
    unset(py_s, data, R)
    unset(cxx_s, data, R)
    ensure_k_groups(py_s, 3, R)
    ensure_k_groups(cxx_s, 3, R)

    for i, yi in enumerate(data):
        egid = i % G
        py_s.add_value(egid, i, yi)
        cxx_s.add_value(egid, i, yi, R)

    assert_suff_stats_equal(py_s, cxx_s, features=range(D), groups=range(G))

    # sample
    y_new_data = mkrow()
    y_new_mask = [randombool() for _ in xrange(D)]
    y_new = ma.masked_array(
        np.array([y_new_data], dtype=dtype),
        mask=y_new_data)[0]

    n_samples = 1000

    py_samples = np.hstack([py_s.sample_post_pred(y_new)[1] for _ in xrange(n_samples)])
    cxx_samples = np.hstack([cxx_s.sample_post_pred(y_new, R)[1] for _ in xrange(n_samples)])

    idmap = { C : i for i, C in enumerate(it.product([False,True], repeat=D)) }
    def todist(samples):
        dist = np.zeros(len(idmap))
        for s in samples:
            dist[idmap[tuple(s)]] += 1.0
        dist /= dist.sum()
        return dist

    py_dist = todist(py_samples)
    cxx_dist = todist(cxx_samples)

    assert_1darray_almst_equals(py_dist, cxx_dist, places=1)
