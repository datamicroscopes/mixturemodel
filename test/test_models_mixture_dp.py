from distributions.dbg.models import bb as py_bb
from microscopes.py.mixture.model import state as py_state, fill as py_fill, sample as py_sample
from microscopes.cxx.models import bb as cxx_bb
from microscopes.cxx.common.rng import rng
from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.py.common.util import KL_discrete, logsumexp

import numpy as np
import numpy.ma as ma
import itertools as it

from nose.plugins.attrib import attr
from nose.tools import assert_almost_equals

N, D = 1000, 5

def _test_sample_post_pred(ctor, bbtype, y_new, r):
    alpha = 2.0
    s = ctor(N, [bbtype]*D)
    s.set_cluster_hp({'alpha':alpha})
    for i in xrange(D):
        s.set_feature_hp(i, {'alpha':1.,'beta':1.})

    py_proto = py_state(N, [py_bb]*D)
    py_proto.set_cluster_hp(s.get_cluster_hp())
    for i in xrange(D):
        py_proto.set_feature_hp(i, s.get_feature_hp(i))

    Y_clustered, _ = py_sample(N, py_proto)
    Y = np.hstack(Y_clustered)
    assert Y.shape[0] == N

    py_fill(s, Y_clustered, r)

    n_samples = 10000
    Y_samples = [s.sample_post_pred(None, r)[1] for _ in xrange(n_samples)]
    Y_samples = np.hstack(Y_samples)

    empty_groups = list(s.empty_groups())
    if len(empty_groups):
        for egid in empty_groups[1:]:
            s.delete_group(egid)
    else:
        s.create_group(r)
    assert len(s.empty_groups()) == 1

    def score_post_pred(y):
        # XXX: the C++ API can only handle structural arrays for now
        y = np.array([y], dtype=[('',bool)]*D)[0]
        _, scores = s.score_value(y, r)
        return logsumexp(scores)

    scores = np.array(list(map(score_post_pred, it.product([False, True], repeat=D))))
    scores = np.exp(scores)
    assert_almost_equals(scores.sum(), 1.0, places=3)

    # lazy man
    idmap = { y : i for i, y in enumerate(it.product([False, True], repeat=D)) }

    smoothing = 1e-5
    sample_hist = np.zeros(len(idmap), dtype=np.int)
    for y in Y_samples:
        sample_hist[idmap[tuple(y)]] += 1.

    sample_hist = np.array(sample_hist, dtype=np.float) + smoothing
    sample_hist /= sample_hist.sum()

    #print 'actual', scores
    #print 'emp', sample_hist
    kldiv = KL_discrete(scores, sample_hist)
    print 'KL:', kldiv

    assert kldiv <= 0.005

def test_py_sample_post_pred_no_given_data():
    _test_sample_post_pred(py_state, py_bb, None, None)

def test_py_sample_post_pred_given_data():
    assert D == 5
    y_new = ma.masked_array(
        np.array([(True, False, True, True, True)], dtype=[('', np.bool)]*5),
        mask=[(False, False, True, True, True)])[0]
    _test_sample_post_pred(py_state, py_bb, y_new, None)

def test_cxx_sample_post_pred_no_given_data():
    _test_sample_post_pred(cxx_state, cxx_bb, None, rng(7589))

def test_cxx_sample_post_pred_given_data():
    assert D == 5
    y_new = ma.masked_array(
        np.array([(True, False, True, True, True)], dtype=[('', np.bool)]*5),
        mask=[(False, False, True, True, True)])[0]
    _test_sample_post_pred(cxx_state, cxx_bb, None, rng(543234))
