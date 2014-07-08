# XXX: this is here, and not in common since we currently don't have a python
# API to interface with our likelihood models (mixturemodel acts as that API)
#
# XXX: fix this!

from microscopes.py.models import niw as py_niw
from microscopes.py.mixture.dp import state as py_state
from microscopes.py.common.recarray.dataview import numpy_dataview as py_numpy_dataview
from microscopes.py.common.util import random_orthonormal_matrix

from microscopes.cxx.models import niw as cxx_niw
from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.common.recarray.dataview import numpy_dataview as cxx_numpy_dataview
from microscopes.cxx.common.rng import rng

import numpy as np

from nose.tools import assert_almost_equals

# XXX: don't copy the assertion code from test_state.py
def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5)

def assert_1darray_almst_equals(a, b, places=5):
    assert len(a.shape) == 1
    assert a.shape[0] == b.shape[0]
    for x, y in zip(a, b):
        assert_almost_equals(x, y, places=places)

def assert_suff_stats_equal(py_s, cxx_s, features, groups):
    for fid, gid in it.product(features, groups):
        py_ss = py_s.get_suff_stats(gid, fid)
        cxx_ss = cxx_s.get_suff_stats(gid, fid)
        assert_dict_almost_equals(py_ss, cxx_ss)

def test_niw_compare_to_py():

    Q = random_orthonormal_matrix(3)
    nu = 6
    psi = np.dot(Q, np.dot(np.diag([1.0, 0.5, 0.2]), Q.T))
    lam = 0.3
    mu0 = np.ones(3)

    N = 10

    r = rng()
    py_s = py_state(N, [py_niw])
    py_s.set_cluster_hp({'alpha':1.})
    py_s.set_feature_hp(0, {'mu0':mu0,'lambda':lam,'psi':psi,'nu':nu})

    cxx_s = cxx_state(N, [cxx_niw])
    cxx_s.set_cluster_hp({'alpha':1.})
    cxx_s.set_feature_hp(0, {'mu0':mu0,'lambda':lam,'psi':psi,'nu':nu})

    py_hp = py_s.get_feature_hp(0)
    cxx_hp = cxx_s.get_feature_hp(0)

    assert_1darray_almst_equals(py_hp['mu0'], cxx_hp['mu0'])
    assert_almost_equals(py_hp['lambda'], cxx_hp['lambda'])
    assert_1darray_almst_equals(py_hp['psi'].flatten(), cxx_hp['psi'].flatten())
    assert_almost_equals(py_hp['nu'], cxx_hp['nu'])

    py_s.create_group()
    py_s.create_group()
    cxx_s.create_group(r)
    cxx_s.create_group(r)

    Y = np.array([
        ([2., -2., 4.],),
        ([3., -1., -0.99],),
        ([-45., -3., 6.7],)
    ], dtype=[('', np.float, (3,))])
    py_s.add_value(0, 0, Y[0])
    cxx_s.add_value(0, 0, Y[0], r)

    _, py_scores = py_s.score_value(Y[1])
    _, cxx_scores = cxx_s.score_value(Y[1], r)

    assert_1darray_almst_equals(py_scores, cxx_scores, places=3)

    py_s.add_value(0, 1, Y[1])
    cxx_s.add_value(0, 1, Y[1], r)

    assert_almost_equals(py_s.score_data(None, None), cxx_s.score_data(None, None, r), places=3)

    py_s.add_value(0, 2, Y[2])
    cxx_s.add_value(0, 2, Y[2], r)

    assert_almost_equals(py_s.score_data(None, None), cxx_s.score_data(None, None, r), places=3)

    py_s.remove_value(0, Y[0])
    cxx_s.remove_value(0, Y[0], r)

    assert_almost_equals(py_s.score_data(None, None), cxx_s.score_data(None, None, r), places=3)
