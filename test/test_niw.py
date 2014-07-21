# XXX: this is here, and not in common since we currently don't have a python
# API to interface with our likelihood models (mixturemodel acts as that API)
#
# XXX: fix this!

from microscopes.models import niw
from microscopes.mixture.definition import model_definition

from microscopes.py.mixture.model import initialize as py_initialize
from microscopes.py.common.recarray.dataview import numpy_dataview as py_numpy_dataview
from microscopes.py.common.util import random_orthonormal_matrix

from microscopes.cxx.mixture.model import initialize as cxx_initialize
from microscopes.cxx.common.recarray.dataview import numpy_dataview as cxx_numpy_dataview
from microscopes.cxx.common.rng import rng

import numpy as np

from nose.tools import assert_almost_equals

# XXX: don't copy the assertion code from test_state.py
def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5)

def assert_1darray_almost_equals(a, b, places=5):
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

    Y = np.array([
            ([2., -2., 4.],),
            ([3., -1., -0.99],),
            ([-45., -3., 6.7],)
        ], dtype=[('', np.float, (3,))])
    py_view, cxx_view = py_numpy_dataview(Y), cxx_numpy_dataview(Y)

    r = rng()
    defn = model_definition([niw(3)])
    init_args = {
        'cluster_hp':{'alpha':1.},
        'feature_hps':[{'mu0':mu0,'lambda':lam,'psi':psi,'nu':nu}],
        'assignment': [0,0,0],
        'r':r,
    }

    py_s = py_initialize(defn, py_view, **init_args)
    cxx_s = cxx_initialize(defn, cxx_view, **init_args)

    # params
    py_hp = py_s.get_feature_hp(0)
    cxx_hp = cxx_s.get_feature_hp(0)
    assert_1darray_almost_equals(py_hp['mu0'], cxx_hp['mu0'])
    assert_almost_equals(py_hp['lambda'], cxx_hp['lambda'])
    assert_1darray_almost_equals(py_hp['psi'].flatten(), cxx_hp['psi'].flatten())
    assert_almost_equals(py_hp['nu'], cxx_hp['nu'])

    # score_data
    assert_almost_equals(py_s.score_data(None, None), cxx_s.score_data(None, None, r), places=3)

    # score value
    py_s.create_group()
    cxx_s.create_group(r)
    py_s.remove_value(1, Y[1], r)
    cxx_s.remove_value(1, Y[1], r)
    _, py_scores = py_s.score_value(Y[1])
    _, cxx_scores = cxx_s.score_value(Y[1], r)
    assert_1darray_almost_equals(py_scores, cxx_scores, places=3)
