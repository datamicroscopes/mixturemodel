from microscopes.cxx.models import bb, bnb, gp, nich
from microscopes.cxx.mixture.model import state, fixed_state

import numpy as np

from nose.tools import assert_almost_equals

def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5) # floats don't have much precision

def assert_lists_almost_equals(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert_almost_equals(x, y, places=5)

def test_get_set_params():
    s = state(10, [bb, bnb, gp, nich])
    s.set_cluster_hp({'alpha':3.0})
    assert_dict_almost_equals(s.get_cluster_hp(), {'alpha':3.0})
    hyperparams = [
        {'alpha':1.2, 'beta':4.3},
        {'alpha': 1., 'beta': 1., 'r': 1},
        {'alpha': 1., 'inv_beta': 1.},
        {'mu': 30., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
    ]
    for i, hp in enumerate(hyperparams):
        s.set_feature_hp(i, hp)
        assert_dict_almost_equals(s.get_feature_hp(i), hp)

def test_get_set_params_fixed():
    s = fixed_state(10, 5, [bb, bnb, gp, nich])
    s.set_cluster_hp({'alphas': np.array([0.1, 0.2, 0.3, 0.4, 0.5])})
    assert_lists_almost_equals(s.get_cluster_hp()['alphas'], [0.1, 0.2, 0.3, 0.4, 0.5])
