from microscopes.mixture.definition import model_definition
from microscopes.models import bb, bnb, gp, nich
from microscopes.mixture.model import initialize
from microscopes.common.rng import rng
from microscopes.common.recarray.dataview import numpy_dataview

import numpy as np
from nose.tools import assert_almost_equals


def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        # floats don't have much precision
        assert_almost_equals(v, b[k], places=5)


def assert_lists_almost_equals(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert_almost_equals(x, y, places=5)


def test_get_set_params():
    defn = model_definition(1, [bb, bnb, gp, nich])
    data = np.array([(True, 3, 5, 10.), ],
                    dtype=[('', bool), ('', int), ('', int), ('', float)])
    s = initialize(defn=defn, data=numpy_dataview(data), r=rng())
    s.set_cluster_hp({'alpha': 3.0})
    assert_dict_almost_equals(s.get_cluster_hp(), {'alpha': 3.0})
    hyperparams = [
        {'alpha': 1.2, 'beta': 4.3},
        {'alpha': 1., 'beta': 1., 'r': 1},
        {'alpha': 1., 'inv_beta': 1.},
        {'mu': 30., 'kappa': 1., 'sigmasq': 1., 'nu': 1.},
    ]
    for i, hp in enumerate(hyperparams):
        s.set_feature_hp(i, hp)
        assert_dict_almost_equals(s.get_feature_hp(i), hp)
