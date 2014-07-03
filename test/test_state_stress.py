from distributions.dbg.models import bb as py_bb
from microscopes.py.mixture.dp import state as py_state

from microscopes.cxx.models import bb as cxx_bb
from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.common.rng import rng

import numpy as np
import math

from nose.plugins.attrib import attr

def _test_stress(ctor, bbmodel, R):
    N = 20

    s = ctor(N, [bbmodel])
    s.set_cluster_hp({'alpha':2.0})
    s.set_feature_hp(0, {'alpha':1., 'beta':1.})
    s.create_group(R)

    CHANGE_GROUP = 1
    CHANGE_VALUE = 2

    y_value = np.array([(True,)], dtype=[('',bool)])[0]

    nops = 100
    while nops:
        assert len(s.groups()) >= 1
        choice = np.random.choice([CHANGE_GROUP, CHANGE_VALUE])
        if choice == CHANGE_GROUP:
            # remove any empty groups. otherwise, add a new group
            egroups = s.empty_groups()
            if len(egroups) > 1:
                s.delete_group(egroups[0])
            else:
                s.create_group(R)
        else:
            eid = np.random.randint(N)
            if s.assignments()[eid] == -1:
                # add to random group
                egid = np.random.choice(s.groups())
                s.add_value(egid, eid, y_value, R)
            else:
                s.remove_value(eid, y_value, R)
        s.dcheck_consistency()
        nops -= 1

def test_stress_py():
    _test_stress(py_state, py_bb, None)

def test_stress_cxx():
    _test_stress(cxx_state, cxx_bb, rng())

def _test_stress_sampler(ctor, bbmodel, R):
    """
    tries to mimic the low level operations of a gibbs sampler
    """

    from sklearn.datasets import fetch_mldata
    mnist_dataset = fetch_mldata('MNIST original')

    Y_2 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 2.)[0]]
    Y_3 = mnist_dataset['data'][np.where(mnist_dataset['target'] == 3.)[0]]
    _, D = Y_2.shape
    W = int(math.sqrt(D))
    assert W * W == D
    dtype = [('', bool)]*D
    Y = np.vstack([Y_2, Y_3])
    Y = np.array([tuple(y) for y in Y], dtype=dtype)

    N = Y.shape[0]

    s = ctor(N, [bbmodel]*D)
    s.set_cluster_hp({'alpha':2.0})
    s.set_feature_hp(0, {'alpha':1., 'beta':1.})

    for _ in xrange(np.random.randint(5) + 1):
        s.create_group(R)
    groups = s.groups()
    for i, y in enumerate(Y):
        s.add_value(np.random.choice(groups), i, y, R)

    egroups = s.empty_groups()
    for egid in egroups[1:]:
        s.delete_group(egid)
    if not egroups:
        s.create_group(R)
    assert len(s.empty_groups()) == 1

    for i in np.random.permutation(Y.shape[0])[:100]:
        egroups = s.empty_groups()
        assert len(egroups) == 1
        y = Y[i]
        egid = s.remove_value(i, y, R)
        if not s.groupsize(egid):
            s.delete_group(egid)
        choice = np.random.choice(s.groups())
        s.add_value(choice, i, y, R)
        if choice == egroups[0]:
            assert s.groupsize(egroups[0]) == 1
            assert not s.empty_groups()
            s.create_group(R)
        s.dcheck_consistency()

@attr('slow')
def test_stress_sampler_cxx():
    _test_stress_sampler(cxx_state, cxx_bb, rng())
