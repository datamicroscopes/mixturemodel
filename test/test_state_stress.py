from distributions.dbg.models import bb as py_bb
from microscopes.py.mixture.dp import state as py_state

from microscopes.cxx.models import bb as cxx_bb
from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.common.rng import rng

import numpy as np

def _test_stress(ctor, bbmodel, R):
    N = 20

    s = ctor(N, [bbmodel])
    s.set_cluster_hp({'alpha':2.0})
    s.set_feature_hp(0, {'alpha':1., 'beta':1.})
    s.create_group(R)

    CHANGE_GROUP = 1
    CHANGE_VALUE = 2

    y_value = np.array([(True,)], dtype=[('',bool)])[0]

    nops = 10000
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
