from microscopes.models import bb
from microscopes.mixture.definition import model_definition
from microscopes.py.mixture.model import initialize as py_initialize
from microscopes.cxx.mixture.model import initialize as cxx_initialize
from microscopes.cxx.common.rng import rng
from microscopes.py.common.recarray.dataview import numpy_dataview as py_numpy_dataview
from microscopes.cxx.common.recarray.dataview import numpy_dataview as cxx_numpy_dataview

import numpy as np

from nose.plugins.attrib import attr

def _test_stress(initialize_fn, dataview, R):
    N = 20
    D = 2
    data = np.random.random(size=(N, D)) < 0.8
    Y = np.array([tuple(y) for y in data], dtype=[('',bool)]*D)
    view = dataview(Y)
    defn = model_definition(N, [bb]*D)

    s = initialize_fn(defn, view, cluster_hp={'alpha':2.0}, r=R)

    CHANGE_GROUP = 1
    CHANGE_VALUE = 2

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
                s.add_value(egid, eid, Y[eid], R)
            else:
                s.remove_value(eid, Y[eid], R)
        s.dcheck_consistency()
        nops -= 1

def test_stress_py():
    _test_stress(py_initialize, py_numpy_dataview, None)

def test_stress_cxx():
    _test_stress(cxx_initialize, cxx_numpy_dataview, rng())
