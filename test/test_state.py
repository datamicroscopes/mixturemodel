# test the low level primitive operations

from distributions.dbg.models import bb as py_bb
from distributions.dbg.models import nich as py_nich
from microscopes.py.mixture.dp import state as py_state

from microscopes.cxx.models import bb as cxx_bb
from microscopes.cxx.models import nich as cxx_nich
from microscopes.cxx.mixture.model import state as cxx_state
from microscopes.cxx.common.rng import rng

import itertools as it
import numpy as np

from nose.tools import assert_almost_equals

def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5) # floats don't have much precision

def test_operations():
    N = 10
    R = rng(12)

    py_s = py_state(N, [py_bb, py_bb, py_nich, py_bb])
    py_s.set_cluster_hp({'alpha':2.0})
    py_s.set_feature_hp(0, py_bb.EXAMPLES[0]['shared'])
    py_s.set_feature_hp(1, py_bb.EXAMPLES[0]['shared'])
    py_s.set_feature_hp(2, py_nich.EXAMPLES[0]['shared'])
    py_s.set_feature_hp(3, py_bb.EXAMPLES[0]['shared'])

    cxx_s = cxx_state(N, [cxx_bb, cxx_bb, cxx_nich, cxx_bb])
    cxx_s.set_cluster_hp({'alpha':2.0})
    cxx_s.set_feature_hp(0, py_bb.EXAMPLES[0]['shared'])
    cxx_s.set_feature_hp(1, py_bb.EXAMPLES[0]['shared'])
    cxx_s.set_feature_hp(2, py_nich.EXAMPLES[0]['shared'])
    cxx_s.set_feature_hp(3, py_bb.EXAMPLES[0]['shared'])

    assert py_s.nentities() == N
    assert cxx_s.nentities() == N

    # non-masked data
    data = [(
        np.random.choice([False, True]),
        np.random.choice([False, True]),
        np.random.random(),
        np.random.choice([False, True])) for _ in xrange(N)]
    data = np.array(data, dtype=[('',bool), ('',bool), ('',float), ('',bool)])

    py_egid = py_s.create_group()
    assert py_egid == 0
    py_egid = py_s.create_group()
    assert py_egid == 1

    cxx_egid = cxx_s.create_group(R)
    assert cxx_egid == 0
    cxx_egid = cxx_s.create_group(R)
    assert cxx_egid == 1

    assert py_s.ngroups() == 2 and set(py_s.empty_groups()) == set([0, 1])
    assert cxx_s.ngroups() == 2 and set(cxx_s.empty_groups()) == set([0, 1])

    for i, yi in enumerate(data):
        egid = i % 2
        py_s.add_value(egid, i, yi)
        cxx_s.add_value(egid, i, yi, R)

    def assert_suff_stats_equal():
        for fid, gid in it.product(range(4), range(2)):
            py_ss = py_s.get_suff_stats(gid, fid)
            cxx_ss = cxx_s.get_suff_stats(gid, fid)
            assert_dict_almost_equals(py_ss, cxx_ss)

    assert_suff_stats_equal()

    for i, yi in it.islice(enumerate(data), 2):
        py_s.remove_value(i, yi)
        cxx_s.remove_value(i, yi, R)

    assert_suff_stats_equal()
