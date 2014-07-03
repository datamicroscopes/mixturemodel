from distributions.dbg.models import bb as py_bb
from microscopes.cxx.models import bb as cxx_bb

from microscopes.py.mixture.dp import state as py_state, sample, fill
from microscopes.cxx.mixture.model import state as cxx_state

from microscopes.cxx.common.rng import rng

from nose.tools import assert_almost_equals

import itertools as it
import numpy as np

def assert_dict_almost_equals(a, b):
    for k, v in a.iteritems():
        assert k in b
        assert_almost_equals(v, b[k], places=5) # floats don't have much precision

def _test_sample_fill(a_ctor, a_bbmodel, b_ctor, b_bbmodel, R):
    N = 10

    def create():
        def init(s):
            s.set_cluster_hp({'alpha':2.})
            s.set_feature_hp(0, {'alpha':0.3,'beta':1.2})
        a_s = a_ctor(N, [a_bbmodel])
        init(a_s)
        b_s = b_ctor(N, [b_bbmodel])
        init(b_s)
        return a_s, b_s

    def dotest(a_s, b_s):
        Y_samples, _ = sample(N, a_s, R)
        fill(a_s, Y_samples, R)
        fill(b_s, Y_samples, R)

        # XXX: not really a requirement, just our current
        # implementations have this property
        assert set(a_s.groups()) == set(b_s.groups())

        for fid, gid in it.product([0], a_s.groups()):
            a_ss = a_s.get_suff_stats(gid, fid)
            b_ss = b_s.get_suff_stats(gid, fid)
            assert_dict_almost_equals(a_ss, b_ss)

    a_s, b_s = create()
    dotest(a_s, b_s)

    a_s, b_s = create()
    dotest(b_s, a_s)

def test_sample_fill():
    _test_sample_fill(py_state, py_bb, cxx_state, cxx_bb, rng())
