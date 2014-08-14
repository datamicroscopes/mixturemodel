from microscopes.mixture.model import sample
from microscopes.mixture.definition import model_definition
from microscopes.models import (
    bb,
    bnb,
    gp,
    nich,
    dd,
    niw,
)

from nose.tools import assert_equals, assert_true


def test_sample_sanity():
    # just a sanity check
    defn = model_definition(10, [bb, bnb, gp, nich, dd(5), niw(4)])
    clusters, samplers = sample(defn)
    assert_equals(len(clusters), len(samplers))
    for cluster in clusters:
        assert_true(len(cluster) > 0)
        for v in cluster:
            assert_equals(len(v), len(defn.models()))
