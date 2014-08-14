from microscopes.mixture.model import initialize
from microscopes.common.recarray.dataview import numpy_dataview

from microscopes.common.rng import rng
from microscopes.models import bb
from microscopes.mixture.definition import model_definition
from distributions.dbg.random import sample_discrete

from microscopes.common.testutil import (
    permutation_iter,
    permutation_canonical,
    assert_discrete_dist_approx,
    scores_to_probs,
)

from nose.tools import assert_almost_equals
#from nose.plugins.attrib import attr

import numpy as np


def _sample_crp(n, alpha):
    """
    generate an assignment vector of length n from a CRP with alpha
    """
    if n <= 0:
        raise ValueError("need positive n")
    if alpha <= 0.:
        raise ValueError("need positive alpha")
    counts = np.array([1])
    assignments = np.zeros(n, dtype=np.int)
    assignments[0] = 0
    for i in xrange(1, n):
        dist = np.append(counts, alpha).astype(np.float, copy=False)
        dist /= dist.sum()
        choice = sample_discrete(dist)
        if choice == len(counts):
            # new cluster
            counts = np.append(counts, 1)
        else:
            # existing cluster
            counts[choice] += 1
        assignments[i] = choice
    return assignments


def _test_crp(initialize_fn, dataview, alpha, r):
    N = 6
    defn = model_definition(N, [bb])
    Y = np.array([(True,)] * N, dtype=[('', bool)])
    view = dataview(Y)

    def crp_score(assignment):
        latent = initialize_fn(
            defn, view, r=r,
            cluster_hp={'alpha': alpha}, assignment=assignment)
        return latent.score_assignment()
    dist = np.array(list(map(crp_score, permutation_iter(N))))
    dist = np.exp(dist)
    assert_almost_equals(dist.sum(), 1.0, places=3)


def test_crp():
    for alpha in (0.1, 1.0, 10.0):
        _test_crp(initialize, numpy_dataview, alpha=alpha, r=rng())


def test_crp_empirical():
    N = 4
    alpha = 2.5
    defn = model_definition(N, [bb])
    Y = np.array([(True,)] * N, dtype=[('', bool)])
    view = numpy_dataview(Y)
    r = rng()

    def crp_score(assignment):
        latent = initialize(
            defn, view, r=r,
            cluster_hp={'alpha': alpha}, assignment=assignment)
        return latent.score_assignment()
    scores = np.array(list(map(crp_score, permutation_iter(N))))
    dist = scores_to_probs(scores)
    idmap = {C: i for i, C in enumerate(permutation_iter(N))}

    def sample_fn():
        sample = permutation_canonical(_sample_crp(N, alpha))
        return idmap[tuple(sample)]
    assert_discrete_dist_approx(sample_fn, dist, ntries=100)
