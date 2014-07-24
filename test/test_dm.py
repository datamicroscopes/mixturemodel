# XXX: this is here, and not in common since we currently don't have a python
# API to interface with our likelihood models (mixturemodel acts as that API)
#
# XXX: fix this!

from microscopes.models import dm
from microscopes.mixture.definition import model_definition

from microscopes.cxx.mixture.model import initialize as cxx_initialize
from microscopes.cxx.common.recarray.dataview import numpy_dataview as cxx_numpy_dataview
from microscopes.cxx.common.rng import rng

import numpy as np

from nose.tools import assert_almost_equals, assert_sequence_equal
from scipy.special import gammaln

def score_multinomial(x, probs):
    assert len(x) == len(probs)
    term1 = gammaln(x.sum() + 1)
    term2 = np.sum(gammaln(x + 1))
    term3 = np.sum(x * np.log(probs))
    return term1 - term2 + term3

def score_data(Y, alphas):
    alphas = np.array(alphas)
    term1 = gammaln(alphas.sum())
    term2 = -np.sum(gammaln(alphas.sum() + Y.flatten().sum()))
    term3 = np.sum(gammaln(alphas + Y.sum(axis=0)))
    term4 = -np.sum(gammaln(alphas))
    term5 = np.sum(gammaln(Y.sum(axis=1) + 1))
    term6 = -gammaln(Y + 1).flatten().sum()
    return term1+term2+term3+term4+term5+term6

def test_dm_cxx():
    K = 4
    Y = np.array([
            ([0, 1, 2, 5],),
            ([1, 0, 1, 2],),
            ([0, 2, 9, 9],),
        ], dtype=[('', np.int, (K,))])
    Y_np = np.vstack(y[0] for y in Y)

    cxx_view = cxx_numpy_dataview(Y)
    r = rng()
    defn = model_definition(Y.shape[0], [dm(K)])
    prior = {'alphas': [1.]*K}
    cxx_s = cxx_initialize(
        defn,
        cxx_view,
        r,
        feature_hps=[prior],
        assignment=[0]*Y.shape[0])

    counts = cxx_s.get_suffstats(0, 0)['counts']
    assert_sequence_equal(counts, list(Y_np.sum(axis=0)))
    pseudocounts = np.array(counts, dtype=float) + np.array(prior['alphas'], dtype=float)
    probs = pseudocounts / pseudocounts.sum()

    for i in xrange(Y.shape[0]):
        _, (cxx_score,) = cxx_s.score_value(Y[i], r)
        true_score = score_multinomial(Y_np[i], probs)
        assert_almost_equals(cxx_score, true_score, places=3)

    assert_almost_equals(score_data(Y_np, prior['alphas']), cxx_s.score_data([], [], r), places=3)
