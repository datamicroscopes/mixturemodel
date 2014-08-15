# XXX: this is here, and not in common since we currently don't have a python
# API to interface with our likelihood models (mixturemodel acts as that API)
#
# XXX: fix this!

from microscopes.models import dm
from microscopes.mixture.definition import model_definition

from microscopes.mixture.model import initialize as cxx_initialize
from microscopes.common.recarray.dataview import (
    numpy_dataview as cxx_numpy_dataview,
)
from microscopes.common.rng import rng

import numpy as np
import itertools as it

from nose.tools import (
    assert_almost_equals,
    assert_sequence_equal,
    assert_equals,
)
from scipy.special import gammaln


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
    prior = {'alphas': [1.] * K}
    cxx_s = cxx_initialize(
        defn,
        cxx_view,
        r,
        feature_hps=[prior],
        assignment=[0] * Y.shape[0])

    counts = cxx_s.get_suffstats(0, 0)['counts']
    assert_sequence_equal(counts, list(Y_np.sum(axis=0)))


def test_betabin_equiv():

    # https://github.com/pymc-devs/pymc/blob/
    # a7ab153f2b58d81824a56166747c678d7f421bde/pymc/distributions/discrete.py#L84
    def betabin_like(value, alpha, beta, n):
        return (gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
                gammaln(n + 1) - gammaln(value + 1) - gammaln(n - value + 1) +
                gammaln(alpha + value) + gammaln(n + beta - value) -
                gammaln(beta + alpha + n))

    # this N refers to the number of trials in the binomial distribution
    N = 10

    # this refers to the dataset size
    M = 100

    # hyperparams of the beta dist
    alpha, beta = 1., 2.

    heads = np.random.randint(low=0, high=N + 1, size=M)
    tails = N - heads

    data = np.vstack((heads, tails)).T

    Y = np.array([(y,) for y in data], dtype=[('', np.int, (2,))])
    view = cxx_numpy_dataview(Y)
    r = rng()
    defn = model_definition(Y.shape[0], [dm(2)])
    prior = {'alphas': [alpha, beta]}
    s = cxx_initialize(
        defn,
        view,
        r,
        feature_hps=[prior],
        assignment=[0] * Y.shape[0])

    assert_equals(s.groups(), [0])

    def all_indices(N):
        for i, j in it.product(range(0, N + 1), repeat=2):
            if (i + j) == N:
                yield i, j

    all_data = [(list(ij),) for ij in all_indices(N)]

    Y_test = np.array(all_data, dtype=[('', np.int, (2,))])

    # the actual score is simply a betabin using the updated alpha, beta
    alpha1, beta1 = np.array([alpha, beta]) + data.sum(axis=0)

    def model_score(Y_value):
        _, (score,) = s.score_value(Y_value, r)
        return score

    def test_score(Y_value):
        score = betabin_like(Y_value[0][0], alpha1, beta1, N)
        return score

    model_scores = np.array(map(model_score, Y_test))
    test_scores = np.array(map(test_score, Y_test))

    assert_almost_equals(np.exp(model_scores).sum(), 1., places=2)
    assert_almost_equals(np.exp(test_scores).sum(), 1., places=2)
    assert_almost_equals(
        np.abs(model_scores - test_scores).max(), 0., places=1)


def test_marginal():

    def score_dataset(counts):
        M, K = counts.shape
        Y = np.array([(y,) for y in counts], dtype=[('', np.int, (K,))])
        view = cxx_numpy_dataview(Y)
        r = rng()
        defn = model_definition(M, [dm(K)])
        prior = {'alphas': [1.] * K}
        s = cxx_initialize(
            defn,
            view,
            r,
            feature_hps=[prior],
            assignment=[0] * M)
        assert_equals(s.groups(), [0])
        return s.score_data(None, None, r)

    M = 5
    N = 4
    K = 2

    def all_indices(N, K):
        for inds in it.product(range(0, N + 1), repeat=K):
            if sum(inds) == N:
                yield list(inds)

    def dataset_iter(inds, M):
        for ptrs in it.product(range(len(inds)), repeat=M):
            dataset = np.array([inds[p] for p in ptrs])
            assert dataset.shape == (M, K)
            score = score_dataset(dataset)
            yield score

    scores = np.array(list(dataset_iter(list(all_indices(N, K)), M)))
    assert_almost_equals(np.exp(scores).sum(), 1., places=2)
