
from microscopes.models import bb
from microscopes.mixture import query, model
from microscopes.mixture.definition import model_definition
from microscopes.mixture.testutil import toy_dataset
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.common.rng import rng

import numpy as np
import numpy.ma as ma
from nose.tools import assert_equals


def test_zmatrix():
    N, D = 10, 4
    defn = model_definition(N, [bb] * D)
    Y = toy_dataset(defn)
    prng = rng()
    view = numpy_dataview(Y)
    latents = [model.initialize(defn, view, prng) for _ in xrange(10)]
    zmat = query.zmatrix(latents)
    assert_equals(zmat.shape, (N, N))


def test_posterior_predictive():
    N, D = 10, 4  # D needs to be even
    defn = model_definition(N, [bb] * D)
    Y = toy_dataset(defn)
    prng = rng()
    view = numpy_dataview(Y)
    latents = [model.initialize(defn, view, prng) for _ in xrange(10)]
    q = ma.masked_array(
        np.array([(False,) * D], dtype=[('', bool)] * D),
        mask=[(False,) * (D / 2) + (True,) * (D / 2)])
    samples = query.posterior_predictive(q, latents, prng)
    assert_equals(len(samples.shape), 1)
    assert_equals(samples.shape[0], len(latents))


def test_posterior_predictive_statistic():
    N, D = 10, 4  # D needs to be even
    defn = model_definition(N, [bb] * D)
    Y = toy_dataset(defn)
    prng = rng()
    view = numpy_dataview(Y)
    latents = [model.initialize(defn, view, prng) for _ in xrange(10)]
    q = ma.masked_array(
        np.array([(False,) * D], dtype=[('', bool)] * D),
        mask=[(False,) * (D / 2) + (True,) * (D / 2)])

    statistic = query.posterior_predictive_statistic(q, latents, prng)
    assert_equals(statistic.shape, (1,))
    assert_equals(len(statistic.dtype), D)

    statistic = query.posterior_predictive_statistic(
        q, latents, prng, merge='mode')
    assert_equals(statistic.shape, (1,))
    assert_equals(len(statistic.dtype), D)

    statistic = query.posterior_predictive_statistic(
        q, latents, prng, merge=['mode', 'mode', 'avg', 'avg'])
    assert_equals(statistic.shape, (1,))
    assert_equals(len(statistic.dtype), D)
