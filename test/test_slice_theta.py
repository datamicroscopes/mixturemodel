from microscopes.kernels.slice import theta
from microscopes.mixture.model import initialize, bind
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.common.rng import rng
from microscopes.models import bbnc
from microscopes.mixture.definition import model_definition

from microscopes.common.testutil import assert_1d_cont_dist_approx_sps
from scipy.stats import beta
import numpy as np

#from nose.plugins.attrib import attr


def test_slice_theta_mm():
    N = 100
    data = np.array(
        [(np.random.random() < 0.8,) for _ in xrange(N)],
        dtype=[('', bool)])
    defn = model_definition(N, [bbnc])
    r = rng()
    prior = {'alpha': 1.0, 'beta': 9.0}
    view = numpy_dataview(data)
    s = initialize(
        defn,
        view,
        cluster_hp={'alpha': 1., 'beta': 9.},
        feature_hps=[prior],
        r=r,
        assignment=[0] * N)

    heads = len([1 for y in data if y[0]])
    tails = N - heads

    alpha1 = prior['alpha'] + heads
    beta1 = prior['beta'] + tails

    bs = bind(s, view)
    params = {0: {'p': 0.05}}

    def sample_fn():
        theta(bs, r, tparams=params)
        return s.get_suffstats(0, 0)['p']

    rv = beta(alpha1, beta1)
    assert_1d_cont_dist_approx_sps(sample_fn, rv, nsamples=50000)
