from microscopes.mixture.model import initialize, bind
from microscopes.kernels.gibbs import hp as gibbs_hp
from microscopes.kernels.slice import hp as slice_hp

from microscopes.common.rng import rng
from microscopes.common.scalar_functions import (
    log_exponential,
    log_noninformative_beta_prior,
    log_normal,
)

from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.models import bb, bnb, gp, nich
from microscopes.mixture.definition import model_definition

from microscopes.common.util import almost_eq

import numpy as np

try:
    import matplotlib.pylab as plt
    has_plt = True
except ImportError:
    has_plt = False

import itertools as it

from microscopes.common.testutil import (
    assert_1d_cont_dist_approx_emp,
    #OurAssertionError,
    #our_assert_almost_equals,
)
#from nose.plugins.attrib import attr


def _bb_hyperprior_pdf(hp):
    alpha, beta = hp['alpha'], hp['beta']
    if alpha > 0.0 and beta > 0.0:
        # http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html
        return -2.5 * np.log(alpha + beta)
    return -np.inf


def data_with_assignment(Y_clusters):
    assignments = it.chain.from_iterable(
        [i] * len(cluster) for i, cluster in enumerate(Y_clusters))
    return np.hstack(Y_clusters), list(assignments)


def _make_one_feature_bb_mm(initialize_fn, dataview, Nk, K, alpha, beta, r):
    # XXX: the rng parameter passed does not get threaded through the
    # random *data* generation
    # use the py_bb for sampling
    py_bb = bb.py_desc()._model_module
    shared = py_bb.Shared()
    shared.load({'alpha': alpha, 'beta': beta})

    def init_sampler():
        samp = py_bb.Sampler()
        samp.init(shared)
        return samp
    samplers = [init_sampler() for _ in xrange(K)]

    def gen_cluster(samp):
        data = [(samp.eval(shared),) for _ in xrange(Nk)]
        return np.array(data, dtype=[('', bool)])
    Y_clustered = tuple(map(gen_cluster, samplers))
    Y, assignment = data_with_assignment(Y_clustered)
    view = dataview(Y)
    s = initialize_fn(model_definition(Y.shape[0], [bb]),
                      view,
                      cluster_hp={'alpha': 2.},
                      feature_hps=[{'alpha': alpha, 'beta': beta}],
                      r=r,
                      assignment=assignment)
    return s, view


def _grid_actual(s, prior_fn, lo, hi, nelems, r):
    x = np.linspace(lo, hi, nelems)
    y = x.copy()
    xv, yv = np.meshgrid(x, y)
    z = np.zeros(xv.shape)
    for i in xrange(nelems):
        for j in xrange(nelems):
            alpha = xv[i, j]
            beta = yv[i, j]
            raw = {'alpha': alpha, 'beta': beta}
            s.set_feature_hp(0, raw)
            z[i, j] = prior_fn(raw) + s.score_data(0, None, r)
    return xv, yv, z


def _add_to_grid(xv, yv, z, value):
    xmin, xmax = xv.min(axis=1).min(), xv.max(axis=1).max()
    ymin, ymax = yv.min(axis=1).min(), yv.max(axis=1).max()
    if (value[0] < xmin or
            value[0] > xmax or
            value[1] < ymin or
            value[1] > ymax):
        # do not add
        return False
    xrep = xv[0, :]
    yrep = yv[:, 0]
    xidx = min(np.searchsorted(xrep, value[0]), len(xrep) - 1)
    yidx = min(np.searchsorted(yrep, value[1]), len(yrep) - 1)
    z[yidx, xidx] += 1
    return True


def _test_hp_inference(initialize_fn,
                       prior_fn,
                       grid_min,
                       grid_max,
                       grid_n,
                       dataview,
                       bind_fn,
                       init_inf_kernel_state_fn,
                       inf_kernel_fn,
                       map_actual_postprocess_fn,
                       grid_filename,
                       prng,
                       burnin=1000,
                       nsamples=1000,
                       skip=10,
                       trials=5,
                       tol=0.1):

    print '_test_hp_inference: burnin', burnin, 'nsamples', nsamples, \
        'skip', skip, 'trials', trials, 'tol', tol

    Nk = 1000
    K = 100
    s, view = _make_one_feature_bb_mm(
        initialize_fn, dataview, Nk, K, 0.8, 1.2, prng)
    bound_s = bind_fn(s, view)

    xgrid, ygrid, z_actual = _grid_actual(
        s, prior_fn, grid_min, grid_max, grid_n, prng)

    i_actual, j_actual = np.unravel_index(np.argmax(z_actual), z_actual.shape)
    assert almost_eq(z_actual[i_actual, j_actual], z_actual.max())
    alpha_map_actual, beta_map_actual = \
        xgrid[i_actual, j_actual], ygrid[i_actual, j_actual]
    map_actual = np.array([alpha_map_actual, beta_map_actual])
    map_actual_postproc = map_actual_postprocess_fn(map_actual)
    print 'MAP actual:', map_actual
    print 'MAP actual postproc:', map_actual_postproc

    th_draw = lambda: np.random.uniform(grid_min, grid_max)
    alpha0, beta0 = th_draw(), th_draw()
    s.set_feature_hp(0, {'alpha': alpha0, 'beta': beta0})
    print 'start values:', alpha0, beta0

    z_sample = np.zeros(xgrid.shape)
    opaque = init_inf_kernel_state_fn(s)
    for _ in xrange(burnin):
        inf_kernel_fn(bound_s, opaque, prng)
    print 'finished burnin of', burnin, 'iterations'

    def trial():
        def posterior(k, skip):
            for _ in xrange(k):
                for _ in xrange(skip - 1):
                    inf_kernel_fn(bound_s, opaque, prng)
                inf_kernel_fn(bound_s, opaque, prng)
                hp = s.get_feature_hp(0)
                yield np.array([hp['alpha'], hp['beta']])
        for samp in posterior(nsamples, skip):
            #print 'gridding:', samp
            _add_to_grid(xgrid, ygrid, z_sample, samp)

    def draw_grid_plot():
        if not has_plt:
            return
        plt.imshow(z_sample, cmap=plt.cm.binary, origin='lower',
                   interpolation='nearest',
                   extent=(grid_min, grid_max, grid_min, grid_max))
        plt.hold(True)  # XXX: restore plt state
        plt.contour(np.linspace(grid_min, grid_max, grid_n),
                    np.linspace(grid_min, grid_max, grid_n),
                    z_actual)
        plt.savefig(grid_filename)
        plt.close()

    while trials:
        trial()
        i_sample, j_sample = np.unravel_index(
            np.argmax(z_sample), z_sample.shape)
        alpha_map_sample, beta_map_sample = \
            xgrid[i_sample, j_sample], ygrid[i_sample, j_sample]
        map_sample = np.array([alpha_map_sample, beta_map_sample])
        diff = np.linalg.norm(map_actual_postproc - map_sample)
        print 'map_sample:', map_sample, 'diff:', diff, \
            'trials left:', (trials - 1)
        if diff <= tol:
            # draw plot and bail
            draw_grid_plot()
            return
        trials -= 1

    draw_grid_plot()  # useful for debugging
    assert False, 'MAP value did not converge to desired tolerance'


def _test_kernel_gibbs_hp(initialize_fn,
                          dataview,
                          bind_fn,
                          gibbs_hp_fn,
                          fname,
                          prng):
    grid_min, grid_max, grid_n = 0.01, 5.0, 10
    grid = it.product(np.linspace(grid_min, grid_max, grid_n), repeat=2)
    grid = tuple({'alpha': alpha, 'beta': beta} for alpha, beta in grid)

    def init_inf_kernel_state_fn(dpmm):
        hparams = {0: {'hpdf': _bb_hyperprior_pdf, 'hgrid': grid}}
        return hparams

    def map_actual_postprocess_fn(map_actual):
        # find closest grid point to actual point
        dists = [np.linalg.norm(np.array([g['alpha'], g['beta']]) - map_actual)
                 for g in grid]
        dists = np.array(dists)
        closest = grid[np.argmin(dists)]
        closest = np.array([closest['alpha'], closest['beta']])
        return closest

    _test_hp_inference(
        initialize_fn,
        _bb_hyperprior_pdf,
        grid_min,
        grid_max,
        grid_n,
        dataview,
        bind_fn,
        init_inf_kernel_state_fn,
        gibbs_hp_fn,
        map_actual_postprocess_fn,
        grid_filename=fname,
        prng=prng,
        burnin=100,
        trials=10,
        nsamples=100)


def test_kernel_gibbs_hp():
    _test_kernel_gibbs_hp(initialize,
                          numpy_dataview,
                          bind,
                          gibbs_hp,
                          'grid_gibbs_hp_samples_pdf',
                          rng())


def _test_kernel_slice_hp(initialize_fn,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          dataview,
                          bind_fn,
                          slice_hp_fn,
                          fname,
                          prng):
    grid_min, grid_max, grid_n = 0.01, 5.0, 200
    _test_hp_inference(
        initialize_fn,
        prior_fn,
        grid_min,
        grid_max,
        grid_n,
        dataview,
        bind_fn,
        init_inf_kernel_state_fn,
        slice_hp_fn,
        map_actual_postprocess_fn=lambda x: x,
        grid_filename=fname,
        prng=prng,
        burnin=100,
        trials=100,
        nsamples=100)


def test_kernel_slice_hp():
    indiv_prior_fn = log_exponential(1.2)

    def init_inf_kernel_state_fn(s):
        hparams = {
            0: {
                'alpha': (indiv_prior_fn, 1.5),
                'beta': (indiv_prior_fn, 1.5),
            }
        }
        return hparams

    def prior_fn(raw):
        return indiv_prior_fn(raw['alpha']) + indiv_prior_fn(raw['beta'])
    kernel_fn = lambda s, arg, rng: slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          numpy_dataview,
                          bind,
                          kernel_fn,
                          'grid_slice_hp_samples.pdf',
                          rng())


def test_kernel_slice_hp_noninform():
    def init_inf_kernel_state_fn(s):
        hparams = {
            0: {
                ('alpha', 'beta'): (log_noninformative_beta_prior, 1.0),
            }
        }
        return hparams

    def prior_fn(raw):
        return log_noninformative_beta_prior(raw['alpha'], raw['beta'])
    kernel_fn = lambda s, arg, rng: slice_hp(s, rng, hparams=arg)
    _test_kernel_slice_hp(initialize,
                          init_inf_kernel_state_fn,
                          prior_fn,
                          numpy_dataview,
                          bind,
                          kernel_fn,
                          'grid_slice_hp_noninform_samples.pdf',
                          rng())


def _test_cluster_hp_inference(initialize_fn,
                               prior_fn,
                               grid_min,
                               grid_max,
                               grid_n,
                               dataview,
                               bind_fn,
                               init_inf_kernel_state_fn,
                               inf_kernel_fn,
                               map_actual_postprocess_fn,
                               prng,
                               burnin=1000,
                               nsamples=1000,
                               skip=10,
                               trials=100,
                               places=2):
    print '_test_cluster_hp_inference: burnin', burnin, 'nsamples', nsamples, \
        'skip', skip, 'trials', trials, 'places', places

    N = 1000
    D = 5

    # create random binary data, doesn't really matter what the values are
    Y = np.random.random(size=(N, D)) < 0.5
    Y = np.array([tuple(y) for y in Y], dtype=[('', np.bool)] * D)
    view = dataview(Y)

    defn = model_definition(N, [bb] * D)
    latent = initialize_fn(defn, view, r=prng)
    model = bind_fn(latent, view)

    def score_alpha(alpha):
        prev_alpha = latent.get_cluster_hp()['alpha']
        latent.set_cluster_hp({'alpha': alpha})
        score = prior_fn(alpha) + latent.score_assignment()
        latent.set_cluster_hp({'alpha': prev_alpha})
        return score

    def sample_fn():
        for _ in xrange(skip - 1):
            inf_kernel_fn(model, opaque, prng)
        inf_kernel_fn(model, opaque, prng)
        return latent.get_cluster_hp()['alpha']

    alpha0 = np.random.uniform(grid_min, grid_max)
    print 'start alpha:', alpha0
    latent.set_cluster_hp({'alpha': alpha0})

    opaque = init_inf_kernel_state_fn(latent)
    for _ in xrange(burnin):
        inf_kernel_fn(model, opaque, prng)
    print 'finished burnin of', burnin, 'iterations'

    print 'grid_min', grid_min, 'grid_max', grid_max
    assert_1d_cont_dist_approx_emp(sample_fn,
                                   score_alpha,
                                   grid_min,
                                   grid_max,
                                   grid_n,
                                   trials,
                                   nsamples,
                                   places)

    # MAP estimation over a large range doesn't really work
    #alpha_grid = np.linspace(grid_min, grid_max, grid_n)
    #alpha_scores = np.array(map(score_alpha, alpha_grid))
    #alpha_grid_map_idx = np.argmax(alpha_scores)
    #alpha_grid_map = alpha_grid[alpha_grid_map_idx]
    #alpha_grid_map_postproc = map_actual_postprocess_fn(alpha_grid_map)
    #print 'alpha MAP:', alpha_grid_map, \
    #      'alpha MAP postproc:', alpha_grid_map_postproc

    #alpha0 = np.random.uniform(grid_min, grid_max)
    #print 'start alpha:', alpha0
    #latent.set_cluster_hp({'alpha':alpha0})

    #opaque = init_inf_kernel_state_fn(latent)
    #for _ in xrange(burnin):
    #    inf_kernel_fn(model, opaque, prng)
    #print 'finished burnin of', burnin, 'iterations'

    #def posterior(k, skip):
    #    for _ in xrange(k):
    #        for _ in xrange(skip-1):
    #            inf_kernel_fn(model, opaque, prng)
    #        inf_kernel_fn(model, opaque, prng)
    #        yield latent.get_cluster_hp()['alpha']

    #bins = np.zeros(grid_n, dtype=np.int)
    #while 1:
    #    for sample in posterior(nsamples, skip):
    #        idx = min(np.searchsorted(alpha_grid, sample), grid_n-1)
    #        bins[idx] += 1
    #    est_map = alpha_grid[np.argmax(bins)]
    #    try:
    #        our_assert_almost_equals(est_map, alpha_grid_map, places=places)
    #        return # success
    #    except OurAssertionError as ex:
    #        print 'warning:', ex._ex.message
    #        trials -= 1
    #        if not trials:
    #            raise ex._ex


def test_kernel_slice_cluster_hp():
    prior_fn = log_exponential(1.5)

    def init_inf_kernel_state_fn(s):
        cparam = {'alpha': (prior_fn, 1.)}
        return cparam
    kernel_fn = lambda s, arg, rng: slice_hp(s, rng, cparam=arg)
    grid_min, grid_max, grid_n = 0.0, 50., 100
    _test_cluster_hp_inference(initialize,
                               prior_fn,
                               grid_min,
                               grid_max,
                               grid_n,
                               numpy_dataview,
                               bind,
                               init_inf_kernel_state_fn,
                               kernel_fn,
                               map_actual_postprocess_fn=lambda x: x,
                               prng=rng())


def _test_scalar_hp_inference(view,
                              prior_fn,
                              w,
                              grid_min,
                              grid_max,
                              grid_n,
                              likelihood_model,
                              scalar_hp_key,
                              burnin=1000,
                              nsamples=1000,
                              every=10,
                              trials=100,
                              places=2):
    """
    view must be 1D
    """
    r = rng()

    hparams = {0: {scalar_hp_key: (prior_fn, w)}}

    def score_fn(scalar):
        d = latent.get_feature_hp(0)
        prev_scalar = d[scalar_hp_key]
        d[scalar_hp_key] = scalar
        latent.set_feature_hp(0, d)
        score = prior_fn(scalar) + latent.score_data(0, None, r)
        d[scalar_hp_key] = prev_scalar
        latent.set_feature_hp(0, d)
        return score

    defn = model_definition(len(view), [likelihood_model])
    latent = initialize(defn, view, r=r)
    model = bind(latent, view)

    def sample_fn():
        for _ in xrange(every):
            slice_hp(model, r, hparams=hparams)
        return latent.get_feature_hp(0)[scalar_hp_key]

    for _ in xrange(burnin):
        slice_hp(model, r, hparams=hparams)
    print 'finished burnin of', burnin, 'iterations'

    print 'grid_min', grid_min, 'grid_max', grid_max
    assert_1d_cont_dist_approx_emp(sample_fn,
                                   score_fn,
                                   grid_min,
                                   grid_max,
                                   grid_n,
                                   trials,
                                   nsamples,
                                   places)


def test_bnb_hp_alpha():
    N = 1000
    Y = np.array([(x,) for x in np.random.randint(low=0, high=10, size=N)],
                 dtype=[('', np.bool)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = 0.01, 5.0, 100
    _test_scalar_hp_inference(view,
                              log_exponential(1.),
                              1.,
                              grid_min,
                              grid_max,
                              grid_n,
                              bnb,
                              'alpha')


def test_bnb_hp_beta():
    N = 1000
    Y = np.array([(x,) for x in np.random.randint(low=0, high=10, size=N)],
                 dtype=[('', np.bool)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = 0.01, 5.0, 100
    _test_scalar_hp_inference(view,
                              log_exponential(1.),
                              1.,
                              grid_min,
                              grid_max,
                              grid_n,
                              bnb,
                              'beta')


def test_gp_hp_alpha():
    N = 1000
    Y = np.array([(x,) for x in np.random.randint(low=0, high=10, size=N)],
                 dtype=[('', np.bool)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = 0.01, 5.0, 100
    _test_scalar_hp_inference(view,
                              log_exponential(1.),
                              1.,
                              grid_min,
                              grid_max,
                              grid_n,
                              gp,
                              'alpha')


def test_gp_hp_inv_beta():
    N = 1000
    Y = np.array([(x,) for x in np.random.randint(low=0, high=10, size=N)],
                 dtype=[('', np.bool)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = 0.001, 2.0, 100
    _test_scalar_hp_inference(view,
                              log_exponential(1.),
                              0.1,
                              grid_min,
                              grid_max,
                              grid_n,
                              gp,
                              'inv_beta')


def test_nich_hp_mu():
    N = 1000
    Y = np.array([(x,) for x in np.random.uniform(low=-10, high=10, size=N)],
                 dtype=[('', np.float32)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = -5., 5., 100
    _test_scalar_hp_inference(view,
                              log_normal(0., 1.),
                              0.1,
                              grid_min,
                              grid_max,
                              grid_n,
                              nich,
                              'mu')


def test_nich_hp_sigmasq():
    N = 1000
    Y = np.array([(x,) for x in np.random.uniform(low=-1, high=1, size=N)],
                 dtype=[('', np.float32)])
    view = numpy_dataview(Y)
    grid_min, grid_max, grid_n = 0.0001, 1.0, 100
    _test_scalar_hp_inference(view,
                              log_exponential(1.),
                              0.1,
                              grid_min,
                              grid_max,
                              grid_n,
                              nich,
                              'sigmasq')
