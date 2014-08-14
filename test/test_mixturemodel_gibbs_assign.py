from microscopes.mixture.model import initialize, bind, sample
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.common.rng import rng
from microscopes.kernels.gibbs import (
    assign as gibbs_assign,
    assign_resample as gibbs_assign_nonconj,
)
from microscopes.kernels.slice import theta as slice_theta

from microscopes.models import bb, niw, bbnc
from microscopes.mixture.definition import model_definition

from microscopes.common.util import random_orthonormal_matrix

from microscopes.common.testutil import (
    assert_discrete_dist_approx,
    permutation_iter,
    permutation_canonical,
    dist_on_all_clusterings,
)

import itertools as it
import numpy as np
import numpy.ma as ma
import time

from nose.plugins.attrib import attr


def _test_convergence(bs,
                      posterior,
                      kernel,
                      burnin_niters,
                      skip,
                      ntries,
                      nsamples,
                      kl_places):
    N = bs.nentities()
    start = time.time()
    last = start
    for i in xrange(burnin_niters):
        kernel(bs)
        if not ((i + 1) % 1000):
            print 'burning finished iteration', (i + 1), \
                'in', (time.time() - last), 'seconds'
            last = time.time()
    print 'finished burnin of', burnin_niters, \
        'iters in', (time.time() - start), 'seconds'
    idmap = {C: i for i, C in enumerate(permutation_iter(N))}

    def sample_fn():
        for _ in xrange(skip):
            kernel(bs)
        return idmap[tuple(permutation_canonical(bs.assignments()))]
    assert_discrete_dist_approx(
        sample_fn, posterior,
        ntries=ntries, nsamples=nsamples, kl_places=kl_places)


def data_with_assignment(Y_clusters):
    assignments = it.chain.from_iterable(
        [i] * len(cluster) for i, cluster in enumerate(Y_clusters))
    return np.hstack(Y_clusters), list(assignments)


def data_with_posterior(N, defn, cluster_hp, feature_hps, preprocess_data_fn):
    Y_clusters, _ = sample(defn, cluster_hp, feature_hps)
    Y = np.hstack(Y_clusters)
    if preprocess_data_fn:
        Y = preprocess_data_fn(Y)
    data = numpy_dataview(Y)
    r = rng()

    def score_fn(assignment):
        s = initialize(defn,
                           data,
                           r,
                           cluster_hp=cluster_hp,
                           feature_hps=feature_hps,
                           assignment=assignment)
        return s.score_joint(r)
    posterior = dist_on_all_clusterings(score_fn, N)
    return Y, posterior


def _test_convergence_bb_cxx(N,
                             D,
                             kernel,
                             preprocess_data_fn=None,
                             nonconj=False,
                             burnin_niters=10000,
                             skip=10,
                             ntries=50,
                             nsamples=1000,
                             kl_places=2):
    r = rng()
    cluster_hp = {'alpha': 2.0}
    feature_hps = [{'alpha': 1.0, 'beta': 1.0}] * D
    defn = model_definition(N, [bb] * D)
    nonconj_defn = model_definition(N, [bbnc] * D)
    Y, posterior = data_with_posterior(
        N, defn, cluster_hp, feature_hps, preprocess_data_fn)
    data = numpy_dataview(Y)
    s = initialize(nonconj_defn if nonconj else defn,
                       data,
                       cluster_hp=cluster_hp,
                       feature_hps=feature_hps,
                       r=r)
    bs = bind(s, data)
    wrapped_kernel = lambda s: kernel(s, r)
    _test_convergence(bs,
                      posterior,
                      wrapped_kernel,
                      burnin_niters,
                      skip,
                      ntries,
                      nsamples,
                      kl_places)


def test_convergence_bb_cxx():
    N, D = 4, 5
    _test_convergence_bb_cxx(N, D, gibbs_assign)


def test_convergence_bb_missing():
    N, D = 4, 5

    def preprocess_fn(Y):
        masks = [tuple(j == (i % len(Y)) for j in xrange(D))
                 for i in xrange(len(Y))]
        return ma.array(Y, mask=masks)
    _test_convergence_bb_cxx(N, D, gibbs_assign, preprocess_fn)


def test_convergence_bb_nonconj_cxx():
    N, D = 3, 5
    thetaparams = {fi: {'p': 0.1} for fi in xrange(D)}

    def kernel(s, r):
        gibbs_assign_nonconj(s, 10, r)
        slice_theta(s, r, tparams=thetaparams)
    _test_convergence_bb_cxx(
        N, D, kernel, preprocess_data_fn=None, nonconj=True)


def _test_multivariate_models(initialize_fn,
                              dataview,
                              bind,
                              gibbs_assign,
                              R):
    # XXX: this test only checks that the operations don't crash
    mu = np.ones(3)
    kappa = 0.3
    Q = random_orthonormal_matrix(3)
    psi = np.dot(Q, np.dot(np.diag([1.0, 0.5, 0.2]), Q.T))
    nu = 6

    N = 10

    def genrow():
        return tuple(
            [np.random.choice([False, True]),
             [np.random.uniform(-3.0, 3.0) for _ in xrange(3)]])
    X = np.array([genrow()
                  for _ in xrange(N)], dtype=[('', bool), ('', float, (3,))])
    view = dataview(X)

    defn = model_definition(N, [bb, niw(3)])
    s = initialize_fn(
        defn,
        view,
        cluster_hp={'alpha': 2.},
        feature_hps=[
            {'alpha': 2., 'beta': 2.},
            {'mu': mu, 'kappa': kappa, 'psi': psi, 'nu': nu}
        ],
        r=R)

    bound_s = bind(s, view)
    for _ in xrange(10):
        gibbs_assign(bound_s, R)


def test_multivariate_models_cxx():
    _test_multivariate_models(
        initialize,
        numpy_dataview,
        bind,
        gibbs_assign,
        rng())


def _test_nonconj_inference(initialize_fn,
                            dataview,
                            bind,
                            assign_nonconj_fn,
                            slice_theta_fn,
                            R,
                            ntries,
                            nsamples,
                            tol):
    N, D = 1000, 5
    defn = model_definition(N, [bbnc] * D)
    cluster_hp = {'alpha': 0.2}
    feature_hps = [{'alpha': 1.0, 'beta': 1.0}] * D

    while True:
        Y_clustered, cluster_samplers = sample(
            defn, cluster_hp, feature_hps, R)
        if len(Y_clustered) == 2:
            break
    dominant = np.argmax(map(len, Y_clustered))
    truth = np.array([s.p for s in cluster_samplers[dominant]])
    print 'truth:', truth

    # see if we can learn the p-values for each of the two clusters. we proceed
    # by running gibbs_assign_nonconj, followed by slice sampling on the
    # posterior p(\theta | Y). we'll "cheat" a little by bootstrapping the
    # DP with the correct assignment (but not with the correct p-values)
    Y, assignment = data_with_assignment(Y_clustered)
    view = dataview(Y)
    s = initialize_fn(
        defn, view, cluster_hp=cluster_hp,
        feature_hps=feature_hps, assignment=assignment, r=R)
    bs = bind(s, view)

    def mkparam():
        return {'p': 0.1}
    thetaparams = {fi: mkparam() for fi in xrange(D)}

    def kernel():
        assign_nonconj_fn(bs, 10, R)
        slice_theta_fn(bs, R, tparams=thetaparams)

    def inference(niters):
        for _ in xrange(niters):
            kernel()
            groups = s.groups()
            inferred_dominant = groups[
                np.argmax([s.groupsize(gid) for gid in groups])]
            inferred = [s.get_suffstats(inferred_dominant, d)['p']
                        for d in xrange(D)]
            inferred = np.array(inferred)
            yield inferred

    posterior = []
    while ntries:
        samples = list(inference(nsamples))
        posterior.extend(samples)
        inferred = sum(posterior) / len(posterior)
        diff = np.linalg.norm(truth - inferred)
        print 'inferred:', inferred
        print 'diff:', diff
        if diff <= tol:
            return
        ntries -= 1
        print 'tries left:', ntries

    assert False, 'did not converge'


@attr('slow')
def test_nonconj_inference_cxx():
    _test_nonconj_inference(
        initialize, numpy_dataview, bind,
        gibbs_assign_nonconj, slice_theta, R=rng(),
        ntries=50, nsamples=100, tol=0.2)
