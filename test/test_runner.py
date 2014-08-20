from microscopes.mixture import model, runner
from microscopes.mixture.definition import (
    model_definition,
    #fixed_model_definition,
)
from microscopes.models import (
    bb,
    bbnc,
    nich,
    niw,
)
from microscopes.common.rng import rng
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.kernels import parallel
from microscopes.common.testutil import (
    permutation_canonical,
    permutation_iter,
    assert_discrete_dist_approx,
)
from microscopes.mixture.testutil import (
    toy_dataset,
    data_with_posterior,
)

import itertools as it
import multiprocessing as mp
from nose.tools import assert_true

from nose.plugins.attrib import attr


def _test_runner_kernel_config(kc_fn, models):
    defn = model_definition(10, models)
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = kc_fn(defn)
    prng = rng()

    ntries = 5
    while ntries:
        latent = model.initialize(defn, view, prng)
        assignments = latent.assignments()
        r = runner.runner(defn, view, latent, kc)
        r.run(r=prng, niters=10)
        assignments1 = r.get_latent().assignments()

        # XXX: it should be very unlikely the assignments are all equal
        if assignments == assignments1:
            ntries -= 1
        else:
            return  # success

    assert_true(False)  # exceeded ntries


def test_runner_default_kernel_config():
    models = [bb, nich, niw(3)]
    _test_runner_kernel_config(runner.default_kernel_config, models)


def test_runner_default_kernel_config_nonconj():
    models = [bbnc, nich, niw(3)]
    _test_runner_kernel_config(runner.default_kernel_config, models)


def test_runner_default_kernel_config_with_cluster():
    models = [bb, nich, niw(3)]

    def kc_fn(defn):
        return list(it.chain(
            runner.default_assign_kernel_config(defn),
            runner.default_feature_hp_kernel_config(defn),
            runner.default_cluster_hp_kernel_config(defn)))
    _test_runner_kernel_config(kc_fn, models)


def test_runner_convergence():
    N, D = 4, 5
    defn = model_definition(N, [bb] * D)
    prng = rng()
    Y, posterior = data_with_posterior(defn, r=prng)
    view = numpy_dataview(Y)
    latent = model.initialize(defn, view, prng)
    r = runner.runner(defn, view, latent, ['assign'])
    r.run(r=prng, niters=1000)  # burnin
    idmap = {C: i for i, C in enumerate(permutation_iter(N))}

    def sample_fn():
        r.run(r=prng, niters=10)
        new_latent = r.get_latent()
        return idmap[tuple(permutation_canonical(new_latent.assignments()))]

    assert_discrete_dist_approx(sample_fn, posterior, ntries=100)


@attr('uses_mp')
def test_runner_multiprocessing():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()
    latents = [model.initialize(defn, view, prng)
               for _ in xrange(mp.cpu_count())]
    runners = [runner.runner(defn, view, latent, kc) for latent in latents]
    r = parallel.runner(runners)
    # check it is restartable
    r.run(r=prng, niters=10)
    r.run(r=prng, niters=10)


@attr('uses_mp')
def test_runner_multiprocessing_convergence():
    N, D = 4, 5
    defn = model_definition(N, [bb] * D)
    prng = rng()
    Y, posterior = data_with_posterior(defn, r=prng)
    view = numpy_dataview(Y)
    latents = [model.initialize(defn, view, prng)
               for _ in xrange(mp.cpu_count())]
    runners = [runner.runner(defn, view, latent, ['assign'])
               for latent in latents]
    r = parallel.runner(runners)
    r.run(r=prng, niters=1000)  # burnin
    idmap = {C: i for i, C in enumerate(permutation_iter(N))}

    def sample_iter():
        r.run(r=prng, niters=10)
        for latent in r.get_latents():
            yield idmap[tuple(permutation_canonical(latent.assignments()))]

    ref = [None]

    def sample_fn():
        if ref[0] is None:
            ref[0] = sample_iter()
        try:
            return next(ref[0])
        except StopIteration:
            ref[0] = None
        return sample_fn()

    assert_discrete_dist_approx(sample_fn, posterior, ntries=100, kl_places=2)


@attr('slow')
def test_runner_multyvac():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()
    latents = [model.initialize(defn, view, prng)
               for _ in xrange(2)]
    runners = [runner.runner(defn, view, latent, kc) for latent in latents]
    r = parallel.runner(runners, backend='multyvac', layer='perf', core='f2')
    r.run(r=prng, niters=1000)
    r.run(r=prng, niters=1000)


@attr('slow')
def test_runner_multyvac_volume():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()
    latents = [model.initialize(defn, view, prng)
               for _ in xrange(2)]
    runners = [runner.runner(defn, view, latent, kc) for latent in latents]
    r = parallel.runner(
        runners, backend='multyvac', layer='perf', core='f2', volume='data')
    r.run(r=prng, niters=1000)
    r.run(r=prng, niters=1000)
