from microscopes.mixture import model, runner
from microscopes.mixture.definition import (
    model_definition,
    fixed_model_definition,
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
from microscopes.mixture.testutil import toy_dataset

from nose.tools import assert_true


def test_runner_default_kernel_config():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()

    ntries = 5
    while ntries:
        latent = model.initialize(defn, view, prng)
        assignments = latent.assignments()
        r = runner.runner(defn, view, latent, kc, r=prng)
        r.run(10)
        assignments1 = r.get_latent().assignments()

        # XXX: it should be very unlikely the assignments are all equal
        if assignments == assignments1:
            ntries -= 1
        else:
            return  # success

    assert_true(False)  # exceeded ntries


def test_runner_multiprocessing():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()
    latents = [model.initialize(defn, view, prng) for _ in xrange(8)]
    runners = [runner.runner(defn, view, latent, kc) for latent in latents]
    r = parallel.runner(runners)
    r.run(10)
