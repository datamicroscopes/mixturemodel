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
from microscopes.mixture.testutil import toy_dataset


def test_runner_default_kernel_config():
    defn = model_definition(10, [bb, nich, niw(3)])
    Y = toy_dataset(defn)
    view = numpy_dataview(Y)
    kc = runner.default_kernel_config(defn)
    prng = rng()
    latent = model.initialize(defn, view, prng)
    r = runner.runner(defn, view, latent, kc, r=prng)
    r.run(10)
