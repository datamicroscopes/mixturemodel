"""Implements the Runner interface for mixture models
"""

from microscopes.common import validator
from microscopes.common.rng import rng
from microscopes.common.recarray._dataview import abstract_dataview
from microscopes.mixture.definition import (
    model_definition,
    fixed_model_definition,
)
from microscopes.mixture.model import (
    state,
    fixed_state,
    bind,
    bind_fixed,
)
from microscopes.kernels import gibbs, slice


def default_kernel_config(defn):
    if not (isinstance(defn, model_definition) or
            isinstance(defn, fixed_model_definition)):
        raise ValueError("bad defn given")
    is_fixed = isinstance(defn, fixed_model_definition)

    if is_fixed:
        return ['assign_fixed']
    else:
        # XXX(stephentu): model_descriptors should implement
        # is_conjugate()
        nonconj_models = filter(lambda x: x.name() == 'bbnc', defn.models())

        # assignment
        if nonconj_models:
            kernels = [('assign_resample', {'m': 10})]
            # XXX(stephentu): also slice sample on the parameter instantiations
        else:
            kernels = ['assign']

        # hyperparams
        hparams = {}
        for i, hp in enumerate(defn.hyperpriors()):
            if not hp:
                continue
            # XXX(stephentu): we are arbitrarily picking w=0.1
            hparams[i] = {k : (fn, 0.1) for k, fn in hp.iteritems()}

        kernels.append(('hp', {'cparam': {}, 'hparams': hparams}))
        return kernels


class runner(object):
    """The dirichlet process mixture model runner

    Parameters
    ----------

    defn : ``model_definition`` or ``fixed_model_definition``
        The structural definition.

    view : a recarray dataview
        The observations.

    latent : ``state`` or ``fixed_state``
        The initialization state.

    kernel_config : list
        A list of either `x` strings or `(x, y)` tuples, where `x` is a string
        containing the name of the kernel and `y` is a dict which configures
        the particular kernel. In the former case where `y` is omitted, then
        the defaults parameters for each kernel are used.

    r : ``rng``, optional

    """

    def __init__(self, defn, view, latent, kernel_config, r=None):
        if not (isinstance(defn, model_definition) or
                isinstance(defn, fixed_model_definition)):
            raise ValueError("bad defn given")

        validator.validate_type(view, abstract_dataview, param_name='view')

        if not (isinstance(latent, state) or
                isinstance(latent, fixed_state)):
            raise ValueError("bad latent given")

        self._is_fixed = isinstance(defn, fixed_model_definition)
        if self._is_fixed != isinstance(latent, fixed_state):
            raise ValueError("definition and latent don't match type")

        validator.validate_len(view, defn.n())

        self._defn = defn
        self._view = view
        self._latent = latent

        self._kernel_config = []
        for kernel in kernel_config:
            if hasattr(kernel, '__iter__'):
                name, config = kernel
            else:
                name, config = kernel, {}
            if name == 'assign_fixed':
                if not self._is_fixed:
                    # really a warning
                    raise ValueError("state should not use fixed kernel")
                if config:
                    raise ValueError("assign_fixed has no parameters")
            elif name == 'assign':
                if self._is_fixed:
                    raise ValueError("fixed_state cannot use variable kernel")
                if config:
                    raise ValueError("assign has no parameters")
            elif name == 'assign_resample':
                if is_fixed:
                    raise ValueError("fixed_state cannot use variable kernel")
                if config.keys() != ['m']:
                    raise ValueError("bad config found: {}".format(config))
            elif name == 'hp':
                validator.validate_kwargs(config, ('cparam', 'hparams'))
            else:
                raise ValueError("bad kernel found: {}".format(name))
            self._kernel_config.append((name, config))

        if r is None:
            r = rng()
        validator.validate_type(r, rng, param_name='r')
        self._r = r

    def run(self, niters=10000):
        validator.validate_positive(niters, param_name='niters')
        if self._is_fixed:
            model = bind_fixed(self._latent, self._view)
        else:
            model = bind(self._latent, self._view)
        for _ in xrange(niters):
            for name, config in self._kernel_config:
                if name == 'assign_fixed':
                    gibbs.assign_fixed(model, self._r)
                elif name == 'assign':
                    gibbs.assign(model, self._r)
                elif name == 'assign_resample':
                    gibbs.assign_resample(model, config['m'], self._r)
                elif name == 'hp':
                    slice.hp(model,
                             self._r,
                             cparam=config['cparam'],
                             hparams=config['hparams'])
                else:
                    assert False, "should not be reach"

    def get_latent(self):
        return self._latent
