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

import itertools as it


def _validate_definition(defn):
    if not (isinstance(defn, model_definition) or
            isinstance(defn, fixed_model_definition)):
        raise ValueError("bad defn given")
    is_fixed = isinstance(defn, fixed_model_definition)
    return defn, is_fixed


def default_assign_kernel_config(defn):
    """Creates a default kernel configuration for sampling the assignment
    (clustering) vector. The default kernel is currently a gibbs sampler.

    Parameters
    ----------
    defn : mixturemodel definition

    """
    # XXX(stephentu): model_descriptors should implement
    # is_conjugate()

    def is_nonconj(x):
        return x.name() == 'bbnc'

    nonconj_indices = [
        idx for idx, x in enumerate(defn.models()) if is_nonconj(x)
    ]

    defn, is_fixed = _validate_definition(defn)
    if is_fixed:
        assert not nonconj_indices
        return ['assign_fixed']

    # assignment
    if nonconj_indices:
        # XXX(stephentu): 0.1 is arbitrary
        # XXX(stephentu): don't assume bbnc
        theta_config = {
            'tparams': {i: {'p': 0.1} for i in nonconj_indices}
        }
        kernels = [
            ('assign_resample', {'m': 10}),
            ('theta', theta_config),
        ]
    else:
        kernels = ['assign']

    return kernels


def default_feature_hp_kernel_config(defn):
    """Creates a default kernel configuration for sampling the component
    (feature) model hyper-parameters. The default kernel is currently
    a one-dimensional slice sampler.

    Parameters
    ----------
    defn : mixturemodel definition
        The hyper-priors set in the definition are used to configure the
        hyper-parameter sampling kernels.

    """
    defn, _ = _validate_definition(defn)

    # hyperparams
    hparams = {}
    for i, hp in enumerate(defn.hyperpriors()):
        if not hp:
            continue
        # XXX(stephentu): we are arbitrarily picking w=0.1
        hparams[i] = {k: (fn, 0.1) for k, fn in hp.iteritems()}

    return [('feature_hp', {'hparams': hparams})]


def default_cluster_hp_kernel_config(defn):
    """Creates a default kernel configuration for sampling the clustering
    (Chinese Restaurant Process) model hyper-parameter. The default kernel is
    currently a one-dimensional slice sampler.

    Parameters
    ----------
    defn : mixturemodel definition
        The hyper-priors set in the definition are used to configure the
        hyper-parameter sampling kernels.
    """
    defn, is_fixed = _validate_definition(defn)

    if is_fixed:
        # XXX(stephentu): cannot specify hyperprior on dirichlet yet
        # XXX(stephentu): should we throw an error here or print a warning?
        return []
    hp = defn.cluster_hyperprior()
    cparam = {k: (fn, 0.1) for k, fn in hp.iteritems()}
    return [('cluster_hp', {'cparam': cparam})]


def default_kernel_config(defn):
    """Creates a default kernel configuration suitable for general purpose
    inference. Currently configures an assignment sampler followed by a
    component hyper-parameter sampler.

    Parameters
    ----------
    defn : mixturemodel definition

    """
    # XXX(stephentu): should the default config also include cluster_hp?
    return list(it.chain(
        default_assign_kernel_config(defn),
        default_feature_hp_kernel_config(defn)))


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

        Possible values of `x` are:
        {'assign_fixed', 'assign', 'assign_resample',
         'feature_hp', 'cluster_hp'}

    """

    def __init__(self, defn, view, latent, kernel_config):
        defn, self._is_fixed = _validate_definition(defn)
        validator.validate_type(view, abstract_dataview, param_name='view')
        if not (isinstance(latent, state) or
                isinstance(latent, fixed_state)):
            raise ValueError("bad latent given")
        if self._is_fixed != isinstance(latent, fixed_state):
            raise ValueError("definition and latent don't match type")
        validator.validate_len(view, defn.n())

        def require_feature_indices(v):
            nfeatures = len(defn.models())
            valid_keys = set(xrange(nfeatures))
            if not set(v.keys()).issubset(valid_keys):
                msg = "bad config found: {}".format(v)
                raise ValueError(msg)

        self._defn = defn
        self._view = view
        self._latent = latent

        self._kernel_config = []
        for kernel in kernel_config:

            if hasattr(kernel, '__iter__'):
                name, config = kernel
            else:
                name, config = kernel, {}
            validator.validate_dict_like(config)

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
                if self._is_fixed:
                    raise ValueError("fixed_state cannot use variable kernel")
                if config.keys() != ['m']:
                    raise ValueError("bad config found: {}".format(config))
                validator.validate_positive(config['m'])

            elif name == 'feature_hp':
                if config.keys() != ['hparams']:
                    raise ValueError("bad config found: {}".format(config))
                require_feature_indices(config['hparams'])

            elif name == 'cluster_hp':
                if config.keys() != ['cparam']:
                    raise ValueError("bad config found: {}".format(config))
                if config['cparam'].keys() != ['alpha']:
                    msg = "bad config found: {}".format(config['cparam'])
                    raise ValueError(msg)

            elif name == 'theta':
                if config.keys() != ['tparams']:
                    raise ValueError("bad config found: {}".format(config))
                require_feature_indices(config['tparams'])

            else:
                raise ValueError("bad kernel found: {}".format(name))

            self._kernel_config.append((name, config))

    def run(self, r, niters=10000):
        """Run the specified mixturemodel kernel for `niters`, in a single
        thread.

        Parameters
        ----------
        r : random state
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')
        if self._is_fixed:
            model = bind_fixed(self._latent, self._view)
        else:
            model = bind(self._latent, self._view)
        for _ in xrange(niters):
            for name, config in self._kernel_config:
                if name == 'assign_fixed':
                    gibbs.assign_fixed(model, r)
                elif name == 'assign':
                    gibbs.assign(model, r)
                elif name == 'assign_resample':
                    gibbs.assign_resample(model, config['m'], r)
                elif name == 'feature_hp':
                    slice.hp(model, r, hparams=config['hparams'])
                elif name == 'cluster_hp':
                    slice.hp(model, r, cparam=config['cparam'])
                elif name == 'theta':
                    slice.theta(model, r, tparams=config['tparams'])
                else:
                    assert False, "should not be reach"

    def get_latent(self):
        """Returns the current value of the underlying state object.
        """
        return self._latent

    @property
    def expensive_state(self):
        return self._view

    @expensive_state.setter
    def expensive_state(self, view):
        self._view = view

    def expensive_state_digest(self, h):
        return self._view.digest(h)
