# cython: embedsignature=True


from microscopes.models import model_descriptor
from microscopes.common import validator
from microscopes.common.scalar_functions import log_exponential
import operator as op
import copy


cdef vector[shared_ptr[c_component_model]] get_cmodels(models):
    cdef vector[shared_ptr[c_component_model]] c_models
    for m in models:
        c_models.push_back((<_base> m._c_descriptor).get())
    return c_models


def _validate(models):
    validator.validate_nonempty(models)
    for m in models:
        if hasattr(m, '__len__'):
            validator.validate_len(m, 2)
            validator.validate_type(m[0], model_descriptor)
            validator.validate_type(m[1], dict)
        else:
            validator.validate_type(m, model_descriptor)


cdef class fixed_model_definition:
    """Structural definition for a fixed mixture model

    Parameters
    ----------
    n : int
        Number of observations
    groups : int
        Number of groups (fixed)
    models : iterable of model descriptors
        The component likelihood models. Each element is either `x` or
        `(x, y)`, where `x` is a ``model_descriptor`` and `y` is a dict
        containing the hyperpriors. If `y` is not given, then the default
        hyperpriors are used per model.

    Notes
    -----
    There is currently no interface to express a hyperprior on the dirichlet
    distribution which governs the clustering behavior.

    This class is not meant to be sub-classable.

    """

    def __cinit__(self, int n, int groups, models):
        validator.validate_positive(n)
        validator.validate_positive(groups)
        _validate(models)

        self._n = n
        self._groups = groups
        self._models = []
        for model in models:
            if hasattr(model, '__len__'):
                m, hp = model
            else:
                m, hp = model, model.default_hyperpriors()
            self._models.append((m, hp))

        self._thisptr.reset(
            new c_fixed_model_definition(
                n,
                groups,
                get_cmodels(map(op.itemgetter(0), self._models))))

    def n(self):
        return self._n

    def groups(self):
        return self._groups

    def models(self):
        return map(op.itemgetter(0), self._models)

    def hyperpriors(self):
        return map(op.itemgetter(1), self._models)

    def __reduce__(self):
        args = (self._n, self._groups, self._models)
        return (_reconstruct_fixed_model_definition, args)

    def __copy__(self):
        res = fixed_model_definition(self._n, self._groups, self._models)
        return res

    def __deepcopy__(self, memo):
        models = copy.deepcopy(self._models, memo)
        res = fixed_model_definition(self._n, self._groups, models)
        return res


cdef class model_definition:
    """Structural definition for a dirichlet process mixture model

    Parameters
    ----------
    n : int
        Number of observations
    models : iterable of model descriptors
        The component likelihood models. Each element is either `x` or
        `(x, y)`, where `x` is a ``model_descriptor`` and `y` is a dict
        containing the hyperpriors. If `y` is not given, then the default
        hyperpriors are used per model.
    cluster_hyperprior : dict, optional
        Describes the hyperior for the CRP

    Notes
    -----
    This class is not meant to be sub-classable.

    """

    def __cinit__(self,
                  int n,
                  models,
                  cluster_hyperprior={'alpha': log_exponential(1.)}):
        validator.validate_positive(n)
        _validate(models)

        self._n = n
        self._models = []
        for model in models:
            if hasattr(model, '__len__'):
                m, hp = model
            else:
                m, hp = model, model.default_hyperpriors()
            self._models.append((m, hp))

        validator.validate_type(cluster_hyperprior, dict)
        if cluster_hyperprior.keys() != ['alpha']:
            msg = "invalid cluster hp: {}".format(cluster_hyperprior)
            raise ValueError(msg)
        self._cluster_hyperprior = cluster_hyperprior

        self._thisptr.reset(
            new c_model_definition(
                n,
                get_cmodels(map(op.itemgetter(0), self._models))))

    def n(self):
        return self._n

    def models(self):
        return map(op.itemgetter(0), self._models)

    def hyperpriors(self):
        return map(op.itemgetter(1), self._models)

    def cluster_hyperprior(self):
        return self._cluster_hyperprior

    def __reduce__(self):
        args = (self._n, self._models, self._cluster_hyperprior)
        return (_reconstruct_model_definition, args)

    def __copy__(self):
        args = self._n, self._groups, self._models, self._cluster_hyperprior
        res = model_definition(*args)
        return res

    def __deepcopy__(self, memo):
        models = copy.deepcopy(self._models, memo)
        cluster_hyperprior = copy.deepcopy(self._cluster_hyperprior, memo)
        args = self._n, self._groups, models, cluster_hyperprior
        res = model_definition(*args)
        return res


def _reconstruct_fixed_model_definition(n, groups, models):
    return fixed_model_definition(n, groups, models)


def _reconstruct_model_definition(n, models, cluster_hyperprior):
    return model_definition(n, models, cluster_hyperprior)
