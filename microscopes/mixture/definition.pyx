# cython: embedsignature=True


from microscopes.models import model_descriptor
from microscopes.common import validator


cdef vector[shared_ptr[c_component_model]] get_cmodels(models):
    cdef vector[shared_ptr[c_component_model]] c_models
    for m in models:
        c_models.push_back((<_base> m._c_descriptor).get())
    return c_models


def _validate(models):
    validator.validate_nonempty(models)
    for m in models:
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
        The component likelihood models

    """

    def __cinit__(self, int n, int groups, models):
        validator.validate_positive(n)
        validator.validate_positive(groups)
        _validate(models)

        self._thisptr.reset(
            new c_fixed_model_definition(n, groups, get_cmodels(models)))
        self._n = n
        self._groups = groups
        self._models = list(models)

    def n(self):
        return self._n

    def groups(self):
        return self._groups

    def models(self):
        return self._models

    def __reduce__(self):
        args = (self._n, self._groups, self._models)
        return (_reconstruct_fixed_model_definition, args)


cdef class model_definition:
    """Structural definition for a dirichlet process mixture model

    Parameters
    ----------
    n : int
        Number of observations
    models : iterable of model descriptors
        The component likelihood models

    """

    def __cinit__(self, int n, models):
        validator.validate_positive(n)
        _validate(models)

        self._thisptr.reset(new c_model_definition(n, get_cmodels(models)))
        self._n = n
        self._models = list(models)

    def n(self):
        return self._n

    def models(self):
        return self._models

    def __reduce__(self):
        return (_reconstruct_model_definition, (self._n, self._models))


def _reconstruct_fixed_model_definition(n, groups, models):
    return fixed_model_definition(n, groups, models)


def _reconstruct_model_definition(n, models):
    return model_definition(n, models)
