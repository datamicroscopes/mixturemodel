from microscopes.models import model_descriptor
from microscopes.common import validator

cdef vector[shared_ptr[c_component_model]] get_cmodels(models):
    cdef vector[shared_ptr[c_component_model]] c_models
    for m in models:
        c_models.push_back((<_base>m._c_descriptor).get())
    return c_models

def _validate(models):
    validator.validate_nonempty(models)
    for m in models:
        validator.validate_type(m, model_descriptor)

cdef class fixed_model_definition:
    def __cinit__(self, int n, int groups, models):
        validator.validate_positive(n)
        validator.validate_positive(groups)
        _validate(models)

        self._thisptr.reset(
            new c_fixed_model_definition(n, groups, get_cmodels(models)))
        self._n = n
        self._groups = groups
        self._models = list(models)

cdef class model_definition:
    def __cinit__(self, int n, models):
        validator.validate_positive(n)
        _validate(models)

        self._thisptr.reset(new c_model_definition(n, get_cmodels(models)))
        self._n = n
        self._models = list(models)
