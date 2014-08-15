from libcpp.vector cimport vector

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes._models cimport _base
from microscopes._models_h cimport model as c_component_model
from microscopes.mixture._model_h cimport (
    fixed_model_definition as c_fixed_model_definition,
    model_definition as c_model_definition,
)


cdef class fixed_model_definition:
    # ideally would not be shared pointer, but
    # doesn't have no-arg ctor
    cdef shared_ptr[c_fixed_model_definition] _thisptr
    cdef readonly int _n
    cdef readonly int _groups
    cdef readonly list _models


cdef class model_definition:
    # ideally would not be shared pointer, but
    # doesn't have no-arg ctor
    cdef shared_ptr[c_model_definition] _thisptr
    cdef readonly int _n
    cdef readonly list _models
    cdef readonly dict _cluster_hyperprior
