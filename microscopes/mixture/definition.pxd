from libcpp.vector cimport vector

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx._models cimport _base
from microscopes.cxx._models_h cimport model as c_component_model
from microscopes.cxx.mixture._model_h cimport \
    fixed_model_definition as c_fixed_model_definition, \
    model_definition as c_model_definition

cdef class fixed_model_definition:
    # ideally would not be shared pointer, but
    # doesn't have no-arg ctor
    cdef shared_ptr[c_fixed_model_definition] _thisptr
    cdef public int _groups
    cdef public list _models

cdef class model_definition:
    # ideally would not be shared pointer, but
    # doesn't have no-arg ctor
    cdef shared_ptr[c_model_definition] _thisptr
    cdef public list _models
