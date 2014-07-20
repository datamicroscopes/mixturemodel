# cimports
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.string cimport string
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx._models_h cimport model as c_component_model
from microscopes.cxx.common._typedefs_h cimport \
    hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._dataview cimport get_c_types, get_np_type
from microscopes.cxx.common.recarray._dataview cimport \
    numpy_dataview, abstract_dataview
from microscopes.cxx.common.recarray._dataview_h cimport \
    row_accessor, row_mutator, row_major_dataview
from microscopes.cxx.common._runtime_type_h cimport runtime_type
from microscopes.cxx.common._rng cimport rng
from microscopes.cxx.common._entity_state_h cimport \
    entity_based_state_object as c_entity_based_state_object, \
    fixed_entity_based_state_object as c_fixed_entity_based_state_object
from microscopes.cxx.common._entity_state cimport \
    entity_based_state_object, \
    fixed_entity_based_state_object
from microscopes.mixture.definition cimport \
    fixed_model_definition, \
    model_definition
from microscopes.cxx.mixture._model_h cimport \
    fixed_state as c_fixed_state, \
    state as c_state, \
    fixed_model as c_fixed_model, \
    model as c_model
from microscopes.cxx.mixture._fixed_state_h cimport \
    initialize as c_initialize_fixed, \
    deserialize as c_deserialize_fixed
from microscopes.cxx.mixture._state_h cimport \
    initialize as c_initialize, \
    deserialize as c_deserialize
cimport numpy as np

# python imports
from microscopes.cxx.common._entity_state import \
    entity_based_state_object, \
    fixed_entity_based_state_object

cdef class fixed_state:
    cdef shared_ptr[c_fixed_state] _thisptr

    # XXX: the type/structure information below is not technically
    # part of the model, and we should find a way to remove this
    # in the future
    cdef fixed_model_definition _defn

cdef class state:
    cdef shared_ptr[c_state] _thisptr

    # XXX: the type/structure information below is not technically
    # part of the model, and we should find a way to remove this
    # in the future
    cdef model_definition _defn
