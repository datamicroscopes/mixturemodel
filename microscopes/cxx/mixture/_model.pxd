from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx._models cimport factory
from microscopes.cxx._models_h cimport model as component_model
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._dataview cimport get_c_types, get_np_type
from microscopes.cxx.common.recarray._dataview cimport numpy_dataview, abstract_dataview
from microscopes.cxx.common.recarray._dataview_h cimport row_accessor, row_mutator, row_major_dataview
from microscopes.cxx.common._runtime_type_h cimport runtime_type
from microscopes.cxx.common._rng cimport rng
from microscopes.cxx.common._entity_state_h cimport entity_based_state_object as c_entity_based_state_object
from microscopes.cxx.common._entity_state import entity_based_state_object
from microscopes.cxx.common._entity_state cimport entity_based_state_object
from microscopes.cxx.mixture._model_h cimport state as c_state, bound_state as c_bound_state

cimport numpy as np

cdef class state:
    cdef shared_ptr[c_state] _thisptr
    cdef list _models
