from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdint cimport uint8_t 
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx._models cimport factory
from microscopes.cxx._models_h cimport model as component_model
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._dataview cimport get_c_types
from microscopes.cxx.common._dataview_h cimport row_accessor, row_mutator, row_major_dataview
from microscopes.cxx.common._type_helper_h cimport GetOffsetsAndSize
from microscopes.cxx.common._rng cimport rng
cimport microscopes.cxx.common._type_info_h as ti
from microscopes.cxx.mixture._model_h cimport state as c_state

cimport numpy as np

cdef class state:
    cdef c_state *_thisptr
    cdef list _models
