from libcpp.vector cimport vector
from libcpp.string cimport string

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._random_fwd_h cimport rng_t
from microscopes.cxx.common.recarray._dataview_h cimport dataview
from microscopes.cxx.mixture._model_h cimport fixed_model_definition, fixed_state

# this is annoying, we need to create a separate namespace to avoid name
# collisions

cdef extern from "microscopes/mixture/model.hpp" namespace "microscopes::mixture::fixed_state":
    shared_ptr[fixed_state] initialize(
            const fixed_model_definition &,
            const hyperparam_bag_t &,
            const vector[hyperparam_bag_t] &,
            dataview &,
            rng_t &) except +

    shared_ptr[fixed_state] deserialize(
            const fixed_model_definition &,
            const string &) except +
