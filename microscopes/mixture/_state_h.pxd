from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.common.recarray._dataview_h cimport dataview
from microscopes.mixture._model_h cimport model_definition, state

# this is annoying, we need to create a separate namespace to avoid name
# collisions

cdef extern from "microscopes/mixture/model.hpp" namespace "microscopes::mixture::state":
    shared_ptr[state] initialize(
            const model_definition &,
            const hyperparam_bag_t &,
            const vector[hyperparam_bag_t] &,
            const vector[size_t] &,
            dataview &,
            rng_t &) except +

    shared_ptr[state] deserialize(
            const model_definition &,
            const string &) except +
