from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx.common._dataview_h cimport row_accessor, row_mutator
from microscopes.cxx.common._random_fwd_h cimport rng_t
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._type_info_h cimport runtime_type_info
from microscopes.cxx._models_h cimport model as component_model

cdef extern from "microscopes/mixture/model.hpp" namespace "microscopes::mixture":
    cdef cppclass state:
        state(size_t, vector[shared_ptr[component_model]] &) except +

        hyperparam_bag_t get_hp() except +
        void set_hp(hyperparam_bag_t &) except +
        hyperparam_bag_t get_feature_hp(size_t) except +
        void set_feature_hp(size_t, hyperparam_bag_t &) except +

        void ensure_k_empty_groups(size_t, rng_t &) except +
        void sample_post_pred(row_accessor &, row_mutator &, rng_t &) except +	
        vector[runtime_type_info] get_runtime_type_info() except +
