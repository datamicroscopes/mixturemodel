from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.set cimport set
from libcpp cimport bool as cbool
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.cxx.common.recarray._dataview_h cimport row_accessor, row_mutator
from microscopes.cxx.common._random_fwd_h cimport rng_t
from microscopes.cxx.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.cxx.common._runtime_type_h cimport runtime_type
from microscopes.cxx._models_h cimport model as component_model

cdef extern from "microscopes/mixture/model.hpp" namespace "microscopes::mixture":
    cdef cppclass state:
        state(size_t, vector[shared_ptr[component_model]] &) except +

        hyperparam_bag_t get_hp() except +
        void set_hp(hyperparam_bag_t &) except +
        hyperparam_bag_t get_feature_hp(size_t) except +
        void set_feature_hp(size_t, hyperparam_bag_t &) except +
        suffstats_bag_t get_suff_stats(size_t, size_t) except +
        void set_suff_stats(size_t, size_t, suffstats_bag_t &) except +

        vector[ssize_t] & assignments()
        set[size_t] & empty_groups()

        size_t nentities()
        size_t ngroups()
        size_t groupsize(size_t) except +
        vector[size_t] groups() except +

        size_t create_group(rng_t &) except +
        void delete_group(size_t) except +

        void add_value(size_t, size_t, row_accessor &, rng_t &) except +
        size_t remove_value(size_t, row_accessor &, rng_t &) except +
        pair[vector[size_t], vector[float]] score_value(row_accessor &, rng_t &) except +
        float score_data(vector[size_t] &, vector[size_t] &, rng_t &) except +

        void ensure_k_empty_groups(size_t, cbool, rng_t &) except +
        vector[runtime_type] get_runtime_types() except +

        size_t sample_post_pred(row_accessor &, row_mutator &, rng_t &) except +
        float score_assignment() except +
        float score_joint(rng_t &) except +

        # for debugging purposes
        void dcheck_consistency() except +
