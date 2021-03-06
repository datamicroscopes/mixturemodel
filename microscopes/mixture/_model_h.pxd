from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.set cimport set
from libcpp cimport bool as cbool
from libc.stddef cimport size_t

from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common.recarray._dataview_h cimport row_accessor, row_mutator, dataview
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.common._typedefs_h cimport hyperparam_bag_t, suffstats_bag_t
from microscopes.common._runtime_type_h cimport runtime_type
from microscopes.common._entity_state_h cimport entity_based_state_object
from microscopes._models_h cimport model as c_model

cdef extern from "microscopes/mixture/model.hpp" namespace "microscopes::mixture":

    cdef cppclass model_definition:
        model_definition(size_t, const vector[shared_ptr[c_model]] &) except +
        vector[runtime_type] get_runtime_types() except +
        size_t nmodels()


    cdef cppclass state:
        hyperparam_bag_t get_cluster_hp() except +
        void set_cluster_hp(const hyperparam_bag_t &) except +
        hyperparam_bag_t get_feature_hp(size_t) except +
        void set_feature_hp(size_t, const hyperparam_bag_t &) except +
        suffstats_bag_t get_suffstats(size_t, size_t) except +
        void set_suffstats(size_t, size_t, const suffstats_bag_t &) except +

        const vector[ssize_t] & assignments()
        const set[size_t] & empty_groups()

        size_t nentities()
        size_t ngroups()
        size_t groupsize(size_t) except +
        vector[size_t] groups() except +
        cbool isactivegroup(size_t)

        size_t create_group(rng_t &) except +
        void delete_group(size_t) except +

        void add_value(size_t, size_t, const row_accessor &, rng_t &) except +
        size_t remove_value(size_t, const row_accessor &, rng_t &) except +
        pair[vector[size_t], vector[float]] score_value(const row_accessor &, rng_t &) except +
        float score_data(const vector[size_t] &, const vector[size_t] &, rng_t &) except +

        void ensure_k_empty_groups(size_t, cbool, rng_t &) except +

        size_t sample_post_pred(const row_accessor &, row_mutator &, rng_t &) except +
        float score_assignment() except +
        float score_joint(rng_t &) except +

        # for debugging purposes
        void dcheck_consistency() except +

        string serialize() except +


    cdef cppclass model:
        model(const shared_ptr[state] &,
              const shared_ptr[dataview] &) except +
