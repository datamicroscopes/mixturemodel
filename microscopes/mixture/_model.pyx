# cython: embedsignature=True


# python imports
import numpy as np
import numpy.ma as ma

from microscopes.common._rng import rng
from microscopes.common._entity_state import \
    entity_based_state_object, \
    fixed_entity_based_state_object
from microscopes.common.recarray._dataview cimport \
    abstract_dataview
from microscopes.io.schema_pb2 import CRP
from distributions.io.schema_pb2 import DirichletDiscrete
from microscopes.common import validator

cdef numpy_dataview get_dataview_for(y):
    """
    creates a dataview for a single recarray

    not very efficient
    """

    cdef np.ndarray inp_data
    cdef np.ndarray inp_mask

    if hasattr(y, 'mask'):
        # deal with the mask
        inp_mask = np.ascontiguousarray(y.mask)
    else:
        inp_mask = None

    # this allows us to unify the two possible representations here
    # notice:
    # In [53]: y
    # Out[53]:
    # masked_array(data = [(--, 10.0)],
    #              mask = [(True, False)],
    #        fill_value = (True, 1e+20),
    #             dtype = [('f0', '?'), ('f1', '<f8')])
    #
    # In [54]: np.ascontiguousarray(y)
    # Out[54]:
    # array([(True, 10.0)],
    #       dtype=[('f0', '?'), ('f1', '<f8')])
    #
    # In [57]: np.ascontiguousarray(y[0])
    # Out[57]:
    # array([(True, 10.0)],
    #       dtype=[('f0', '?'), ('f1', '<f8')])

    inp_data = np.ascontiguousarray(y)
    if inp_mask is not None:
        inp_data = ma.array(inp_data, mask=inp_mask)

    return numpy_dataview(inp_data)

# XXX: fixed_state and state duplicate code for now

cdef class fixed_state:
    """The underlying state of a fixed group bayesian mixture model.
    """
    def __cinit__(self, fixed_model_definition defn, **kwargs):
        self._defn = defn
        cdef vector[hyperparam_bag_t] c_feature_hps_bytes
        cdef vector[size_t] c_assignment

        # note: python cannot overload __cinit__(), so we
        # use kwargs to handle both the random initialization case and
        # the deserialize from string case
        if not (('data' in kwargs) ^ ('bytes' in kwargs)):
            raise ValueError("need exaclty one of `data' or `bytes'")

        valid_kwargs = ('data', 'bytes', 'r',
                        'cluster_hp', 'feature_hps', 'assignment',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        if 'data' in kwargs:
            # handle the random initialization case

            data = kwargs['data']
            validator.validate_type(data, abstract_dataview, "data")
            validator.validate_len(data, defn.n(), "data")

            if 'r' not in kwargs:
                raise ValueError("need parameter `r'")
            r = kwargs['r']
            validator.validate_type(r, rng, "r")

            if 'cluster_hp' in kwargs:
                cluster_hp = kwargs['cluster_hp']
            else:
                cluster_hp = {'alphas': [1.] * defn._groups}

            def make_cluster_hp_bytes(cluster_hp):
                m = DirichletDiscrete.Shared()
                for alpha in cluster_hp['alphas']:
                    m.alphas.append(float(alpha))
                return m.SerializeToString()
            cluster_hp_bytes = make_cluster_hp_bytes(cluster_hp)

            if 'feature_hps' in kwargs:
                feature_hps = kwargs['feature_hps']
                validator.validate_len(
                    feature_hps, len(defn._models), "feature_hps")
            else:
                feature_hps = [m._default_params for m in defn._models]

            feature_hps_bytes = [
                m.py_desc().shared_dict_to_bytes(hp)
                for hp, m in zip(feature_hps, defn._models)]
            for s in feature_hps_bytes:
                c_feature_hps_bytes.push_back(s)

            if 'assignment' in kwargs:
                assignment = kwargs['assignment']
                validator.validate_len(assignment, data.size(), "assignment")
                for s in assignment:
                    validator.validate_in_range(s, len(defn._groups))
                    c_assignment.push_back(s)

            self._thisptr = c_initialize_fixed(
                defn._thisptr.get()[0],
                cluster_hp_bytes,
                c_feature_hps_bytes,
                c_assignment,
                (<abstract_dataview> data)._thisptr.get()[0],
                (<rng> r)._thisptr[0])
        else:
            # handle the deserialize case
            self._thisptr = c_deserialize_fixed(
                defn._thisptr.get()[0],
                kwargs['bytes'])

        if self._thisptr.get() == NULL:
            raise RuntimeError("could not properly construct fixed_state")

    # XXX: get rid of these introspection methods in the future
    def get_feature_types(self):
        models = self._defn._models
        types = [m.py_desc()._model_module for m in models]
        return types

    def get_feature_dtypes(self):
        models = self._defn._models
        dtypes = [('', m.py_desc().get_np_dtype()) for m in models]
        return np.dtype(dtypes)

    def get_cluster_hp(self):
        m = DirichletDiscrete.Shared()
        raw = str(self._thisptr.get().get_cluster_hp())
        m.ParseFromString(raw)
        return {'alphas': np.array(m.alphas)}

    def set_cluster_hp(self, dict raw):
        m = DirichletDiscrete.Shared()
        for alpha in raw['alphas']:
            m.alphas.append(float(alpha))
        self._thisptr.get().set_cluster_hp(m.SerializeToString())

    def _validate_eid(self, eid):
        validator.validate_in_range(eid, self.nentities())

    def _validate_fid(self, fid):
        validator.validate_in_range(fid, self.nfeatures())

    def _validate_gid(self, gid):
        validator.validate_in_range(gid, self.ngroups())

    def get_feature_hp(self, int i):
        self._validate_fid(i)
        models = self._defn._models
        raw = str(self._thisptr.get().get_feature_hp(i))
        return models[i].py_desc().shared_bytes_to_dict(raw)

    def set_feature_hp(self, int i, dict d):
        self._validate_fid(i)
        models = self._defn._models
        cdef hyperparam_bag_t raw = models[i].py_desc().shared_dict_to_bytes(d)
        self._thisptr.get().set_feature_hp(i, raw)

    def get_suffstats(self, int gid, int fid):
        self._validate_fid(fid)
        self._validate_gid(gid)
        models = self._defn._models
        raw = str(self._thisptr.get().get_suffstats(gid, fid))
        return models[fid].py_desc().group_bytes_to_dict(raw)

    def set_suffstats(self, int gid, int fid, dict d):
        self._validate_fid(fid)
        self._validate_gid(gid)
        models = self._defn._models
        cdef suffstats_bag_t raw = (
            models[fid].py_desc().shared_dict_to_bytes(d)
        )
        self._thisptr.get().set_suffstats(gid, fid, raw)

    def assignments(self):
        return list(self._thisptr.get().assignments())

    def ngroups(self):
        return self._thisptr.get().ngroups()

    def nentities(self):
        return self._thisptr.get().nentities()

    def nfeatures(self):
        return len(self._defn._models)

    def groupsize(self, int gid):
        self._validate_gid(gid)
        return self._thisptr.get().groupsize(gid)

    def is_group_empty(self, int gid):
        self._validate_gid(gid)
        return not self._groups.nentities_in_group(gid)

    def groups(self):
        cdef list g = self._thisptr.get().groups()
        return g

    def add_value(self, int gid, int eid, y, rng r):
        self._validate_gid(gid)
        self._validate_eid(eid)
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        self._thisptr.get().add_value(gid, eid, acc, r._thisptr[0])

    def remove_value(self, int eid, y, rng r):
        self._validate_eid(eid)
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        return self._thisptr.get().remove_value(eid, acc, r._thisptr[0])

    def score_value(self, y, rng r):
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        cdef pair[vector[size_t], vector[float]] ret = (
            self._thisptr.get().score_value(acc, r._thisptr[0])
        )
        ret0 = list(ret.first)
        ret1 = np.array(list(ret.second))
        return ret0, ret1

    def score_data(self, features, groups, rng r):
        validator.validate_not_none(r)
        if features is None:
            features = range(len(self._defn._models))
        elif not hasattr(features, '__iter__'):
            features = [features]

        if groups is None:
            groups = self.groups()
        elif not hasattr(groups, '__iter__'):
            groups = [groups]

        cdef vector[size_t] f
        for i in features:
            self._validate_fid(i)
            f.push_back(i)

        cdef vector[size_t] g
        for i in groups:
            self._validate_gid(i)
            g.push_back(i)

        return self._thisptr.get().score_data(f, g, r._thisptr[0])

    def sample_post_pred(self, y_new, rng r):
        # XXX: need to validate y
        validator.validate_not_none(r)
        if y_new is None:
            D = self.nfeatures()
            y_new = ma.masked_array(
                np.array([tuple(0 for _ in xrange(D))], dtype=[('', int)] * D),
                mask=[tuple(True for _ in xrange(D))])

        cdef numpy_dataview view = get_dataview_for(y_new)
        cdef row_accessor acc = view._thisptr.get().get()

        cdef vector[runtime_type] out_ctypes = \
            self._defn._thisptr.get().get_runtime_types()
        out_dtype = [('', get_np_type(t)) for t in out_ctypes]

        # build an appropriate numpy array to store the output
        cdef np.ndarray out_npd = np.zeros(1, dtype=out_dtype)

        cdef row_mutator mut = (
            row_mutator(<uint8_t *> out_npd.data, &out_ctypes)
        )
        gid = self._thisptr.get().sample_post_pred(acc, mut, r._thisptr[0])

        return gid, out_npd

    def score_assignment(self):
        return self._thisptr.get().score_assignment()

    def score_joint(self, rng r):
        validator.validate_not_none(r)
        return self._thisptr.get().score_joint(r._thisptr[0])

    def dcheck_consistency(self):
        self._thisptr.get().dcheck_consistency()

    def serialize(self):
        return self._thisptr.get().serialize()

cdef class state:
    """The underlying state of a Dirichlet Process mixture model.
    """
    def __cinit__(self, model_definition defn, **kwargs):
        self._defn = defn
        cdef vector[hyperparam_bag_t] c_feature_hps_bytes
        cdef vector[size_t] c_assignment

        # note: python cannot overload __cinit__(), so we
        # use kwargs to handle both the random initialization case and
        # the deserialize from string case
        if not (('data' in kwargs) ^ ('bytes' in kwargs)):
            raise ValueError("need exaclty one of `data' or `bytes'")

        valid_kwargs = ('data', 'bytes', 'r',
                        'cluster_hp', 'feature_hps', 'assignment',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        if 'data' in kwargs:
            # handle the random initialization case

            data = kwargs['data']
            validator.validate_type(data, abstract_dataview, "data")
            validator.validate_len(data, defn.n(), "data")

            if 'r' not in kwargs:
                raise ValueError("need parameter `r'")
            r = kwargs['r']
            validator.validate_type(r, rng, "r")

            if 'cluster_hp' in kwargs:
                cluster_hp = kwargs['cluster_hp']
            else:
                cluster_hp = {'alpha': 1.}

            def make_cluster_hp_bytes(cluster_hp):
                m = CRP()
                m.alpha = cluster_hp['alpha']
                return m.SerializeToString()
            cluster_hp_bytes = make_cluster_hp_bytes(cluster_hp)

            if 'feature_hps' in kwargs:
                feature_hps = kwargs['feature_hps']
                validator.validate_len(
                    feature_hps, len(defn._models), "feature_hps")
            else:
                feature_hps = [m.default_params() for m in defn._models]

            feature_hps_bytes = [
                m.py_desc().shared_dict_to_bytes(hp)
                for hp, m in zip(feature_hps, defn._models)]
            for s in feature_hps_bytes:
                c_feature_hps_bytes.push_back(s)

            if 'assignment' in kwargs:
                assignment = kwargs['assignment']
                validator.validate_len(assignment, data.size(), "assignment")
                for s in assignment:
                    validator.validate_nonnegative(s)
                    c_assignment.push_back(s)

            self._thisptr = c_initialize(
                defn._thisptr.get()[0],
                cluster_hp_bytes,
                c_feature_hps_bytes,
                c_assignment,
                (<abstract_dataview> data)._thisptr.get()[0],
                (<rng> r)._thisptr[0])
        else:
            # handle the deserialize case
            self._thisptr = c_deserialize(
                defn._thisptr.get()[0],
                kwargs['bytes'])

        if self._thisptr.get() == NULL:
            raise RuntimeError("could not properly construct state")

    # XXX: get rid of these introspection methods in the future
    def get_feature_types(self):
        models = self._defn._models
        types = [m.py_desc()._model_module for m in models]
        return types

    def get_feature_dtypes(self):
        models = self._defn._models
        dtypes = [('', m.py_desc().get_np_dtype()) for m in models]
        return np.dtype(dtypes)

    def get_cluster_hp(self):
        m = CRP()
        raw = str(self._thisptr.get().get_cluster_hp())
        m.ParseFromString(raw)
        return {'alpha': m.alpha}

    def set_cluster_hp(self, dict raw):
        m = CRP()
        m.alpha = float(raw['alpha'])
        self._thisptr.get().set_cluster_hp(m.SerializeToString())

    def _validate_eid(self, eid):
        validator.validate_in_range(eid, self.nentities())

    def _validate_fid(self, fid):
        validator.validate_in_range(fid, self.nfeatures())

    def _validate_gid(self, gid):
        if not self._thisptr.get().isactivegroup(gid):
            raise ValueError("invalid gid")

    def get_feature_hp(self, int i):
        self._validate_fid(i)
        raw = str(self._thisptr.get().get_feature_hp(i))
        models = self._defn._models
        return models[i].py_desc().shared_bytes_to_dict(raw)

    def set_feature_hp(self, int i, dict d):
        self._validate_fid(i)
        models = self._defn._models
        cdef hyperparam_bag_t raw = models[i].py_desc().shared_dict_to_bytes(d)
        self._thisptr.get().set_feature_hp(i, raw)

    def get_suffstats(self, int gid, int fid):
        self._validate_fid(fid)
        self._validate_gid(gid)
        models = self._defn._models
        raw = str(self._thisptr.get().get_suffstats(gid, fid))
        return models[fid].py_desc().group_bytes_to_dict(raw)

    def set_suffstats(self, int gid, int fid, dict d):
        self._validate_fid(fid)
        self._validate_gid(gid)
        models = self._defn._models
        cdef suffstats_bag_t raw = (
            models[fid].py_desc().shared_dict_to_bytes(d)
        )
        self._thisptr.get().set_suffstats(gid, fid, raw)

    def assignments(self):
        return list(self._thisptr.get().assignments())

    def empty_groups(self):
        return list(self._thisptr.get().empty_groups())

    def ngroups(self):
        return self._thisptr.get().ngroups()

    def nentities(self):
        return self._thisptr.get().nentities()

    def nfeatures(self):
        return len(self._defn._models)

    def groupsize(self, int gid):
        self._validate_gid(gid)
        return self._thisptr.get().groupsize(gid)

    def is_group_empty(self, int gid):
        self._validate_gid(gid)
        return not self._groups.nentities_in_group(gid)

    def groups(self):
        cdef list g = self._thisptr.get().groups()
        return g

    def create_group(self, rng r):
        assert r
        return self._thisptr.get().create_group(r._thisptr[0])

    def delete_group(self, int gid):
        self._validate_gid(gid)
        self._thisptr.get().delete_group(gid)

    def add_value(self, int gid, int eid, y, rng r):
        self._validate_gid(gid)
        self._validate_eid(eid)
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        self._thisptr.get().add_value(gid, eid, acc, r._thisptr[0])

    def remove_value(self, int eid, y, rng r):
        self._validate_eid(eid)
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        return self._thisptr.get().remove_value(eid, acc, r._thisptr[0])

    def score_value(self, y, rng r):
        # XXX: need to validate y
        validator.validate_not_none(r)

        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr.get().get()
        cdef pair[vector[size_t], vector[float]] ret = (
            self._thisptr.get().score_value(acc, r._thisptr[0])
        )
        ret0 = list(ret.first)
        ret1 = np.array(list(ret.second))
        return ret0, ret1

    def score_data(self, features, groups, rng r):
        validator.validate_not_none(r)
        if features is None:
            features = range(len(self._defn._models))
        elif not hasattr(features, '__iter__'):
            features = [features]

        if groups is None:
            groups = self.groups()
        elif not hasattr(groups, '__iter__'):
            groups = [groups]

        cdef vector[size_t] f
        for i in features:
            self._validate_fid(i)
            f.push_back(i)

        cdef vector[size_t] g
        for i in groups:
            self._validate_gid(i)
            g.push_back(i)

        return self._thisptr.get().score_data(f, g, r._thisptr[0])

    def sample_post_pred(self, y_new, rng r):
        # XXX: need to validate y
        validator.validate_not_none(r)
        if y_new is None:
            D = self.nfeatures()
            y_new = ma.masked_array(
                np.array([tuple(0 for _ in xrange(D))], dtype=[('', int)] * D),
                mask=[tuple(True for _ in xrange(D))])

        cdef numpy_dataview view = get_dataview_for(y_new)
        cdef row_accessor acc = view._thisptr.get().get()

        # ensure the state has 1 empty group
        self._thisptr.get().ensure_k_empty_groups(1, False, r._thisptr[0])

        cdef vector[runtime_type] out_ctypes = \
            self._defn._thisptr.get().get_runtime_types()
        out_dtype = [('', get_np_type(t)) for t in out_ctypes]

        # build an appropriate numpy array to store the output
        cdef np.ndarray out_npd = np.zeros(1, dtype=out_dtype)

        cdef row_mutator mut = (
            row_mutator(<uint8_t *> out_npd.data, &out_ctypes)
        )
        gid = self._thisptr.get().sample_post_pred(acc, mut, r._thisptr[0])

        return gid, out_npd

    def score_assignment(self):
        return self._thisptr.get().score_assignment()

    def score_joint(self, rng r):
        validator.validate_not_none(r)
        return self._thisptr.get().score_joint(r._thisptr[0])

    def dcheck_consistency(self):
        self._thisptr.get().dcheck_consistency()

    def serialize(self):
        return self._thisptr.get().serialize()


def bind_fixed(fixed_state s, abstract_dataview data):
    """
    """
    cdef shared_ptr[c_fixed_entity_based_state_object] px
    px.reset(new c_fixed_model(s._thisptr, data._thisptr))
    cdef fixed_entity_based_state_object ret = (
        fixed_entity_based_state_object(s._defn._models)
    )
    ret.set_fixed(px)
    ret._refs = data
    return ret


def bind(state s, abstract_dataview data):
    """
    """
    cdef shared_ptr[c_entity_based_state_object] px
    px.reset(new c_model(s._thisptr, data._thisptr))
    cdef entity_based_state_object ret = (
        entity_based_state_object(s._defn._models)
    )
    ret.set_entity(px)
    ret._refs = data
    return ret


def initialize_fixed(fixed_model_definition defn,
                     abstract_dataview data,
                     rng r,
                     **kwargs):
    """
    """
    return fixed_state(defn=defn, data=data, r=r, **kwargs)


def initialize(model_definition defn,
               abstract_dataview data,
               rng r,
               **kwargs):
    """
    """
    return state(defn=defn, data=data, r=r, **kwargs)


def deserialize_fixed(fixed_model_definition defn, bytes):
    """
    """
    return fixed_state(defn=defn, bytes=bytes)


def deserialize(model_definition defn, bytes):
    """
    """
    return state(defn=defn, bytes=bytes)
