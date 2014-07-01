import numpy as np
import numpy.ma as ma
from microscopes.io.schema_pb2 import CRP

def get_np_type(tpe):
    if tpe == ti.TYPE_INFO_B:
        return np.bool
    if tpe == ti.TYPE_INFO_I8:
        return np.int8
    if tpe == ti.TYPE_INFO_I16:
        return np.int16
    if tpe == ti.TYPE_INFO_I32:
        return np.int32
    if tpe == ti.TYPE_INFO_I64:
        return np.int64
    if tpe == ti.TYPE_INFO_F32:
        return np.float32
    if tpe == ti.TYPE_INFO_F64:
        return np.float64
    raise Exception("unknown type: " + tpe)

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

cdef class state:
    def __cinit__(self, n, list models):
        self._models = models
        cdef vector[shared_ptr[component_model]] cmodels
        for py_m, c_m in models:
            cmodels.push_back((<factory>c_m).new_cmodel())
        self._thisptr = new c_state(n, cmodels)

    def __dealloc__(self):
        del self._thisptr

    def get_cluster_hp(self):
        m = CRP()
        raw = str(self._thisptr[0].get_hp())
        m.ParseFromString(raw)
        return {'alpha':m.alpha}

    def set_cluster_hp(self, raw):
        m = CRP()
        m.alpha = float(raw['alpha'])
        self._thisptr[0].set_hp(m.SerializeToString())

    def get_feature_hp(self, int i):
        raw = str(self._thisptr[0].get_feature_hp(i))
        return self._models[i][0].shared_bytes_to_dict(raw)

    def set_feature_hp(self, int i, dict d):
        cdef hyperparam_bag_t raw = self._models[i][0].shared_dict_to_bytes(d)
        self._thisptr[0].set_feature_hp(i, raw)

    def get_suff_stats(self, int gid, int fid):
        raw = str(self._thisptr[0].get_suff_stats(gid, fid))
        return self._models[fid][0].group_bytes_to_dict(raw)

    def set_suff_stats(self, int gid, int fid, dict d):
        cdef suffstats_bag_t raw = self._models[fid][0].shared_dict_to_bytes(d)
        self._thisptr[0].set_suff_stats(gid, fid, raw)

    def assignments(self):
        cdef list ass = self._thisptr[0].assignments()
        return ass

    def empty_groups(self):
        cdef list egroups = list(self._thisptr[0].empty_groups())
        return egroups

    def ngroups(self):
        return self._thisptr[0].ngroups()

    def nentities(self):
        return self._thisptr[0].nentities()

    def nfeatures(self):
        return len(self._models)

    def groupsize(self, int gid):
        return self._thisptr[0].groupsize(gid)

    def is_group_empty(self, int gid):
        return not self._groups.nentities_in_group(gid)

    def groups(self):
        cdef list g = self._thisptr[0].groups()
        return g

    def create_group(self, rng r):
        return self._thisptr[0].create_group(r._thisptr[0])

    def delete_group(self, int gid):
        self._thisptr[0].delete_group(gid)

    def add_value(self, int gid, int eid, y, rng r):
        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr[0].get()
        self._thisptr[0].add_value(gid, eid, acc, r._thisptr[0])

    def remove_value(self, int eid, y, rng r):
        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr[0].get()
        self._thisptr[0].remove_value(eid, acc, r._thisptr[0])

    def score_value(self, y, rng r):
        cdef numpy_dataview view = get_dataview_for(y)
        cdef row_accessor acc = view._thisptr[0].get()
        cdef pair[vector[size_t], vector[float]] ret = self._thisptr[0].score_value(acc, r._thisptr[0])
        ret0 = list(ret.first)
        ret1 = np.array(list(ret.second))
        return ret0, ret1

    def score_data(self, features, rng r):
        if features is None:
            features = range(len(self._models))
        elif type(features) == int:
            features = [features]
        cdef vector[size_t] f
        for i in features:
            f.push_back(i)
        return self._thisptr[0].score_data(f, r._thisptr[0])

    def sample_post_pred(self, y_new, rng r):
        if y_new is None:
            D = self.nfeatures()
            y_new = ma.masked_array(
                np.array([tuple(0 for _ in xrange(D))], dtype=[('',int)]*D),
                mask=[tuple(True for _ in xrange(D))])

        cdef numpy_dataview view = get_dataview_for(y_new)
        cdef row_accessor acc = view._thisptr[0].get()

        # ensure the state has 1 empty group
        self._thisptr[0].ensure_k_empty_groups(1, r._thisptr[0])

        out_ctypes = self._thisptr[0].get_runtime_type_info()
        out_dtype = []
        for t in out_ctypes:
            out_dtype.append(('', get_np_type(t)))

        # build an appropriate numpy array to store the output
        cdef np.ndarray out_npd = np.zeros(1, dtype=out_dtype)

        # construct the output offsets
        cdef pair[vector[size_t], size_t] out_ret
        out_ret = GetOffsetsAndSize(out_ctypes)
        cdef vector[size_t] *out_offsets = &out_ret.first

        cdef row_mutator mut = row_mutator(
            <uint8_t *> out_npd.data,
            &out_ctypes,
            out_offsets)

        gid = self._thisptr[0].sample_post_pred(acc, mut, r._thisptr[0])

        return gid, out_npd

    def dcheck_consistency(self):
        self._thisptr[0].dcheck_consistency()
