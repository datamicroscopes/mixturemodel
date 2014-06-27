import numpy as np
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

cdef class state:
    def __cinit__(self, n, list models):
        self._models = models
        cdef vector[shared_ptr[component_model]] cmodels
        for py_m, c_m in models:
            cmodels.push_back((<factory>c_m).new_cmodel())
        self._thisptr = new c_state(n, cmodels)

    def __dealloc__(self):
        del self._thisptr

    def get_hp(self):
        m = CRP()
        raw = str(self._thisptr[0].get_hp())
        m.ParseFromString(raw)
        return {'alpha':m.alpha}

    def set_hp(self, raw):
        m = CRP()
        m.alpha = float(raw['alpha'])
        self._thisptr[0].set_hp(m.SerializeToString())

    def get_feature_hp(self, int i):
        raw = str(self._thisptr[0].get_feature_hp(i))
        return self._models[i][0].shared_bytes_to_dict(raw)

    def set_feature_hp(self, int i, dict d):
        cdef hyperparam_bag_t raw = self._models[i][0].shared_dict_to_bytes(d) 
        self._thisptr[0].set_feature_hp(i, raw)

    def sample_post_pred(self, np.ndarray inp, rng rng, size=1):
        ret = [self._sample_post_pred_one(inp, rng) for _ in xrange(size)]
        return np.hstack(ret)

    def _sample_post_pred_one(self, np.ndarray inp, rng rng):
        cdef np.ndarray inp_data = None
        cdef np.ndarray inp_mask = None
        cdef vector[ti.runtime_type_info] inp_ctypes
        cdef vector[ti.runtime_type_info] out_ctypes

        if hasattr(inp, 'mask'):
            inp_data = np.ascontiguousarray(inp.data)
            inp_mask = np.ascontiguousarray(inp.mask)
        else:
            inp_data = np.ascontiguousarray(inp.data)

        inp_ctypes = get_c_types(inp)
        cdef pair[vector[size_t], size_t] inp_ret     
        inp_ret = GetOffsetsAndSize(inp_ctypes)
        cdef vector[size_t] *inp_offsets = &inp_ret.first

        # build row_accessor
        # XXX: can we stack allocate?
        cdef row_accessor *acc = new row_accessor( 
            <uint8_t *> inp_data.data, 
            <cbool *> inp_mask.data if inp_mask is not None else NULL,
            &inp_ctypes,
            inp_offsets)

        # ensure the state has 1 empty group
        self._thisptr[0].ensure_k_empty_groups(1, rng._thisptr[0])     

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

        cdef row_mutator *mut = new row_mutator(
            <uint8_t *> out_npd.data,
            &out_ctypes,
            out_offsets)

        self._thisptr[0].sample_post_pred(acc[0], mut[0], rng._thisptr[0])

        del acc
        del mut
        return out_npd
