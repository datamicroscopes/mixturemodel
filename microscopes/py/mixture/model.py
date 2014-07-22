import numpy as np
import numpy.ma as ma

from microscopes.py.common.groups import GroupManager
from microscopes.py.common.util import random_assignment_vector
from microscopes.io.schema_pb2 import \
    MixtureModelState as MixtureModelStateMessage, \
    MixtureModelGroup as MixtureModelGroupMessage
from distributions.dbg.random import sample_discrete_log, sample_discrete

def sample(defn, cluster_hp=None, feature_hps=None, r=None):
    """
    sample iid values from the generative process described by defn
    """
    dtypes = [m.py_desc().get_np_dtype() for m in defn._models]
    dtypes = np.dtype([('', dtype) for dtype in dtypes])
    cluster_counts = np.array([1], dtype=np.int)
    featuretypes = tuple(m.py_desc()._model_module for m in defn._models)
    featureshares = [t.Shared() for t in featuretypes]
    # init with defaults
    for share, m in zip(featureshares, defn._models):
        share.load(m.default_params())
    alpha = 1.0
    if cluster_hp is not None:
        alpha = float(cluster_hp['alpha'])
    if alpha <= 0.0:
        raise ValueError("alpha needs to be a positive real")
    if feature_hps is not None:
        if len(feature_hps) != len(defn._models):
            raise ValueError("invalid # of feature hps")
        for share, hp in zip(featureshares, feature_hps):
            share.load(hp)
    def init_sampler(arg):
        typ, s = arg
        samp = typ.Sampler()
        samp.init(s)
        return samp
    def new_cluster_params():
        return tuple(map(init_sampler, zip(featuretypes, featureshares)))
    def new_sample(params):
        data = tuple(samp.eval(s) for samp, s in zip(params, featureshares))
        return data
    cluster_params = [new_cluster_params()]
    samples = [[new_sample(cluster_params[-1])]]
    for _ in xrange(1, defn._n):
        dist = np.append(cluster_counts, alpha).astype(np.float, copy=False)
        choice = sample_discrete(dist)
        if choice == len(cluster_counts):
            cluster_counts = np.append(cluster_counts, 1)
            cluster_params.append(new_cluster_params())
            samples.append([new_sample(cluster_params[-1])])
        else:
            cluster_counts[choice] += 1
            params = cluster_params[choice]
            samples[choice].append(new_sample(params))
    return tuple(np.array(ys, dtype=dtypes) for ys in samples), \
           tuple(cluster_params)

class state(object):
    """
    state object API current has an (optional) unused random parameter, to make
    it conform to the C++ API. at some point we should thread the
    randomness through the python implementation (which is trickier since
    the underlying distributions python objects don't bother)
    """

    def __init__(self, defn, **kwargs):
        # XXX: let's get rid of this dependency (this dep also exists in our
        # C++ versions)
        self._defn = defn

        # type information
        self._featuretypes = tuple(
            m.py_desc()._model_module for m in defn._models)
        self._nomask = tuple(
                False for _ in xrange(len(self._featuretypes)))
        self._y_dtype = self._mk_y_dtype()

        if not (('data' in kwargs) ^ ('bytes' in kwargs)):
            raise ValueError("need exaclty one of `data' or `bytes'")

        if 'data' in kwargs:
            # handle the random initialization case
            if 'cluster_hp' in kwargs:
                cluster_hp = kwargs['cluster_hp']
            else:
                cluster_hp = {'alpha':1.}

            if 'feature_hps' in kwargs:
                feature_hps = kwargs['feature_hps']
                if len(feature_hps) != len(defn._models):
                    raise ValueError("expecting {} models, got {}".format(
                        len(feature_hps), len(defn._models)))
            else:
                feature_hps = [m.default_params() for m in defn._models]

            data = kwargs['data']
            self._groups = GroupManager(data.size())
            self._groups.set_hp(cluster_hp)

            def init_shared(args):
                typ, hp = args
                s = typ.Shared()
                s.load(hp)
                return s
            self._featureshares = map(
                    init_shared, zip(self._featuretypes, feature_hps))

            if 'assignment' in kwargs:
                assignment = kwargs['assignment']
                if len(assignment) != data.size():
                    raise ValueError("invalid assignment vector length")
                for s in assignment:
                    if s < 0:
                        raise ValueError("non-negative labels only")
            else:
                assignment = random_assignment_vector(data.size())
            ngroups = max(assignment) + 1
            for _ in xrange(ngroups):
                self.create_group()

            for i, yi in data.view(shuffle=False):
                self.add_value(assignment[i], i, yi)
        else:
            m = MixtureModelStateMessage()
            m.ParseFromString(kwargs['bytes'])

            if len(m.hypers) != len(defn._models):
                raise ValueError("model # mismatch")
            self._featureshares = []
            for raw, model in zip(m.hypers, defn._models):
                ps = model.py_desc()._pb_type.Shared()
                ps.ParseFromString(raw)
                s = model.py_desc()._model_module.Shared()
                s.load_protobuf(ps)
                self._featureshares.append(s)

            def group_deserialize(raw):
                m = MixtureModelGroupMessage()
                m.ParseFromString(raw)
                if len(m.suffstats) != len(defn._models):
                    raise ValueError("suffstat len mismatch")
                gdata = []
                for raw, model in zip(m.suffstats, defn._models):
                    pg = model.py_desc()._pb_type.Group()
                    pg.ParseFromString(raw)
                    g = model.py_desc()._model_module.Group()
                    g.load_protobuf(pg)
                    gdata.append(g)
                return gdata
            self._groups = GroupManager.deserialize(m.groups, group_deserialize)

    def serialize(self):
        m = MixtureModelStateMessage()
        for s, model in zip(self._featureshares, self._defn._models):
            pb = model.py_desc()._pb_type.Shared()
            s.dump_protobuf(pb)
            m.hypers.append(pb.SerializeToString())
        def group_serialize(gdata):
            m = MixtureModelGroupMessage()
            for g, model in zip(gdata, self._defn._models):
                pg = model.py_desc()._pb_type.Group()
                g.dump_protobuf(pg)
                m.suffstats.append(pg.SerializeToString())
            return m.SerializeToString()
        m.groups = self._groups.serialize(group_serialize)
        return m.SerializeToString()

    def _mk_y_dtype(self):
        models = self._defn._models
        dtypes = [m.py_desc().get_np_dtype() for m in models]
        return np.dtype([('', dtype) for dtype in dtypes])

    def get_feature_types(self):
        return list(self._featuretypes)

    def get_feature_dtypes(self):
        # XXX: make a copy
        return self._y_dtype

    def get_cluster_hp(self):
        return self._groups.get_hp()

    def set_cluster_hp(self, raw):
        self._groups.set_hp(raw)

    def get_feature_hp(self, fi):
        return self._featureshares[fi].dump()

    def set_feature_hp(self, fi, raw):
        self._featureshares[fi].load(raw)

    def get_feature_hp_shared(self, fi):
        return self._featureshares[fi]

    def get_suffstats(self, gid, fid):
        return self._groups.group_data(gid)[fid].dump()

    def set_suffstats(self, gid, fid, raw):
        self._groups.group_data(gid)[fid].load(raw)

    def assignments(self):
        return self._groups.assignments()

    def empty_groups(self):
        return self._groups.empty_groups()

    def ngroups(self):
        return self._groups.ngroups()

    def nentities(self):
        return self._groups.nentities()

    def nfeatures(self):
        return len(self._featuretypes)

    def groupsize(self, gid):
        return self._groups.groupsize(gid)

    def is_group_empty(self, gid):
        return not self._groups.nentities_in_group(gid)

    def groups(self):
        return [gid for gid, _ in self._groups.groupiter()]

    def create_group(self, rng=None):
        """
        returns gid
        """
        def init_group(arg):
            typ, shared = arg
            g = typ.Group()
            g.init(shared)
            return g
        gdata = map(init_group, zip(self._featuretypes, self._featureshares))
        return self._groups.create_group(gdata)

    def delete_group(self, gid):
        self._groups.delete_group(gid)

    def _mask(self, y):
        return y.mask if hasattr(y, 'mask') else self._nomask

    def add_value(self, gid, eid, y, rng=None):
        gdata = self._groups.add_entity_to_group(gid, eid)
        mask = self._mask(y)
        for (g, s), (yi, mi) in zip(zip(gdata, self._featureshares), zip(y, mask)):
            if not mi:
                g.add_value(s, yi)

    def remove_value(self, eid, y, rng=None):
        """
        returns gid
        """
        gid, gdata = self._groups.remove_entity_from_group(eid)
        mask = self._mask(y)
        for (g, s), (yi, mi) in zip(zip(gdata, self._featureshares), zip(y, mask)):
            if not mi:
                g.remove_value(s, yi)
        return gid

    def score_value(self, y, rng=None):
        """
        returns idmap, scores
        """
        scores = np.zeros(self._groups.ngroups(), dtype=np.float)
        idmap = [0]*self._groups.ngroups()
        n_empty_groups = len(self.empty_groups())
        assert n_empty_groups > 0
        # all empty groups share the alpha equally
        empty_group_alpha = self._groups.alpha() / n_empty_groups
        mask = self._mask(y)
        nentities = 0
        for idx, (gid, (cnt, gdata)) in enumerate(self._groups.groupiter()):
            lg_term1 = np.log(empty_group_alpha if not cnt else cnt)
            nentities += cnt
            lg_term2 = sum(0. if mi else g.score_value(s, yi) \
                for (g, s), (yi, mi) in \
                    zip(zip(gdata, self._featureshares), zip(y, mask)))
            scores[idx] = lg_term1 + lg_term2
            idmap[idx] = gid
        scores -= np.log(nentities + self._groups.alpha())
        return idmap, scores

    def score_data(self, features, groups, rng=None):
        """
        computes log p(Y_{fi} | C) = \sum{k=1}^{K} log p(Y_{fi}^{k}),
        where Y_{fi}^{k} is the slice of data along the fi-th feature belonging to the
        k-th cluster
        """
        if features is None:
            features = np.arange(len(self._featuretypes))
        elif not hasattr(features, '__iter__'):
            features = [features]
        if groups is None:
            groups = self.groups()
        elif not hasattr(groups, '__iter__'):
            groups = [groups]
        score = 0.0
        for gid in groups:
            gdata = self._groups.group_data(gid)
            for fi in features:
                score += gdata[fi].score_data(self._featureshares[fi])
        return score

    def sample_post_pred(self, y_new, rng=None):
        """
        draw a sample from p(y_new | C, Y)

        y_new is a masked array indicating which features to condition on. if
        y_new is None, then condition on no features

        this can be interpreted as "filling in the missing values"
        """

        if y_new is None:
            y_new = ma.zeros(len(self._featuretypes))
            y_new[:] = ma.masked

        # sample a cluster using the given values to weight the cluster
        # probabilities

        ### XXX: groups should really have a "resample from prior" interface
        ### so we don't have to keep creating and deleting groups
        empty_gids = list(self.empty_groups())
        for gid in empty_gids:
            self.delete_group(gid)
        egid = self.create_group()

        idmap, scores = self.score_value(y_new)

        gid = idmap[sample_discrete_log(scores)]
        gdata = self._groups.group_data(gid)

        # sample the missing values conditioned on the sampled cluster
        def pick(i):
            if y_new.mask[i]:
                return gdata[i].sample_value(self._featureshares[i])
            else:
                return y_new[i]
        return gid, np.array([tuple(map(pick, xrange(len(self._featuretypes))))], dtype=self._y_dtype)

    def score_assignment(self):
        """
        computes log p(C)
        """
        return self._groups.score_assignment()

    def score_joint(self):
        """
        computes log p(C, Y) = log p(C) + log p(Y|C)
        """
        return self.score_assignment() + self.score_data(None, None)

    def dcheck_consistency(self):
        # XXX: TODO
        pass

class model(object):
    def __init__(self, impl, data):
        self._impl = impl
        self._data = data

    def nentities(self):
        return self._impl.nentities()

    def ngroups(self):
        return self._impl.ngroups()

    def ncomponents(self):
        return self._impl.nfeatures()

    def assignments(self):
        return self._impl.assignments()

    def empty_groups(self):
        return self._impl.empty_groups()

    def groupsize(self, gid):
        return self._impl.groupsize(gid)

    def get_cluster_hp(self):
        return self._impl.get_cluster_hp()

    def set_cluster_hp(self, raw):
        self._impl.set_cluster_hp(raw)

    def get_component_hp(self, i):
        return self._impl.get_feature_hp(i)

    def set_component_hp(self, i, raw):
        self._impl.set_feature_hp(i, raw)

    def suffstats_identifiers(self, i):
        return self._impl.groups()

    def get_suffstats(self, fi, gi):
        return self._impl.get_suffstats(gi, fi)

    def set_suffstats(self, fi, gi, raw):
        self._impl.set_suffstats(gi, fi, raw)

    def add_value(self, gid, eid, rng=None):
        self._impl.add_value(gid, eid, self._data.get(eid), rng)

    def remove_value(self, eid, rng=None):
        return self._impl.remove_value(eid, self._data.get(eid), rng)

    def score_value(self, eid, rng=None):
        return self._impl.score_value(self._data.get(eid), rng)

    def score_assignment(self):
        return self._impl.score_assignment()

    def score_likelihood_indiv(self, fi, gi, rng=None):
        return self._impl.score_data([fi], [gi], rng)

    def score_likelihood(self, fi, rng=None):
        return self._impl.score_data([fi], None, rng)

    def create_group(self, rng=None):
        return self._impl.create_group(rng)

    def delete_group(self, gid):
        self._impl.delete_group(gid)

def initialize(defn, data, **kwargs):
    return state(defn=defn, data=data, **kwargs)

def deserialize(defn, bytes):
    return state(defn=defn, bytes=bytes)

def bind(state, data):
    return model(state, data)
