import numpy as np
import numpy.ma as ma

from microscopes.py.common.groups import FixedNGroupManager
from distributions.dbg.random import sample_discrete_log, sample_discrete

class state(object):

    def __init__(self, n, featuretypes):
        self._groups = FixedNGroupManager(n)
        self._alpha = None
        self._featuretypes = featuretypes
        def init_shared(typ):
            return typ.Shared()
        self._featureshares = map(init_shared, self._featuretypes)
        self._nomask = tuple(False for _ in xrange(len(self._featuretypes)))
        self._y_dtype = self._mk_y_dtype()

    def _mk_dtype_desc(self, fi):
        typ, shared = self._featuretypes[fi], self._featureshares[fi]
        if hasattr(shared, 'dimension') and shared.dimension() > 1:
            return ('', typ.Value, (shared.dimension(),))
        return ('', typ.Value)

    def _mk_y_dtype(self):
        return [self._mk_dtype_desc(i) for i in xrange(len(self._featuretypes))]

    def get_cluster_hp(self):
        return {'alpha':self._alpha}

    def set_cluster_hp(self, raw):
        self._alpha = float(raw['alpha'])

    def get_feature_hp(self, fi):
        return self._featureshares[fi].dump()

    def set_feature_hp(self, fi, raw):
        self._featureshares[fi].load(raw)

    def get_feature_hp_shared(self, fi):
        return self._featureshares[fi]

    def get_suff_stats(self, gid, fid):
        return self._groups.group_data(gid)[fid].dump()

    def set_suff_stats(self, gid, fid, raw):
        self._groups.group_data(gid)[fid].load(raw)

    def assignments(self):
        return self._groups.assignments()

    def empty_groups(self):
        return self._groups.empty_groups()

    def ngroups(self):
        return self._groups.ngroups()

    def nentities(self):
        return self._groups.nentities()

    def groupsize(self, gid):
        return self._groups.groupsize(gid)

    def is_group_empty(self, gid):
        return not self._groups.nentities_in_group(gid)

    def groups(self):
        return [gid for gid, _ in self._groups.groupiter()]

    def create_group(self):
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

    def add_value(self, gid, eid, y):
        gdata = self._groups.add_entity_to_group(gid, eid)
        mask = self._mask(y)
        for (g, s), (yi, mi) in zip(zip(gdata, self._featureshares), zip(y, mask)):
            if not mi:
                g.add_value(s, yi)

    def remove_value(self, eid, y):
        """
        returns gid
        """
        gid, gdata = self._groups.remove_entity_from_group(eid)
        mask = self._mask(y)
        for (g, s), (yi, mi) in zip(zip(gdata, self._featureshares), zip(y, mask)):
            if not mi:
                g.remove_value(s, yi)
        return gid

    def score_value(self, y):
        """
        returns idmap, scores
        """
        scores = np.zeros(self._groups.ngroups(), dtype=np.float)
        idmap = [0]*self._groups.ngroups()
        n_empty_groups = len(self.empty_groups())
        assert n_empty_groups > 0
        empty_group_alpha = self._alpha / n_empty_groups # all empty groups share the alpha equally
        mask = self._mask(y)
        nentities = 0
        for idx, (gid, (cnt, gdata)) in enumerate(self._groups.groupiter()):
            lg_term1 = np.log(empty_group_alpha if not cnt else cnt)
            nentities += cnt
            lg_term2 = sum(0. if mi else g.score_value(s, yi) for (g, s), (yi, mi) in zip(zip(gdata, self._featureshares), zip(y, mask)))
            scores[idx] = lg_term1 + lg_term2
            idmap[idx] = gid
        scores -= np.log(nentities + self._alpha)
        return idmap, scores

    def score_data(self, features=None):
        """
        computes log p(Y_{fi} | C) = \sum{k=1}^{K} log p(Y_{fi}^{k}),
        where Y_{fi}^{k} is the slice of data along the fi-th feature belonging to the
        k-th cluster
        """

        if features is None:
            features = np.arange(len(self._featuretypes))
        elif type(features) == int:
            features = [features]

        score = 0.0
        for _, (_, gdata) in self._groups.groupiter():
            for fi in features:
                score += gdata[fi].score_data(self._featureshares[fi])
        return score

    def score_assignment(self):
        """
        computes log p(C)
        """
        # CRP
        lg_sum = 0.0
        assignments = self._groups.assignments()
        counts = { assignments[0] : 1 }
        for i, ci in enumerate(assignments):
            if i == 0:
                continue
            cnt = counts.get(ci, 0)
            numer = cnt if cnt else self._alpha
            denom = i + self._alpha
            lg_sum += np.log(numer / denom)
            counts[ci] = cnt + 1
        return lg_sum

    def score_joint(self):
        """
        computes log p(C, Y) = log p(C) + log p(Y|C)
        """
        return self.score_assignment() + self.score_data(features=None, groups=None)

    def sample_post_pred(self, y_new):
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

    #def reset(self):
    #    """
    #    reset to the same condition as upon construction
    #    """
    #    self._groups = FixedNGroupManager(self._groups.nentities())

    #def bootstrap(self, it):
    #    """
    #    bootstraps assignments
    #    """
    #    assert not self.ngroups()
    #    assert self._groups.no_entities_assigned()

    #    ei0, y0 = next(it)
    #    gid0 = self.create_group()
    #    self.add_entity_to_group(gid0, ei0, y0)
    #    empty_gid = self.create_group()
    #    for ei, yi in it:
    #        idmap, scores = self.score_value(yi)
    #        gid = idmap[sample_discrete_log(scores)]
    #        self.add_entity_to_group(gid, ei, yi)
    #        if gid == empty_gid:
    #            empty_gid = self.create_group()

    #    assert self._groups.all_entities_assigned()

    #def fill(self, clusters):
    #    """
    #    form a cluster assignment and sufficient statistics out of an given
    #    clustering of N points.

    #    useful to bootstrap a model as the ground truth model
    #    """
    #    assert not self.ngroups()
    #    assert self._groups.no_entities_assigned()

    #    counts = [c.shape[0] for c in clusters]
    #    cumcounts = np.cumsum(counts)
    #    gids = [self.create_group() for _ in xrange(len(clusters))]
    #    for cid, (gid, data) in enumerate(zip(gids, clusters)):
    #        off = cumcounts[cid-1] if cid else 0
    #        for ei, yi in enumerate(data):
    #            self.add_entity_to_group(gid, off + ei, yi)

    #    assert self._groups.all_entities_assigned()


    #def sample(self, n):
    #    """
    #    generate n iid samples from the underlying generative process described by this DirichletProcess.

    #    does not affect the state of the DirichletProcess, and only depends on the prior parameters of the
    #    DirichletProcess

    #    returns a tuple of
    #        (
    #            k-length tuple of observations, where k is the # of sampled clusters from the CRP,
    #            k-length tuple of cluster samplers
    #        )
    #    """
    #    cluster_counts = np.array([1], dtype=np.int)
    #    def init_sampler(arg):
    #        typ, s = arg
    #        samp = typ.Sampler()
    #        samp.init(s)
    #        return samp
    #    def new_cluster_params():
    #        return tuple(map(init_sampler, zip(self._featuretypes, self._featureshares)))
    #    def new_sample(params):
    #        data = tuple(samp.eval(s) for samp, s in zip(params, self._featureshares))
    #        return data
    #    cluster_params = [new_cluster_params()]
    #    samples = [[new_sample(cluster_params[-1])]]
    #    for _ in xrange(1, n):
    #        dist = np.append(cluster_counts, self._alpha).astype(np.float, copy=False)
    #        choice = sample_discrete(dist)
    #        if choice == len(cluster_counts):
    #            cluster_counts = np.append(cluster_counts, 1)
    #            cluster_params.append(new_cluster_params())
    #            samples.append([new_sample(cluster_params[-1])])
    #        else:
    #            cluster_counts[choice] += 1
    #            params = cluster_params[choice]
    #            samples[choice].append(new_sample(params))
    #    return tuple(np.array(ys, dtype=self._y_dtype) for ys in samples), tuple(cluster_params)
