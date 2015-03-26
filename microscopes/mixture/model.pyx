# cython: embedsignature=True


from microscopes.mixture._model import (
    state,
    bind,
    initialize,
    deserialize,
)

from microscopes.common import validator
from microscopes.common.random import sample_discrete
import numpy as np


def sample(defn, cluster_hp=None, feature_hps=None, r=None):
    """Sample i.i.d. values from the generative process described by `defn`.

    Parameters
    ----------
    defn : ``model_definition``
        The generative process
    cluster_hp : dict, optional
    feature_hps : iterable of dicts, optional
    r : ``rng``, optional

    Returns
    -------
    samples : tuple of samples in clusters
    params : tuple of Sampler objects, used to sample the clusters

    Notes
    -----
    Currently, the `r` parameter is ignored

    """
    dtypes = [m.py_desc().get_np_dtype() for m in defn.models()]
    dtypes = np.dtype([('', dtype) for dtype in dtypes])
    cluster_counts = np.array([1], dtype=np.int)
    featuretypes = tuple(m.py_desc()._model_module for m in defn.models())
    featureshares = [t.Shared() for t in featuretypes]
    # init with defaults
    for share, m in zip(featureshares, defn.models()):
        share.load(m.default_hyperparams())
    alpha = 1.0
    if cluster_hp is not None:
        alpha = float(cluster_hp['alpha'])
    validator.validate_positive(alpha, "alpha")
    if feature_hps is not None:
        validator.validate_len(feature_hps, len(defn.models()), "feature_hps")
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
    return (
        tuple(np.array(ys, dtype=dtypes) for ys in samples),
        tuple(cluster_params)
    )
