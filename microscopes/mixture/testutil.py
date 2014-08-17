"""Test helpers specific to mixture models

"""

from microscopes.common.rng import rng
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.mixture.model import sample, initialize
from microscopes.common.testutil import dist_on_all_clusterings

import numpy as np


def toy_dataset(defn):
    samples, _ = sample(defn)
    return np.hstack(samples)


def data_with_posterior(defn,
                        cluster_hp=None,
                        feature_hps=None,
                        preprocess_data_fn=None,
                        r=None):
    if r is None:
        r = rng()
    Y_clusters, _ = sample(defn, cluster_hp, feature_hps, r)
    Y = np.hstack(Y_clusters)
    if preprocess_data_fn:
        Y = preprocess_data_fn(Y)
    data = numpy_dataview(Y)

    def score_fn(assignment):
        s = initialize(defn,
                       data,
                       r,
                       cluster_hp=cluster_hp,
                       feature_hps=feature_hps,
                       assignment=assignment)
        return s.score_joint(r)

    posterior = dist_on_all_clusterings(score_fn, defn.n())
    return Y, posterior
