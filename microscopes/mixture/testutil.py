"""Test helpers specific to mixture models

"""

from microscopes.mixture.model import sample
import numpy as np


def toy_dataset(defn):
    samples, _ = sample(defn)
    return np.hstack(samples)
