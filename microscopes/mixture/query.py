"""The query interface for mixturemodels.

Note that the methods of this interface all take a list of latent state objects
(as opposed to a single latent).

"""

import numpy as np
import operator as op
from scipy.stats import mode
from microscopes.common import query, validator


def zmatrix(latents):
    """Compute a z-matrix (cluster co-assignment matrix). The ij-th entry of a
    z-matrix is a real value scalar between [0, 1] indicating the frequency of
    how often entities i and j appear in the same cluster.

    Parameters
    ----------
    latents : list of mixturemodel latent objects
        The latents should all be points in the state space of the same
        structural model. The implementation currently does not check for this.

    Returns
    -------
    zmat : ndarray
        a (N, N) shape matrix

    Notes
    -----
    Currently does not support a sparse zmatrix representation, so only use
    this for small N.

    """
    return query.zmatrix([latent.assignments() for latent in latents])


def posterior_predictive(q, latents, r, samples_per_chain=1):
    """Generate a bag of samples from the posterior distribution of each
    mixturemodel state object.

    Parameters
    ----------
    q : masked recarray
        The query object
    latents : list of mixturemodel latent objects
    r : random state
    samples_per_chain : int, optional
        Default is 1.

    Returns
    -------
    samples : (N,) recarray
        where `N = len(latents) * samples_per_chain`

    """

    if not len(latents):
        raise ValueError("no latents given")
    validator.validate_positive(
        samples_per_chain, param_name='samples_per_chain')

    samples = []
    for latent in latents:
        for _ in xrange(samples_per_chain):
            samples.append(latent.sample_post_pred(q, r)[1])
    return np.hstack(samples)


def _is_discrete_dtype(dtype):
    # XXX(stephentu): is there a better way?
    return (np.issubdtype(dtype, np.integer) or
            np.issubdtype(dtype, np.bool))


def posterior_predictive_statistic(q,
                                   latents,
                                   r,
                                   samples_per_chain=1,
                                   merge='avg'):
    """Sample many values and combine each feature independently using the
    given `merge` strategy.

    Parameters
    ----------
    q : masked recarray
        The query object
    latents : list of mixturemodel latent objects
    r : random state
    samples_per_chain : int, optional
        Default is 1.
    merge : str or list of strs, each str is one of {'avg', 'mode'}
        Note that 'mode' only works for discrete data types.

    Returns
    -------
    statistic : (1,) recarray

    Notes
    -----
    This method exists as a convenience, primarily because ndarray methods such
    as `mean()` do not work with recarrays.

    """

    samples = posterior_predictive(
        q, latents, r, samples_per_chain=samples_per_chain)

    nfeatures = len(samples.dtype)

    # NOTE: samples.dtype is not iterable
    dtypes = [samples.dtype[i] for i in xrange(nfeatures)]

    if not hasattr(merge, '__iter__'):
        merge = [merge] * nfeatures

    for strat, dtype in zip(merge, dtypes):
        if strat not in ('avg', 'mode'):
            raise ValueError("bad merge strategy: {}".format(strat))
        if strat == 'mode' and not _is_discrete_dtype(dtype):
            msg = ("`mode' merge strategy cannot work "
                   "with non-integral types: {}").format(dtype)
            raise ValueError(msg)

    values = [[] for _ in xrange(nfeatures)]
    for sample in samples:
        for lst, v in zip(values, sample):
            lst.append(v)

    values = [np.array(v, dtype=dtype) for v, dtype in zip(values, dtypes)]

    def statistic(value, strat):
        if strat == 'avg':
            mean = value.mean()
            return mean, mean.dtype
        elif strat == 'mode':
            # scipy.stats.mode() is weird
            arr = mode(value, axis=None)[0]
            assert arr.shape == (1,)
            return arr[0], value.dtype
        else:
            assert False, 'should not be reached'

    stat_with_dtypes = (
        [statistic(value, strat) for value, strat in zip(values, merge)]
    )

    return np.array(
        [tuple(map(op.itemgetter(0), stat_with_dtypes))],
        dtype=[('', dt) for _, dt in stat_with_dtypes])
