import argparse
from microscopes.common.recarray.dataview import numpy_dataview
from microscopes.common.rng import rng
from microscopes.common.scalar_functions import log_exponential
from microscopes.mixture.model import initialize, bind
from microscopes.kernels.gibbs import assign
from microscopes.kernels.slice import hp
from microscopes.common.util import mkdirp
from microscopes.models import bb, dd
from microscopes.mixture.definition import model_definition

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    #roc_curve,
)

import numpy as np
import numpy.ma as ma
import math
import time
import os

from nose.plugins.attrib import attr


def _get_mnist_dataset():
    return fetch_mldata('MNIST original')


def groupcounts(s):
    counts = np.zeros(s.ngroups(), dtype=np.int)
    for i, gid in enumerate(s.groups()):
        counts[i] = s.groupsize(gid)
    return np.sort(counts)[::-1]


def groupsbysize(s):
    """groupids by decreasing size"""
    counts = [(gid, s.groupsize(gid)) for gid in s.groups()]
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    return counts


@attr('slow')
#@profile
def test_mnist_supervised(n):
    mnist_dataset = _get_mnist_dataset()
    classes = range(10)
    classmap = {c: i for i, c in enumerate(classes)}
    train_data, test_data = [], []
    for c in classes:
        Y = mnist_dataset['data'][
            np.where(mnist_dataset['target'] == float(c))[0]]
        Y_train, Y_test = train_test_split(Y, test_size=0.01)
        train_data.append(Y_train)
        test_data.append(Y_test)

    sample_size_max = n

    def mk_class_data(c, Y):
        n, D = Y.shape
        print 'number of digit', c, 'in training is', n
        dtype = [('', bool)] * D + [('', int)]
        inds = np.random.permutation(Y.shape[0])[:sample_size_max]
        Y = np.array([tuple(list(y) + [classmap[c]]) for y in Y[inds]],
                     dtype=dtype)
        return Y
    Y_train = np.hstack([mk_class_data(c, y)
                        for c, y in zip(classes, train_data)])
    Y_train = Y_train[np.random.permutation(np.arange(Y_train.shape[0]))]

    n, = Y_train.shape
    D = len(Y_train.dtype)
    print 'training data is', n, 'examples'
    print 'image dimension is', (D - 1), 'pixels'

    view = numpy_dataview(Y_train)
    defn = model_definition(n, [bb] * (D - 1) + [dd(len(classes))])
    r = rng()
    s = initialize(defn,
                   view,
                   cluster_hp={'alpha': 0.2},
                   feature_hps=[{'alpha': 1., 'beta': 1.}] *
                   (D - 1) + [{'alphas': [1. for _ in classes]}],
                   r=r)

    bound_s = bind(s, view)

    indiv_prior_fn = log_exponential(1.2)
    hparams = {
        i: {
            'alpha': (indiv_prior_fn, 1.5),
            'beta': (indiv_prior_fn, 1.5),
        } for i in xrange(D - 1)}
    hparams[D - 1] = {
        'alphas[{}]'.format(idx): (indiv_prior_fn, 1.5)
        for idx in xrange(len(classes))
    }

    def print_prediction_results():
        results = []
        for c, Y_test in zip(classes, test_data):
            for y in Y_test:
                query = ma.masked_array(
                    np.array([tuple(y) + (0,)],
                             dtype=[('', bool)] * (D - 1) + [('', int)]),
                    mask=[(False,) * (D - 1) + (True,)])[0]
                samples = [
                    s.sample_post_pred(query, r)[1][0][-1] for _ in xrange(30)]
                samples = np.bincount(samples, minlength=len(classes))
                prediction = np.argmax(samples)
                results.append((classmap[c], prediction, samples))
            print 'finished predictions for class', c

        Y_actual = np.array([a for a, _, _ in results], dtype=np.int)
        Y_pred = np.array([b for _, b, _ in results], dtype=np.int)
        print 'accuracy:', accuracy_score(Y_actual, Y_pred)
        print 'confusion matrix:'
        print confusion_matrix(Y_actual, Y_pred)

        # AUROC for one vs all (each class)
        for i, clabel in enumerate(classes):
            Y_true = np.copy(Y_actual)

            # treat class c as the "positive" example
            positive_examples = Y_actual == i
            negative_examples = Y_actual != i
            Y_true[positive_examples] = 1
            Y_true[negative_examples] = 0
            Y_prob = np.array([float(c[i]) / c.sum() for _, _, c in results])
            cls_auc = roc_auc_score(Y_true, Y_prob)
            print 'class', clabel, 'auc=', cls_auc

        #import matplotlib.pylab as plt
        #Y_prob = np.array([c for _, _, c in results])
        #fpr, tpr, thresholds = roc_curve(Y_actual, Y_prob, pos_label=0)
        #plt.plot(fpr, tpr)
        #plt.show()

    def kernel(rid):
        start0 = time.time()
        assign(bound_s, r)
        sec0 = time.time() - start0

        start1 = time.time()
        hp(bound_s, r, hparams=hparams)
        sec1 = time.time() - start1

        print 'rid=', rid, 'nclusters=', s.ngroups(), \
            'iter0=', sec0, 'sec', 'iter1=', sec1, 'sec'

        sec_per_post_pred = sec0 / (float(view.size()) * (float(s.ngroups())))
        print '  time_per_post_pred=', sec_per_post_pred, 'sec'

    # training
    iters = 30
    for rid in xrange(iters):
        kernel(rid)

    # print group size breakdown
    sizes = [(gid, s.groupsize(gid)) for gid in s.groups()]
    sizes = sorted(sizes, key=lambda x: x[1], reverse=True)
    print '  group_sizes=', sizes

    #print_prediction_results()

    # save state
    mkdirp("mnist-states")
    fname = os.path.join("mnist-states", "state-iter{}.ser".format(rid))
    with open(fname, "w") as fp:
        fp.write(s.serialize())

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(
        description=globals()['__doc__'],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-n', '--nsamples', required=True, type=int,
        help='Number of samples from each digit')

    args = parser.parse_args()
    test_mnist_supervised(args.nsamples)
    end = time.time()
    print 'sampler with %d samples took %.2f seconds' % \
        (args.nsamples*10, end-start)
