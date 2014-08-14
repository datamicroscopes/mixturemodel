from microscopes.mixture.definition import (
    model_definition,
    fixed_model_definition,
)
from microscopes.models import bb, niw

from nose.tools import assert_equals
import pickle


def test_model_definition_pickle():
    defn = model_definition(10, [bb, niw(3)])
    bstr = pickle.dumps(defn)
    defn1 = pickle.loads(bstr)
    assert_equals(defn.n(), defn1.n())
    assert_equals(len(defn.models()), len(defn1.models()))
    for a, b in zip(defn.models(), defn1.models()):
        assert_equals(a.name(), b.name())

    defn = fixed_model_definition(10, 3, [bb, niw(2)])
    bstr = pickle.dumps(defn)
    defn1 = pickle.loads(bstr)
    assert_equals(defn.n(), defn1.n())
    assert_equals(defn.groups(), defn1.groups())
    assert_equals(len(defn.models()), len(defn1.models()))
    for a, b in zip(defn.models(), defn1.models()):
        assert_equals(a.name(), b.name())
