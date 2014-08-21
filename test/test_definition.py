from microscopes.mixture.definition import (
    model_definition,
    fixed_model_definition,
)
from microscopes.models import bb, niw

from nose.tools import (
    assert_equals,
    assert_is_not,
)
import pickle
import copy


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


def test_model_definition_copy():
    defn = model_definition(10, [bb, niw(3)])
    defn_shallow = copy.copy(defn)
    defn_deep = copy.deepcopy(defn)
    assert_is_not(defn, defn_shallow)
    assert_is_not(defn, defn_deep)
    assert_is_not(defn._models, defn_deep._models)
    assert_equals(defn.n(), defn_shallow.n())
    assert_equals(defn.n(), defn_deep.n())

    defn = fixed_model_definition(10, 3, [bb, niw(2)])
    defn_shallow = copy.copy(defn)
    defn_deep = copy.deepcopy(defn)
    assert_is_not(defn, defn_shallow)
    assert_is_not(defn, defn_deep)
    assert_is_not(defn._models, defn_deep._models)
    assert_equals(defn.n(), defn_shallow.n())
    assert_equals(defn.n(), defn_deep.n())
    assert_equals(defn.groups(), defn_shallow.groups())
    assert_equals(defn.groups(), defn_deep.groups())
