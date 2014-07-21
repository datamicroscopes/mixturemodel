def test_cxx_import():
    from microscopes.cxx.mixture.model import \
        state, fixed_state, \
        bind, bind_fixed, \
        initialize, initialize_fixed, \
        deserialize, deserialize_fixed
    assert state and fixed_state
    assert bind and bind_fixed
    assert initialize and initialize_fixed
    assert deserialize and deserialize_fixed

def test_py_import():
    from microscopes.py.mixture.model import \
        state, \
        bind, \
        initialize, \
        deserialize
    assert state
    assert bind
    assert initialize
    assert deserialize
