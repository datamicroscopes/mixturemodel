def test_cxx_import():
    from microscopes.mixture.model import \
        state, \
        bind, \
        initialize, \
        deserialize
    assert state
    assert bind
    assert initialize
    assert deserialize
