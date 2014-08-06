def test_cxx_import():
    from microscopes.mixture.model import \
        state, fixed_state, \
        bind, bind_fixed, \
        initialize, initialize_fixed, \
        deserialize, deserialize_fixed
    assert state and fixed_state
    assert bind and bind_fixed
    assert initialize and initialize_fixed
    assert deserialize and deserialize_fixed
