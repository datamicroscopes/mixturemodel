def test_import():
    from microscopes.cxx.mixture.model import state, fixed_state, bind, bind_fixed
    assert state and fixed_state and bind and bind_fixed
    from microscopes.cxx.models import bb, nich
    s = state(10, [bb, nich])
    assert s
