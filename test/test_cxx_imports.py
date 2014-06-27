def test_import_dp():
    from microscopes.cxx.mixture.model import state
    assert state
    from microscopes.cxx.models import bb, nich
    s = state(10, [bb, nich])
    assert s
