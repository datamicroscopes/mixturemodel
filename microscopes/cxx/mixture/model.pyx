from microscopes.cxx.mixture._model cimport \
    state as _state, \
    fixed_state as _fixed_state
from microscopes.cxx.mixture._model import bind, bind_fixed
class fixed_state(_fixed_state):
    pass
class state(_state):
    pass
