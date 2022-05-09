from .. import core
from .assign import assign


@core.inline
def zeros_(y):
    if core.ndim(y) == 0:
        y[()] = 0
    else:
        'nid: L_elem'
        for i in range(core.shape(y, 0)):
            'nid: recur'
            zeros_(y[i])


@core.inline
def zeros(shape, dtype, mtype=None):
    y = core.empty(shape, dtype, mtype)
    'nid: recur'
    zeros_(y)
    return y
