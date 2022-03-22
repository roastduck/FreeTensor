from .. import core
from .shape_utils import *


def _assign_op(op):

    @core.inline
    def f_binary_op(y, x):
        if core.ndim(y) == 0:
            y[()] = op(y, x)
        else:
            'nid: L_elem'
            for i in range(core.shape(y, 0)):
                if core.ndim(x) < core.ndim(y):
                    'nid: recur'
                    _assign_op(op)(y[i], x)
                else:
                    assert core.shape(x, 0) == core.shape(y, 0) or core.shape(
                        x, 0) == 1
                    'nid: recur'
                    _assign_op(op)(y[i], x[i % x.shape(0)])

    return f_binary_op


assign = _assign_op(lambda y, x: x)
add_to = _assign_op(lambda y, x: x + y)
mul_to = _assign_op(lambda y, x: x * y)
