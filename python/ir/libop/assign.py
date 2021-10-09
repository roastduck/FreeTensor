from .. import core
from .shape_utils import *


def _assign_op(op):

    @core.inline
    def f_binary_op(y, x):
        if y.ndim == 0:
            y[()] = op(y[()], x[()])
        else:
            'nid: L_elem'
            for i in range(y.shape(0)):
                if x.ndim < y.ndim:
                    'nid: recur'
                    _assign_op(op)(y[i], x)
                else:
                    'nid: recur'
                    _assign_op(op)(y[i], x[i % x.shape(0)])

    return f_binary_op


assign = _assign_op(lambda y, x: x)
add_to = _assign_op(lambda y, x: x + y)
