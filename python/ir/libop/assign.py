from .. import core
from .shape_utils import *


def _assign_op(io_mem, op):

    @core.inline
    def f_binary_op(y, x):
        if y.ndim == 0:
            y[()] = op(y[()], x[()])
        else:
            'nid: L_elem'
            for i in range(y.shape(0)):
                if x.ndim < y.ndim:
                    'nid: recur'
                    _assign_op(io_mem, op)(y[i], x)
                else:
                    'nid: recur'
                    _assign_op(io_mem, op)(y[i], x[i % x.shape(0)])

    return f_binary_op


def assign(io_mem):
    return _assign_op(io_mem, lambda y, x: x)


def add_to(io_mem):
    return _assign_op(io_mem, lambda y, x: x + y)
