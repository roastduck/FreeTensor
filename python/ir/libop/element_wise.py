from .. import core
from .shape_utils import *


def _binary_op_(io_mem, op):

    @core.inline
    def f_binary_op(a, b, out):
        if out.ndim == 0:
            out[()] = op(a[()], b[()])
        else:
            'nid: L_elem'
            for i in range(out.shape(0)):
                if a.ndim < out.ndim:
                    'nid: recur'
                    _binary_op_(io_mem, op)(a, b[i], out[i])
                elif b.ndim < out.ndim:
                    'nid: recur'
                    _binary_op_(io_mem, op)(a[i], b, out[i])
                else:
                    'nid: recur'
                    _binary_op_(io_mem, op)(a[i % a.shape(0)],
                                            b[i % b.shape(0)], out[i])

    return f_binary_op


def _binary_op(io_mem, op):

    @core.inline
    def f_binary_op(a, b):
        'nid: broadcast_shape'
        out = core.create_var(broadcast_shape(a, b),
                              core.up_cast(a.dtype, b.dtype), "output", io_mem)
        'nid: recur'
        _binary_op_(io_mem, op)(a, b, out)
        return out

    return f_binary_op


def add_(io_mem):
    return _binary_op_(io_mem, lambda x, y: x + y)


def add(io_mem):
    return _binary_op(io_mem, lambda x, y: x + y)


def sub_(io_mem):
    return _binary_op_(io_mem, lambda x, y: x - y)


def sub(io_mem):
    return _binary_op(io_mem, lambda x, y: x - y)


def mul_(io_mem):
    return _binary_op_(io_mem, lambda x, y: x * y)


def mul(io_mem):
    return _binary_op(io_mem, lambda x, y: x * y)


def div_(io_mem):
    return _binary_op_(io_mem, lambda x, y: x / y)


def div(io_mem):
    return _binary_op(io_mem, lambda x, y: x / y)


def _unary_op_(io_mem, op):

    @core.inline
    def f_unary_op(x, y):
        if x.ndim == 0:
            y[()] = op(x[()])
        else:
            'nid: L_elem'
            for i in range(x.shape(0)):
                'nid: recur'
                _unary_op_(io_mem, op)(x[i], y[i])

    return f_unary_op


def _unary_op(io_mem, op):

    @core.inline
    def f_unary_op(x):
        y = core.create_var(copy_shape(x), x.dtype, "output", io_mem)
        'nid: recur'
        _unary_op_(io_mem, op)(x, y)
        return y

    return f_unary_op


def relu_(io_mem):
    return _unary_op_(io_mem, lambda x: core.max(x, 0))


def relu(io_mem):
    return _unary_op(io_mem, lambda x: core.max(x, 0))


def abs_(io_mem):
    return _unary_op_(io_mem, lambda x: core.abs(x))


def abs(io_mem):
    return _unary_op(io_mem, lambda x: core.abs(x))


def sqrt_(io_mem):
    return _unary_op_(io_mem, lambda x: core.sqrt(x))


def sqrt(io_mem):
    return _unary_op(io_mem, lambda x: core.sqrt(x))


def exp_(io_mem):
    return _unary_op_(io_mem, lambda x: core.exp(x))


def exp(io_mem):
    return _unary_op(io_mem, lambda x: core.exp(x))
