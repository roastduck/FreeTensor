import functools

from .. import core
from .shape_utils import *


@core.inline
def _binary_op_(op, a, b, out):
    if core.ndim(out) == 0:
        out[()] = op(a, b)
    else:
        'nid: L_elem'
        for i in range(out.shape(0)):
            if core.ndim(a) < core.ndim(out):
                assert b.shape(0) == out.shape(0)
                'nid: recur'
                _binary_op_(op, a, b[i], out[i])
            elif core.ndim(b) < core.ndim(out):
                assert a.shape(0) == out.shape(0)
                'nid: recur'
                _binary_op_(op, a[i], b, out[i])
            else:
                assert a.shape(0) == out.shape(0) or a.shape(0) == 1
                assert b.shape(0) == out.shape(0) or b.shape(0) == 1
                'nid: recur'
                _binary_op_(op, a[i % a.shape(0)], b[i % b.shape(0)], out[i])


@core.inline(fallback=lambda op, a, b: op(a, b))
def _binary_op(op, a, b):
    'nid: out'
    out = core.create_var(broadcast_shape(a, b),
                          core.up_cast(core.dtype(a), core.dtype(b)),
                          core.same_mtype(core.mtype(a), core.mtype(b)))
    'nid: recur'
    _binary_op_(op, a, b, out)
    return out


add_ = functools.partial(_binary_op_, lambda x, y: x + y)
add = functools.partial(_binary_op, lambda x, y: x + y)

sub_ = functools.partial(_binary_op_, lambda x, y: x - y)
sub = functools.partial(_binary_op, lambda x, y: x - y)

mul_ = functools.partial(_binary_op_, lambda x, y: x * y)
mul = functools.partial(_binary_op, lambda x, y: x * y)

div_ = functools.partial(_binary_op_, lambda x, y: x / y)
div = functools.partial(_binary_op, lambda x, y: x / y)


@core.inline
def _unary_op_(op, x, y):
    if core.ndim(x) == 0:
        y[()] = op(x)
    else:
        assert x.shape(0) == y.shape(0)
        'nid: L_elem'
        for i in range(x.shape(0)):
            'nid: recur'
            _unary_op_(op, x[i], y[i])


@core.inline
def _unary_op(op, x):
    'nid: y'
    y = core.create_var(copy_shape(x), core.dtype(x), core.mtype(x))
    'nid: recur'
    _unary_op_(op, x, y)
    return y


relu_ = functools.partial(_unary_op_, lambda x: core.max(x, 0))
relu = functools.partial(_unary_op, lambda x: core.max(x, 0))

abs_ = functools.partial(_unary_op_, lambda x: core.abs(x))
abs = functools.partial(_unary_op, lambda x: core.abs(x))

sqrt_ = functools.partial(_unary_op_, lambda x: core.sqrt(x))
sqrt = functools.partial(_unary_op, lambda x: core.sqrt(x))

exp_ = functools.partial(_unary_op_, lambda x: core.exp(x))
exp = functools.partial(_unary_op, lambda x: core.exp(x))

sigmoid_ = functools.partial(_unary_op_, lambda x: core.sigmoid(x))
sigmoid = functools.partial(_unary_op, lambda x: core.sigmoid(x))

tanh_ = functools.partial(_unary_op_, lambda x: core.tanh(x))
tanh = functools.partial(_unary_op, lambda x: core.tanh(x))
