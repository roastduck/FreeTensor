from .. import core
from .shape_utils import *


def _binary_op_(op):

    @core.inline
    def f_binary_op(a, b, out):
        if core.ndim(out) == 0:
            out[()] = op(a, b)
        else:
            'nid: L_elem'
            for i in range(out.shape(0)):
                if core.ndim(a) < core.ndim(out):
                    assert b.shape(0) == out.shape(0)
                    'nid: recur'
                    _binary_op_(op)(a, b[i], out[i])
                elif core.ndim(b) < core.ndim(out):
                    assert a.shape(0) == out.shape(0)
                    'nid: recur'
                    _binary_op_(op)(a[i], b, out[i])
                else:
                    assert a.shape(0) == out.shape(0) or a.shape(0) == 1
                    assert b.shape(0) == out.shape(0) or b.shape(0) == 1
                    'nid: recur'
                    _binary_op_(op)(a[i % a.shape(0)], b[i % b.shape(0)],
                                    out[i])

    return f_binary_op


def _binary_op(op):

    @core.inline
    def f_binary_op(a, b):
        out = core.create_var(broadcast_shape(a, b),
                              core.up_cast(core.dtype(a), core.dtype(b)),
                              core.same_mtype(core.mtype(a), core.mtype(b)))
        'nid: recur'
        _binary_op_(op)(a, b, out)
        return out

    return f_binary_op


add_ = _binary_op_(lambda x, y: x + y)
add = _binary_op(lambda x, y: x + y)
# add.set_fallback(lambda x, y: x + y)

sub_ = _binary_op_(lambda x, y: x - y)
sub = _binary_op(lambda x, y: x - y)
# sub.set_fallback(lambda x, y: x - y)

mul_ = _binary_op_(lambda x, y: x * y)
mul = _binary_op(lambda x, y: x * y)
# mul.set_fallback(lambda x, y: x * y)

div_ = _binary_op_(lambda x, y: x / y)
div = _binary_op(lambda x, y: x / y)
# div.set_fallback(lambda x, y: x / y)


def _unary_op_(op):

    @core.inline
    def f_unary_op(x, y):
        if core.ndim(x) == 0:
            y[()] = op(x)
        else:
            assert x.shape(0) == y.shape(0)
            'nid: L_elem'
            for i in range(x.shape(0)):
                'nid: recur'
                _unary_op_(op)(x[i], y[i])

    return f_unary_op


def _unary_op(op):

    @core.inline
    def f_unary_op(x):
        y = core.create_var(copy_shape(x), core.dtype(x), core.mtype(x))
        'nid: recur'
        _unary_op_(op)(x, y)
        return y

    return f_unary_op


relu_ = _unary_op_(lambda x: core.max(x, 0))
relu = _unary_op(lambda x: core.max(x, 0))

abs_ = _unary_op_(lambda x: core.abs(x))
abs = _unary_op(lambda x: core.abs(x))

sqrt_ = _unary_op_(lambda x: core.sqrt(x))
sqrt = _unary_op(lambda x: core.sqrt(x))

exp_ = _unary_op_(lambda x: core.exp(x))
exp = _unary_op(lambda x: core.exp(x))

sigmoid_ = _unary_op_(lambda x: core.sigmoid(x))
sigmoid = _unary_op(lambda x: core.sigmoid(x))

tanh_ = _unary_op_(lambda x: core.tanh(x))
tanh = _unary_op(lambda x: core.tanh(x))
