from .. import core
from .utils import *
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
    out = core.empty(broadcast_shape(a, b),
                     core.up_cast(core.dtype(a), core.dtype(b)),
                     core.same_mtype(core.mtype(a), core.mtype(b)))
    'nid: recur'
    _binary_op_(op, a, b, out)
    return out


add_ = named_partial("add_", _binary_op_, lambda x, y: x + y)
add = named_partial("add", _binary_op, lambda x, y: x + y)

sub_ = named_partial("sub_", _binary_op_, lambda x, y: x - y)
sub = named_partial("sub", _binary_op, lambda x, y: x - y)

mul_ = named_partial("mul_", _binary_op_, lambda x, y: x * y)
mul = named_partial("mul", _binary_op, lambda x, y: x * y)

truediv_ = named_partial("truediv_", _binary_op_, lambda x, y: x / y)
truediv = named_partial("truediv", _binary_op, lambda x, y: x / y)


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
    y = core.empty(copy_shape(x), core.dtype(x), core.mtype(x))
    'nid: recur'
    _unary_op_(op, x, y)
    return y


relu_ = named_partial("relu_", _unary_op_, lambda x: core.max(x, 0))
relu = named_partial("relu", _unary_op, lambda x: core.max(x, 0))

abs_ = named_partial("abs_", _unary_op_, lambda x: core.abs(x))
abs = named_partial("abs", _unary_op, lambda x: core.abs(x))

sqrt_ = named_partial("sqrt_", _unary_op_, lambda x: core.sqrt(x))
sqrt = named_partial("sqrt", _unary_op, lambda x: core.sqrt(x))

exp_ = named_partial("exp_", _unary_op_, lambda x: core.exp(x))
exp = named_partial("exp", _unary_op, lambda x: core.exp(x))

sigmoid_ = named_partial("sigmoid_", _unary_op_, lambda x: core.sigmoid(x))
sigmoid = named_partial("sigmoid", _unary_op, lambda x: core.sigmoid(x))

tanh_ = named_partial("tanh_", _unary_op_, lambda x: core.tanh(x))
tanh = named_partial("tanh", _unary_op, lambda x: core.tanh(x))
