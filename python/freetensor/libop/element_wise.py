from .. import core
from .utils import *
from .shape_utils import *


def _named_partial(name: str, doc: str, f, *args, **kvs):
    ''' Similar to functools.partial, but it sets the returned function's __name__ and __doc__ '''

    # This function should be defined in the same file that uses it
    # https://github.com/mkdocstrings/pytkdocs/issues/143

    def g(*_args, **_kvs):
        return f(*args, *_args, **kvs, **_kvs)

    g.__name__ = name
    g.__doc__ = doc
    return g


@core.inline
def _binary_op_(op, a, b, out):
    if core.ndim(out) == 0:
        out[()] = op(a, b)
    else:
        #! nid: L_elem
        for i in range(out.shape(0)):
            if core.ndim(a) < core.ndim(out):
                assert b.shape(0) == out.shape(0)
                #! nid: recur
                _binary_op_(op, a, b[i], out[i])
            elif core.ndim(b) < core.ndim(out):
                assert a.shape(0) == out.shape(0)
                #! nid: recur
                _binary_op_(op, a[i], b, out[i])
            else:
                assert a.shape(0) == out.shape(0) or a.shape(0) == 1
                assert b.shape(0) == out.shape(0) or b.shape(0) == 1
                #! nid: recur
                _binary_op_(op, a[i % a.shape(0)], b[i % b.shape(0)], out[i])


@core.inline(fallback=lambda op, a, b: op(a, b))
def _binary_op(op, a, b):
    #! nid: out
    out = core.empty(broadcast_shape(a, b),
                     core.up_cast(core.dtype(a), core.dtype(b)),
                     core.same_mtype(core.mtype(a), core.mtype(b)))
    #! nid: recur
    _binary_op_(op, a, b, out)
    return out


implace_binary_doc_template = '''
(Broadcasted) element-wise {} of two tensors. The result is written to another tensor

Parameters
----------
a : VarRef
    Left-hand-side operand
b : VarRef
    Right-hand-side operand
out : VarRef
    The result tensor
'''

out_of_place_binary_doc_template = '''
(Broadcasted) element-wise {} of two tensors and return the result

Parameters
----------
a : VarRef
    Left-hand-side operand
b : VarRef
    Right-hand-side operand

Returns
-------
VarRef
    The result tensor
'''

add_ = _named_partial("add_", implace_binary_doc_template.format('addition'),
                      _binary_op_, lambda x, y: x + y)
add = _named_partial("add", out_of_place_binary_doc_template.format('addition'),
                     _binary_op, lambda x, y: x + y)

sub_ = _named_partial("sub_", implace_binary_doc_template.format('subtraction'),
                      _binary_op_, lambda x, y: x - y)
sub = _named_partial("sub",
                     out_of_place_binary_doc_template.format('subtraction'),
                     _binary_op, lambda x, y: x - y)

mul_ = _named_partial("mul_",
                      implace_binary_doc_template.format('multiplication'),
                      _binary_op_, lambda x, y: x * y)
mul = _named_partial("mul",
                     out_of_place_binary_doc_template.format('multiplication'),
                     _binary_op, lambda x, y: x * y)

truediv_ = _named_partial(
    "truediv_", implace_binary_doc_template.format('floating-point division'),
    _binary_op_, lambda x, y: x / y)
truediv = _named_partial(
    "truediv",
    out_of_place_binary_doc_template.format('floating-point division'),
    _binary_op, lambda x, y: x / y)


def _floordivMayFallback(lhs, rhs):
    if (hasattr(lhs, '__module__') and
            lhs.__module__ == 'torch') or (hasattr(rhs, '__module__') and
                                           rhs.__module__ == 'torch'):
        import torch
        return torch.div(lhs, rhs, rounding_mode='floor')
    return lhs // rhs


floordiv_ = _named_partial(
    "floordiv_",
    implace_binary_doc_template.format(
        'rounding-towards-negative-infinity integer division (following Python convention, but not C)'
    ), _binary_op_, _floordivMayFallback)
floordiv = _named_partial(
    "floordiv",
    out_of_place_binary_doc_template.format(
        'rounding-towards-negative-infinity integer division (following Python convention, but not C)'
    ), _binary_op, _floordivMayFallback)

mod_ = _named_partial(
    "mod_",
    implace_binary_doc_template.format(
        'modulo (results are non-negative, following Python convention, but not C)'
    ), _binary_op_, lambda x, y: x % y)
mod = _named_partial(
    "mod",
    out_of_place_binary_doc_template.format(
        'modulo (results are non-negative, following Python convention, but not C)'
    ), _binary_op, lambda x, y: x % y)

l_and_ = _named_partial("l_and_",
                        implace_binary_doc_template.format("logical and"),
                        _binary_op_, core.l_and)
l_and = _named_partial("l_and",
                       out_of_place_binary_doc_template.format("logical and"),
                       _binary_op, core.l_and)

l_or_ = _named_partial("l_or_",
                       implace_binary_doc_template.format("logical or"),
                       _binary_op_, core.l_or)
l_or = _named_partial("l_or",
                      out_of_place_binary_doc_template.format("logical or"),
                      _binary_op, core.l_or)

lt_ = _named_partial("lt_", implace_binary_doc_template.format("less-than"),
                     _binary_op_, lambda x, y: x < y)
lt = _named_partial("lt", out_of_place_binary_doc_template.format("less-than"),
                    _binary_op, lambda x, y: x < y)

le_ = _named_partial(
    "le_", implace_binary_doc_template.format("less-than-or-equal-to"),
    _binary_op_, lambda x, y: x <= y)
le = _named_partial(
    "le", out_of_place_binary_doc_template.format("less-than-or-equal-to"),
    _binary_op, lambda x, y: x <= y)

gt_ = _named_partial("gt_", implace_binary_doc_template.format("greater-than"),
                     _binary_op_, lambda x, y: x > y)
gt = _named_partial("gt",
                    out_of_place_binary_doc_template.format("greater-than"),
                    _binary_op, lambda x, y: x > y)

ge_ = _named_partial(
    "ge_", implace_binary_doc_template.format("greater-than-or-equal-to"),
    _binary_op_, lambda x, y: x >= y)
ge = _named_partial(
    "ge", out_of_place_binary_doc_template.format("greater-than-or-equal-to"),
    _binary_op, lambda x, y: x >= y)

eq_ = _named_partial("eq_", implace_binary_doc_template.format("equal"),
                     _binary_op_, lambda x, y: x == y)
eq = _named_partial("eq", out_of_place_binary_doc_template.format("equal"),
                    _binary_op, lambda x, y: x == y)

ne_ = _named_partial("ne_", implace_binary_doc_template.format("non-equal"),
                     _binary_op_, lambda x, y: x != y)
ne = _named_partial("ne", out_of_place_binary_doc_template.format("non-equal"),
                    _binary_op, lambda x, y: x != y)


@core.inline
def _unary_op_(op, x, y):
    if core.ndim(x) == 0:
        y[()] = op(x)
    else:
        assert x.shape(0) == y.shape(0)
        #! nid: L_elem
        for i in range(x.shape(0)):
            #! nid: recur
            _unary_op_(op, x[i], y[i])


@core.inline
def _unary_op(op, x):
    #! nid: y
    y = core.empty(copy_shape(x), core.dtype(x), core.mtype(x))
    #! nid: recur
    _unary_op_(op, x, y)
    return y


implace_unary_doc_template = '''
Element-wise {} of a tensor. The result is written to another tensor

Parameters
----------
x : VarRef
    The input tensor
out : VarRef
    The result tensor
'''

out_of_place_unary_doc_template = '''
Element-wise {} of a tensor and return the result

Parameters
----------
x : VarRef
    The input tensor

Returns
-------
VarRef
    The result tensor
'''

neg_ = _named_partial("neg_", implace_unary_doc_template.format("negation"),
                      _unary_op_, lambda x: -x)
neg = _named_partial("neg", out_of_place_unary_doc_template.format("negation"),
                     _unary_op, lambda x: -x)

l_not_ = _named_partial("l_not_",
                        implace_unary_doc_template.format("logical not"),
                        _unary_op_, core.l_not)
l_not = _named_partial("l_not",
                       out_of_place_unary_doc_template.format("logical not"),
                       _unary_op, core.l_not)

relu_ = _named_partial("relu_", implace_unary_doc_template.format("ReLU"),
                       _unary_op_, lambda x: core.max(x, 0))
relu = _named_partial("relu", out_of_place_unary_doc_template.format("ReLU"),
                      _unary_op, lambda x: core.max(x, 0))

abs_ = _named_partial("abs_",
                      implace_unary_doc_template.format("absolute value"),
                      _unary_op_, lambda x: core.abs(x))
abs = _named_partial("abs",
                     out_of_place_unary_doc_template.format("absolute value"),
                     _unary_op, lambda x: core.abs(x))

sqrt_ = _named_partial("sqrt_",
                       implace_unary_doc_template.format("square root"),
                       _unary_op_, lambda x: core.sqrt(x))
sqrt = _named_partial("sqrt",
                      out_of_place_unary_doc_template.format("square root"),
                      _unary_op, lambda x: core.sqrt(x))

exp_ = _named_partial("exp_",
                      implace_unary_doc_template.format("natrual exponent"),
                      _unary_op_, lambda x: core.exp(x))
exp = _named_partial("exp",
                     out_of_place_unary_doc_template.format("natrual exponent"),
                     _unary_op, lambda x: core.exp(x))

sigmoid_ = _named_partial("sigmoid_",
                          implace_unary_doc_template.format("sigmoid"),
                          _unary_op_, lambda x: core.sigmoid(x))
sigmoid = _named_partial("sigmoid",
                         out_of_place_unary_doc_template.format("sigmoid"),
                         _unary_op, lambda x: core.sigmoid(x))

tanh_ = _named_partial("tanh_", implace_unary_doc_template.format("tanh"),
                       _unary_op_, lambda x: core.tanh(x))
tanh = _named_partial("tanh", out_of_place_unary_doc_template.format("tanh"),
                      _unary_op, lambda x: core.tanh(x))
