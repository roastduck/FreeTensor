import operator

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
def binary_op_(op, a, b, out):
    '''
    (Broadcasted) any element-wise operation on two tensors. The result is written to another tensor

    Parameters
    ----------
    op : Callable
        The operation applied to each item
    a : VarRef
        Left-hand-side operand
    b : VarRef
        Right-hand-side operand
    out : VarRef
        The result tensor
    '''

    if core.ndim(out) == 0:
        out[()] = op(a, b)
    else:
        #! label: L_elem
        for i in range(out.shape(0)):
            if core.ndim(a) < core.ndim(out):
                assert b.shape(0) == out.shape(0)
                #! label: recur
                binary_op_(op, a, b[i], out[i])
            elif core.ndim(b) < core.ndim(out):
                assert a.shape(0) == out.shape(0)
                #! label: recur
                binary_op_(op, a[i], b, out[i])
            else:
                assert a.shape(0) == out.shape(0) or a.shape(0) == 1
                assert b.shape(0) == out.shape(0) or b.shape(0) == 1
                #! label: recur
                binary_op_(op, a[i % a.shape(0)], b[i % b.shape(0)], out[i])


@core.inline
def binary_op(op, a, b, out_dtype=None):
    '''
    (Broadcasted) any element-wise operation on two tensors and return the result

    Parameters
    ----------
    op : Callable
        The operation applied to each item
    a : VarRef
        Left-hand-side operand
    b : VarRef
        Right-hand-side operand

    Returns
    -------
    VarRef
        The result tensor
    '''

    #! label: out
    out = core.empty(
        broadcast_shape(a, b),
        out_dtype if out_dtype is not None else core.up_cast(
            core.dtype(a), core.dtype(b)),
        core.same_mtype(core.mtype(a), core.mtype(b)))
    #! label: recur
    binary_op_(op, a, b, out)
    return out


inplace_binary_doc_template = '''
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

add_ = _named_partial("add_", inplace_binary_doc_template.format('addition'),
                      binary_op_, operator.add)
add = _named_partial("add", out_of_place_binary_doc_template.format('addition'),
                     binary_op, operator.add)

sub_ = _named_partial("sub_", inplace_binary_doc_template.format('subtraction'),
                      binary_op_, operator.sub)
sub = _named_partial("sub",
                     out_of_place_binary_doc_template.format('subtraction'),
                     binary_op, operator.sub)

mul_ = _named_partial("mul_",
                      inplace_binary_doc_template.format('multiplication'),
                      binary_op_, operator.mul)
mul = _named_partial("mul",
                     out_of_place_binary_doc_template.format('multiplication'),
                     binary_op, operator.mul)

truediv_ = _named_partial(
    "truediv_", inplace_binary_doc_template.format('floating-point division'),
    binary_op_, operator.truediv)
truediv = _named_partial(
    "truediv",
    out_of_place_binary_doc_template.format('floating-point division'),
    binary_op, operator.truediv)

floordiv_ = _named_partial(
    "floordiv_",
    inplace_binary_doc_template.format(
        'rounding-towards-negative-infinity integer division (following Python convention, but not C, recommended for performance)'
    ), binary_op_, operator.floordiv)
floordiv = _named_partial(
    "floordiv",
    out_of_place_binary_doc_template.format(
        'rounding-towards-negative-infinity integer division (following Python convention, but not C, recommended for performance)'
    ), binary_op, operator.floordiv)

ceildiv_ = _named_partial(
    "ceildiv_",
    inplace_binary_doc_template.format(
        'rounding-towards-positive-infinity integer division'), binary_op_,
    core.ceildiv)
ceildiv = _named_partial(
    "ceildiv",
    out_of_place_binary_doc_template.format(
        'rounding-towards-positive-infinity integer division'), binary_op,
    core.ceildiv)

round_towards_0_div_ = _named_partial(
    "round_towards_0_div_",
    inplace_binary_doc_template.format(
        'rounding-towards-0 integer division (following C convention, but not Python, NOT recommended for performance)'
    ), binary_op_, core.round_towards_0_div)
round_towards_0_div = _named_partial(
    "round_towards_0_div",
    out_of_place_binary_doc_template.format(
        'rounding-towards-0 integer division (following C convention, but not Python, NOT recommended for performance)'
    ), binary_op, core.round_towards_0_div)

mod_ = _named_partial(
    "mod_",
    inplace_binary_doc_template.format(
        'modulo (results are non-negative, following Python convention, but not C, recommended for performance)'
    ), binary_op_, operator.mod)
mod = _named_partial(
    "mod",
    out_of_place_binary_doc_template.format(
        'modulo (results are non-negative, following Python convention, but not C, recommended for performance)'
    ), binary_op, operator.mod)

remainder_ = _named_partial(
    "remainder_",
    inplace_binary_doc_template.format(
        'remainder (results can be positive or negative, following C convention, but not Python, NOT recommended for performance)'
    ), binary_op_, core.remainder)
remainder = _named_partial(
    "remainder",
    out_of_place_binary_doc_template.format(
        'remainder (results can be positive or negative, following C convention, but not Python, NOT recommended for performance)'
    ), binary_op, core.remainder)

min_ = _named_partial("min_", inplace_binary_doc_template.format('minimum'),
                      binary_op_, core.min)
min = _named_partial("min", out_of_place_binary_doc_template.format('minimum'),
                     binary_op, core.min)

max_ = _named_partial("max_", inplace_binary_doc_template.format('maximum'),
                      binary_op_, core.max)
max = _named_partial("max", out_of_place_binary_doc_template.format('maximum'),
                     binary_op, core.max)

l_and_ = _named_partial("l_and_",
                        inplace_binary_doc_template.format("logical and"),
                        binary_op_, core.l_and)
l_and = _named_partial("l_and",
                       out_of_place_binary_doc_template.format("logical and"),
                       binary_op, core.l_and)

l_or_ = _named_partial("l_or_",
                       inplace_binary_doc_template.format("logical or"),
                       binary_op_, core.l_or)
l_or = _named_partial("l_or",
                      out_of_place_binary_doc_template.format("logical or"),
                      binary_op, core.l_or)

lt_ = _named_partial("lt_", inplace_binary_doc_template.format("less-than"),
                     binary_op_, operator.lt)
lt = _named_partial("lt",
                    out_of_place_binary_doc_template.format("less-than"),
                    binary_op,
                    operator.lt,
                    out_dtype="bool")

le_ = _named_partial(
    "le_", inplace_binary_doc_template.format("less-than-or-equal-to"),
    binary_op_, operator.le)
le = _named_partial(
    "le",
    out_of_place_binary_doc_template.format("less-than-or-equal-to"),
    binary_op,
    operator.le,
    out_dtype="bool")

gt_ = _named_partial("gt_", inplace_binary_doc_template.format("greater-than"),
                     binary_op_, operator.gt)
gt = _named_partial("gt",
                    out_of_place_binary_doc_template.format("greater-than"),
                    binary_op,
                    operator.gt,
                    out_dtype="bool")

ge_ = _named_partial(
    "ge_", inplace_binary_doc_template.format("greater-than-or-equal-to"),
    binary_op_, operator.ge)
ge = _named_partial(
    "ge",
    out_of_place_binary_doc_template.format("greater-than-or-equal-to"),
    binary_op,
    operator.ge,
    out_dtype="bool")

eq_ = _named_partial("eq_", inplace_binary_doc_template.format("equal"),
                     binary_op_, operator.eq)
eq = _named_partial("eq",
                    out_of_place_binary_doc_template.format("equal"),
                    binary_op,
                    operator.eq,
                    out_dtype="bool")

ne_ = _named_partial("ne_", inplace_binary_doc_template.format("non-equal"),
                     binary_op_, operator.ne)
ne = _named_partial("ne",
                    out_of_place_binary_doc_template.format("non-equal"),
                    binary_op,
                    operator.ne,
                    out_dtype="bool")


@core.inline
def unary_op_(op, x, y):
    '''
    Any element-wise operation on a tensor. The result is written to another tensor

    Parameters
    ----------
    op : Callable
        The operation applied to each item
    x : VarRef
        The input tensor
    out : VarRef
        The result tensor
    '''

    if core.ndim(x) == 0:
        y[()] = op(x)
    else:
        assert x.shape(0) == y.shape(0)
        #! label: L_elem
        for i in range(x.shape(0)):
            #! label: recur
            unary_op_(op, x[i], y[i])


@core.inline
def unary_op(op, x):
    '''
    Any element-wise operation on a tensor and return the result

    Parameters
    ----------
    op : Callable
        The operation applied to each item
    x : VarRef
        The input tensor

    Returns
    -------
    VarRef
        The result tensor
    '''

    #! label: y
    y = core.empty(copy_shape(x), core.dtype(x), core.mtype(x))
    #! label: recur
    unary_op_(op, x, y)
    return y


inplace_unary_doc_template = '''
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

neg_ = _named_partial("neg_", inplace_unary_doc_template.format("negation"),
                      unary_op_, operator.neg)
neg = _named_partial("neg", out_of_place_unary_doc_template.format("negation"),
                     unary_op, operator.neg)

l_not_ = _named_partial("l_not_",
                        inplace_unary_doc_template.format("logical not"),
                        unary_op_, core.l_not)
l_not = _named_partial("l_not",
                       out_of_place_unary_doc_template.format("logical not"),
                       unary_op, core.l_not)

relu_ = _named_partial("relu_", inplace_unary_doc_template.format("ReLU"),
                       unary_op_, lambda x: core.max(x, 0))
relu = _named_partial("relu", out_of_place_unary_doc_template.format("ReLU"),
                      unary_op, lambda x: core.max(x, 0))

abs_ = _named_partial("abs_",
                      inplace_unary_doc_template.format("absolute value"),
                      unary_op_, lambda x: core.abs(x))
abs = _named_partial("abs",
                     out_of_place_unary_doc_template.format("absolute value"),
                     unary_op, lambda x: core.abs(x))

sqrt_ = _named_partial("sqrt_",
                       inplace_unary_doc_template.format("square root"),
                       unary_op_, lambda x: core.sqrt(x))
sqrt = _named_partial("sqrt",
                      out_of_place_unary_doc_template.format("square root"),
                      unary_op, lambda x: core.sqrt(x))

square_ = _named_partial("square_", inplace_unary_doc_template.format("square"),
                         unary_op_, lambda x: core.square(x))
square = _named_partial("square",
                        out_of_place_unary_doc_template.format("square"),
                        unary_op, lambda x: core.square(x))

exp_ = _named_partial("exp_",
                      inplace_unary_doc_template.format("natrual exponent"),
                      unary_op_, lambda x: core.exp(x))
exp = _named_partial("exp",
                     out_of_place_unary_doc_template.format("natrual exponent"),
                     unary_op, lambda x: core.exp(x))

ln_ = _named_partial("ln_",
                     inplace_unary_doc_template.format("natrual logarithm"),
                     unary_op_, lambda x: core.ln(x))
ln = _named_partial("ln",
                    out_of_place_unary_doc_template.format("natrual logarithm"),
                    unary_op, lambda x: core.ln(x))

sigmoid_ = _named_partial("sigmoid_",
                          inplace_unary_doc_template.format("sigmoid"),
                          unary_op_, lambda x: core.sigmoid(x))
sigmoid = _named_partial("sigmoid",
                         out_of_place_unary_doc_template.format("sigmoid"),
                         unary_op, lambda x: core.sigmoid(x))

tanh_ = _named_partial("tanh_", inplace_unary_doc_template.format("tanh"),
                       unary_op_, lambda x: core.tanh(x))
tanh = _named_partial("tanh", out_of_place_unary_doc_template.format("tanh"),
                      unary_op, lambda x: core.tanh(x))

floor_ = _named_partial("floor_", inplace_unary_doc_template.format("floor"),
                        unary_op_, lambda x: core.floor(x))
floor = _named_partial("floor", out_of_place_unary_doc_template.format("floor"),
                       unary_op, lambda x: core.floor(x))

ceil_ = _named_partial("ceil_", inplace_unary_doc_template.format("ceil"),
                       unary_op_, lambda x: core.ceil(x))
ceil = _named_partial("ceil", out_of_place_unary_doc_template.format("ceil"),
                      unary_op, lambda x: core.ceil(x))
