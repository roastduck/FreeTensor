__all__ = [
    'reshape', 'reshape_', 'flatten', 'flatten_', 'flatten_onnx',
    'flatten_onnx_', 'unsqueeze', 'unsqueeze_', 'expand', 'expand_'
]

from typing import Sequence
from numbers import Number

from .. import core
from .utils import begin_with_0, all_minus_one
from .shape_utils import copy_shape


class _ExprHolder:
    '''
    Helper class used for hashing and comparing expressions
    '''

    def __init__(self, expr):
        self.expr = expr

    def __hash__(self):
        return self.expr.hash()

    def __eq__(lhs, rhs):
        return lhs.expr.same_as(rhs.expr)


def _factor_pairs_mul(pair_l, pair_r):
    int_factor_l, var_factors_l = pair_l
    int_factor_r, var_factors_r = pair_r
    int_factor = int_factor_l * int_factor_r
    var_factors = dict(var_factors_l)
    for var, exponent in var_factors_r.items():
        if var not in var_factors:
            var_factors[var] = exponent
        else:
            var_factors[var] += exponent
    return int_factor, var_factors


def _factor_pairs_divisible(pair_l, pair_r):
    int_factor_l, var_factors_l = pair_l
    int_factor_r, var_factors_r = pair_r
    if int_factor_l % int_factor_r != 0:
        return False
    for var, exponent in var_factors_r.items():
        if var not in var_factors_l or exponent > var_factors_l[var]:
            return False
    return True


def _factor_pairs_div(pair_l, pair_r):
    int_factor_l, var_factors_l = pair_l
    int_factor_r, var_factors_r = pair_r
    int_factor = int_factor_l // int_factor_r
    var_factors = dict(var_factors_l)
    for var, exponent in var_factors_r.items():
        var_factors[var] -= exponent
        if var_factors[var] == 0:
            del var_factors[var]
    return int_factor, var_factors


def _factorize(expr):
    '''
    Factor an expression into a `k * n1 * n2 * ...` form, where `k` is an integer,
    and `n1, n2, ...` are random expressions

    Returns
    -------
    (int, map of expressions to its exponent)
        A pair of an integer factor and zero or more variable factors
    '''
    if isinstance(expr, Number):
        return (expr, {})
    elif isinstance(expr, core.VarRef):
        return (1, {_ExprHolder(expr.as_load()): 1})
    elif isinstance(expr, core.ffi.Expr):
        if isinstance(expr, core.ffi.Mul):
            return _factor_pairs_mul(_factorize(expr.lhs), _factorize(expr.rhs))
        if isinstance(expr, core.ffi.FloorDiv) or isinstance(
                expr, core.ffi.CeilDiv) or isinstance(
                    expr, core.ffi.RoundTowards0Div):
            factor_l = _factorize(expr.lhs)
            factor_r = _factorize(expr.rhs)
            if _factor_pairs_divisible(factor_l, factor_r):
                return _factor_pairs_div(factor_l, factor_r)
        return (1, {_ExprHolder(expr): 1})
    else:
        assert False


@core.inline
def reshape_(x, y):
    '''
    Fill a tensor into another tensor with the same size but maybe different shape

    This operator will try to generate nested loops instead of looping over all
    elements in a plain loop, so schedules can be better applied. It guarantees to
    generates loops in the following cases:

    1. Splitting a dimension. E.g. 4 to 2x2, and there will be a 2x2 loop nest.
    2. Merging dimensions. E.g. 2x2 to 4, and there will be a 2x2 loop nest.
    3. Each non-affecting dimension will be iterated by a unique loop. E.g. 3x5x7
       to 5x3x7, and there will be a 15x7 loop nest, where the "7" dimension will
       be iterated by a unique loop.

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    '''

    if core.ndim(x) == 0 and core.ndim(y) == 0:
        y[...] = x[...]
    elif core.ndim(x) > 0 and core.ndim(y) == 0:
        assert x.shape(0) == 1
        reshape_(x[0], y)
    elif core.ndim(y) > 0 and core.ndim(x) == 0:
        assert y.shape(0) == 1
        reshape_(x, y[0])
    else:
        factor_x0 = _factorize(x.shape(0))
        factor_y0 = _factorize(y.shape(0))
        x0_divisible_y0 = _factor_pairs_divisible(factor_x0, factor_y0)
        y0_divisible_x0 = _factor_pairs_divisible(factor_y0, factor_x0)
        if x0_divisible_y0 and y0_divisible_x0:
            # Identical dimension
            assert x.shape(0) == y.shape(0)
            for i in range(x.shape(0)):
                reshape_(x[i], y[i])
        elif x0_divisible_y0:
            # Splitting a dimension. Iterating y
            assert x.shape(0) % y.shape(0) == 0
            x_chunk_len = x.shape(0) // y.shape(0)
            for i in range(y.shape(0)):
                # Construct the slice with `length` here to make the next dimension simple
                reshape_(
                    x[core.ffi.FrontendVarIdx(i * x_chunk_len, None,
                                              x_chunk_len)], y[i])
        elif y0_divisible_x0:
            # Merging dimensions. Iterating x
            assert y.shape(0) % x.shape(0) == 0
            y_chunk_len = y.shape(0) // x.shape(0)
            for i in range(x.shape(0)):
                # Construct the slice with `length` here to make the next dimension simple
                reshape_(
                    x[i], y[core.ffi.FrontendVarIdx(i * y_chunk_len, None,
                                                    y_chunk_len)])
        else:
            # Find next non-affecting dimension, and use one loop to reshape all
            # affecting dimensions before it
            factor_x = (1, {})
            factor_y = (1, {})
            l = 0
            r = 0
            while l < core.ndim(x):
                factor_x = _factor_pairs_mul(factor_x, _factorize(x.shape(l)))
                l += 1
                while r < core.ndim(y):
                    factor_y_new = _factor_pairs_mul(factor_y,
                                                     _factorize(y.shape(r)))
                    if _factor_pairs_divisible(factor_x, factor_y_new):
                        factor_y = factor_y_new
                        r += 1
                    else:
                        break
                if _factor_pairs_divisible(factor_y, factor_x):
                    break
            if not _factor_pairs_divisible(factor_y, factor_x):
                r = core.ndim(y)
            x_lengths = [1] * (l + 1)
            y_lengths = [1] * (r + 1)
            for k in core.static_range(l - 1, -1, -1):
                x_lengths[k] = x.shape(k) * x_lengths[k + 1]
            for k in core.static_range(r - 1, -1, -1):
                y_lengths[k] = y.shape(k) * y_lengths[k + 1]
            assert x_lengths[0] == y_lengths[0]
            for i in range(x_lengths[0]):
                x_next, y_next = x, y
                for k in core.static_range(l):
                    x_next = x_next[i // x_lengths[k + 1] % x.shape(k)]
                for k in core.static_range(r):
                    y_next = y_next[i // y_lengths[k + 1] % y.shape(k)]
                reshape_(x_next, y_next)


@core.inline
def reshape(x, shape):
    '''
    Reshape a tensor into a different shape with the same size

    This operator will try to generate nested loops instead of looping over all
    elements in a plain loop, so schedules can be better applied. It guarantees to
    generates loops in the following cases:

    1. Splitting a dimension. E.g. 4 to 2x2, and there will be a 2x2 loop nest.
    2. Merging dimensions. E.g. 2x2 to 4, and there will be a 2x2 loop nest.
    3. Each non-affecting dimension will be iterated by a unique loop. E.g. 3x5x7
       to 5x3x7, and there will be a 15x7 loop nest, where the "7" dimension will
       be iterated by a unique loop.

    Parameters
    ----------
    x : VarRef
        The input tensor
    shape : list of expression
        The target shape

    Returns
    -------
    VarRef
        The result tensor
    '''
    y = core.empty(shape, core.dtype(x), core.mtype(x))
    reshape_(x, y)
    return y


@core.inline
def _flatten_inner_(x, y):
    if core.ndim(x) == 0:
        y[0] = x
    else:
        #! label: L_inner
        for i in range(x.shape(0)):
            #! label: recur
            _flatten_inner_(
                x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                        (y.shape(0) // x.shape(0))])


@core.inline
def flatten_(x, y, axis: int = 1):
    '''
    Flatten a tensor to have two dimensions, and write to another tensor

    NOTE: This function follows the ONNX convension that reshapes to 2-D instead
    of 1-D.

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    axis : int (Optional)
        The result tensor will have 2 dimensions. All dimensions up to `axis`
        (inclusive) will be flattend to the first dimension. All dimensions after
        `axis` (exclusive) will be flatten to the second dimension. Negative axis
        means counting form the last dimension
    '''
    if axis == 0:
        #! label: recur
        _flatten_inner_(x, y[0])
    else:
        #! label: L_outer
        for i in range(x.shape(0)):
            #! label: recur
            flatten_(
                x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                        (y.shape(0) // x.shape(0))], axis - 1)


def _flatten_comp_shape(x, axis):
    y_shape = [1, 1]
    for i in range(axis):
        y_shape[0] *= core.shape(x, i)
    for i in range(axis, core.ndim(x)):
        y_shape[1] *= core.shape(x, i)
    return y_shape


@core.inline
def flatten(x, axis: int = 1):
    '''
    Flatten a tensor to have two dimensions, and return the result

    NOTE: This function follows the ONNX convension that reshapes to 2-D instead
    of 1-D.

    Parameters
    ----------
    x : VarRef
        The input tensor
    axis : int (Optional)
        The result tensor will have 2 dimensions. All dimensions up to `axis`
        (inclusive) will be flattend to the first dimension. All dimensions after
        `axis` (exclusive) will be flatten to the second dimension. Negative axis
        means counting form the last dimension

    Returns
    -------
    VarRef
        The result tensor
    '''
    y = core.empty(_flatten_comp_shape(x, axis), core.dtype(x), core.mtype(x))
    #! label: recur
    flatten_(x, y, axis)
    return y


flatten_onnx_ = flatten_
''' Alias of `flatten_` '''

flatten_onnx = flatten
''' Alias of `flatten` '''


def _circular_axes(axes, x_ndim):
    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    return sorted(map(lambda x: x if x >= 0 else x_ndim + len(axes) + x, axes))


@core.inline
def unsqueeze_(x, y, axes: Sequence[int]):
    '''
    Insert singleton dimensions to a tensor, and write the result to another tensor

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The resulting tensor
    axes :
        Dimension numbers of the new singleton dimensions. Negative axis means counting
        from the last dimension
    '''
    axes = _circular_axes(axes, core.ndim(x))
    if y.ndim == 0:
        y[()] = x
    elif begin_with_0(axes):
        #! label: recur
        unsqueeze_(x, y[0], all_minus_one(axes[1:]))
    else:
        #! label: L
        for i in range(x.shape(0)):
            #! label: recur
            unsqueeze_(x[i], y[i], all_minus_one(axes))


def _unsqueeze_comp_shape(axes, x):
    y_shape = copy_shape(x)
    for item in axes:
        y_shape.insert(item, 1)
    return y_shape


@core.inline
def unsqueeze(x, axes: Sequence[int]):
    '''
    Insert singleton dimensions to a tensor, and return the result

    Parameters
    ----------
    x : VarRef
        The input tensor
    axes :
        Dimension numbers of the new singleton dimensions. Negative axis means counting
        from the last dimension

    Returns
    -------
    VarRef
        The resulting tensor
    '''
    y = core.empty(_unsqueeze_comp_shape(_circular_axes(axes, core.ndim(x)), x),
                   core.dtype(x), core.mtype(x))
    #! label: recur
    unsqueeze_(x, y, axes)
    return y


@core.inline
def expand_(a, out):
    '''
    Broadcast a tensor to an existing tensor, following the broadcasting rules

    Parameters
    ----------
    a : VarRef
        The input tensor
    b : VarRef
        The broadcasted tensor
    '''
    if out.ndim == 0:
        out[()] = a
    else:
        #! label: L_elem
        for i in range(out.shape(0)):
            if core.ndim(a) < out.ndim:
                #! label: recur
                expand_(a, out[i])
            else:
                #! label: recur
                expand_(a[i % a.shape(0)], out[i])


@core.inline
def expand(a, expand_shape):
    '''
    Broadcast a tensor to a given shape, following the broadcasting rules

    Parameters
    ----------
    a : VarRef
        The input tensor
    b : Sequence of expressions
        The broadcasted shape

    Returns
    -------
    VarRef :
        The broadcasted tensor
    '''
    # FIXME: out_shape = broadcast(a.shape, expand_shape)
    out = core.empty(expand_shape, core.dtype(a), core.mtype(a))
    #! label: recur
    expand_(a, out)
    return out
