__all__ = ['einsum', 'einsum_', 'matmul', 'matmul_', 'gemm', 'gemm_']

import functools
from typing import Sequence

from .. import core
from .assign import add_to, mul_to
from .element_wise import add, mul


def _next_arg(i, arg, offset):
    if offset == -1:
        return arg
    else:
        return arg.select(i % arg.shape(offset), offset)


@core.inline
def _einsum_(lefts: Sequence[str], right: str, order: str, init: bool, *args):
    next_init = init
    if init and right == '':
        args[-1][()] = 0
        next_init = False

    if len(order) == 0:
        args[-1][()] += functools.reduce(lambda x, y: x * y, args[:-1])
    else:
        v = order[0]
        next_lefts = [left.replace(v, '') for left in lefts]
        next_right = right.replace(v, '')
        next_order = order[1:]
        # -1 = not found
        offsets = [left.find(v) for left in lefts] + [right.find(v)]
        iter_args, iter_offsets = zip(
            *filter(lambda x: x[1] != -1, zip(args, offsets)))
        length = functools.reduce(
            core.max,
            [arg.shape(offset) for arg, offset in zip(iter_args, iter_offsets)])

        if right != '':
            assert offsets[-1] != -1
            iter_left_args, iter_left_offset = zip(
                *filter(lambda x: x[1] != -1, zip(args[:-1], offsets[:-1])))
            for arg, offset in zip(iter_left_args, iter_left_offset):
                assert arg.shape(offset) == args[-1].shape(
                    offsets[-1]) or arg.shape(offset) == 1
        else:
            for arg, offset in zip(iter_args, iter_offsets):
                assert arg.shape(offset) == length or arg.shape(offset) == 1
        #! prefer_libs
        for i in range(length):
            _einsum_(
                next_lefts, next_right, next_order, next_init, *[
                    _next_arg(i, arg, offset)
                    for arg, offset in zip(args, offsets)
                ])


@core.inline
def einsum_(fmt: str, *args):
    '''
    Einstein summation. The result is written to the last argument

    Parameters
    ----------
    fmt : str
        The format string. E.g. `"ik,kj->ij"` represents a matrix multiplcation
    args : Sequence[VarRef]
        All arguments including inputs and the output. E.g. if `fmt` is `"ik,kj->ij"`,
        it iterates axis `i` and `k` of `args[0]`, axis `k` and `j` of `args[1]`,
        axis `i` and `j` of `args[2]`
    '''
    lefts, right = fmt.split('->')
    lefts = lefts.split(',')
    order = right
    for left in lefts:
        for idx in left:
            if idx not in order:
                order += idx
    _einsum_(lefts, right, order, True, *args)


@core.inline
def einsum(fmt: str, *args):
    '''
    Einstein summation. The result is returned

    Parameters
    ----------
    fmt : str
        The format string. E.g. `"ik,kj->ij"` represents a matrix multiplcation
    args : Sequence[VarRef]
        All inputs arguments. E.g. if `fmt` is `"ik,kj->ij"`,
        it iterates axis `i` and `k` of `args[0]`, axis `k` and `j` of `args[1]`,
        axis `i` and `j` of the returned value

    Returns
    -------
    VarRef :
        The result tensor
    '''
    lefts, right = fmt.split('->')
    lefts = lefts.split(',')

    shapes = []
    for v in right:
        offsets = [left.find(v) for left in lefts]
        iter_args, iter_offsets = zip(
            *filter(lambda x: x[1] != -1, zip(args, offsets)))
        assert len(iter_args) > 0
        shapes.append(iter_args[0].shape(iter_offsets[0]))

    # FIXME: compute dtype and mtype from every inputs
    Y = core.empty(shapes, args[0].dtype, args[0].mtype)
    einsum_(fmt, *args, Y)
    return Y


def _make_matmul_fmt(a_ndim, b_ndim):
    a_fmt = ''
    b_fmt = ''
    y_fmt = ''
    if a_ndim > 0 and b_ndim > 0:
        a_fmt = 'z'
        b_fmt = 'z'
        a_ndim -= 1
        b_ndim -= 1
    if a_ndim > 0:
        a_fmt = 'x' + a_fmt
        y_fmt = 'x' + y_fmt
        a_ndim -= 1
    if b_ndim > 0:
        b_fmt += 'y'
        y_fmt += 'y'
        b_ndim -= 1
    for i in range(max(a_ndim, b_ndim)):
        d = chr(ord('a') + i)
        assert ord(d) < ord('x')
        if i < a_ndim:
            a_fmt = d + a_fmt
        if i < b_ndim:
            b_fmt = d + b_fmt
        y_fmt = d + y_fmt
    return a_fmt + "," + b_fmt + "->" + y_fmt


@core.inline
def matmul_(A, B, Y):
    '''
    Matrix multiplcation. The result is written to an existing tensor

    Parameters
    ----------
    A : VarRef
        The left-hand-side operand
    B : VarRef
        The right-hand-side operand
    C : VarRef
        The resulting tensor
    '''
    #! label: einsum
    einsum_(_make_matmul_fmt(A.ndim, B.ndim), A, B, Y)


@core.inline
def matmul(A, B):
    '''
    Matrix multiplcation. The result is returned

    Parameters
    ----------
    A : VarRef
        The left-hand-side operand
    B : VarRef
        The right-hand-side operand

    Returns
    -------
    VarRef :
        The resulting tensor
    '''
    #! label: einsum
    Y = einsum(_make_matmul_fmt(A.ndim, B.ndim), A, B)
    return Y


@core.inline
def gemm_(A,
          B,
          C,
          Y,
          trans_A: bool = False,
          trans_B: bool = False,
          alpha: float = 1.0,
          beta: float = 1.0):
    '''
    General matrix multiplcation following BLAS convention. The result is written to an existing tensor

    It performs `Y = alpha tr?(A) @ tr?(B) + C`, where `@` represents matrix multiplication, `tr?`
    represents an optional transposition

    Parameters
    ----------
    A : VarRef
        The left-hand-side operand of matrix multiplication
    B : VarRef
        The right-hand-side operand of matrix multiplication
    C : VarRef (Optional)
        The bias tensor
    Y : VarRef
        The resulting tensor
    trans_A : bool (Optional)
        If true, transpose `A`. Defaults to False
    trans_B : bool (Optional)
        If true, transpose `B`. Defaults to False
    alpha : Number (Optional)
        Coefficient of `tr?(A) @ tr?(B)`. Defaults to 1.0
    beta : Number (Optional)
        Coefficient of `C`. Defaults to 1.0
    '''

    a_fmt = 'ki' if trans_A else 'ik'
    b_fmt = 'jk' if trans_B else 'kj'
    fmt = f"{a_fmt},{b_fmt}->ij"

    if C is None:
        #! label: einsum
        einsum_(fmt, A, B, Y)
        #! label: mul_to
        mul_to(Y, alpha)

    else:
        #! label: einsum
        einsum_(fmt, A, B, Y)
        #! label: mul_to
        mul_to(Y, alpha)
        #! label: add_to
        add_to(Y, mul(beta, C))


def _comp_shape(A, B, trans_A, trans_B):
    if not trans_A:
        if not trans_B:
            return [A.shape(0), B.shape(1)]
        else:
            return [A.shape(0), B.shape(0)]
    else:
        if not trans_B:
            return [A.shape(1), B.shape(1)]
        else:
            return [A.shape(1), B.shape(0)]


@core.inline
def gemm(A,
         B,
         C=None,
         has_bias: bool = False,
         trans_A: bool = False,
         trans_B: bool = False,
         alpha: float = 1.0,
         beta: float = 1.0):
    '''
    General matrix multiplcation following BLAS convention and return the result

    It performs `Y = alpha tr?(A) @ tr?(B) + C`, where `@` represents matrix multiplication, `tr?`
    represents an optional transposition

    Parameters
    ----------
    A : VarRef
        The left-hand-side operand of matrix multiplication
    B : VarRef
        The right-hand-side operand of matrix multiplication
    C : VarRef (Optional)
        The bias tensor
    trans_A : bool (Optional)
        If true, transpose `A`. Defaults to False
    trans_B : bool (Optional)
        If true, transpose `B`. Defaults to False
    alpha : Number (Optional)
        Coefficient of `tr?(A) @ tr?(B)`. Defaults to 1.0
    beta : Number (Optional)
        Coefficient of `C`. Defaults to 1.0

    Returns
    -------
    VarRef :
        The resulting tensor
    '''

    dtype = core.up_cast(A.dtype, B.dtype).base
    mtype = core.same_mtype(A.mtype, B.mtype)
    if C is not None:
        dtype = core.up_cast(dtype, C.dtype).base
        mtype = core.same_mtype(mtype, C.mtype)

    Y = core.empty(_comp_shape(A, B, trans_A, trans_B), dtype, mtype)
    #! label: recur
    gemm_(A, B, C, Y, trans_A, trans_B, alpha, beta)
    return Y
