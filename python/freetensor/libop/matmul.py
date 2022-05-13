import itertools
import functools
from typing import Optional, Sequence

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
    if len(order) == 0:
        args[-1][()] += functools.reduce(lambda x, y: x * y,
                                         [x[()] for x in args[:-1]])
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

        next_init = init
        if init and right == '':
            args[-1][()] = 0
            next_init = False

        assert_exprs = []
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
        'prefer_libs'
        for i in range(length):
            _einsum_(
                next_lefts, next_right, next_order, next_init, *[
                    _next_arg(i, arg, offset)
                    for arg, offset in zip(args, offsets)
                ])


@core.inline
def einsum_(fmt, *args):
    lefts, right = fmt.split('->')
    lefts = lefts.split(',')
    order = right
    for left in lefts:
        for idx in left:
            if idx not in order:
                order += idx
    _einsum_(lefts, right, order, True, *args)


@core.inline
def einsum(fmt, *args):
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
    a_fmt = 'z'
    b_fmt = 'z'
    y_fmt = ''
    if a_ndim > 1:
        a_fmt = 'x' + a_fmt
        y_fmt = 'x' + y_fmt
    if b_ndim > 1:
        b_fmt += 'y'
        y_fmt += 'y'
    for i in range(2, max(a_ndim, b_ndim)):
        d = chr(ord('a') + i - 2)
        assert ord('d') < ord('x')
        if i < a_ndim:
            a_fmt = d + a_fmt
        if i < b_ndim:
            b_fmt = d + b_fmt
        y_fmt = d + y_fmt
    return a_fmt + "," + b_fmt + "->" + y_fmt


@core.inline
def matmul_(A, B, Y):
    #! nid: einsum
    einsum_(_make_matmul_fmt(A.ndim, B.ndim), A, B, Y)


@core.inline
def matmul(A, B):
    #! nid: einsum
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

    a_fmt = 'ki' if trans_A else 'ik'
    b_fmt = 'jk' if trans_B else 'kj'
    fmt = f"{a_fmt},{b_fmt}->ij"

    if C is None:
        #! nid: einsum
        einsum_(fmt, A, B, Y)
        #! nid: mul_to
        mul_to(Y, alpha)

    else:
        #! nid: einsum
        einsum_(fmt, A, B, Y)
        #! nid: mul_to
        mul_to(Y, alpha)
        #! nid: add_to
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

    dtype = core.up_cast(A.dtype, B.dtype)
    mtype = core.same_mtype(A.mtype, B.mtype)
    if C is not None:
        dtype = core.up_cast(dtype, C.dtype)
        mtype = core.same_mtype(mtype, C.mtype)

    Y = core.empty(_comp_shape(A, B, trans_A, trans_B), dtype, mtype)
    #! nid: recur
    gemm_(A, B, C, Y, trans_A, trans_B, alpha, beta)
    return Y
