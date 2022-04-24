import itertools
import functools
from typing import Optional, Sequence

from .. import core
from .assign import add_to, mul_to
from .element_wise import add, mul


def _einsum_(lefts: Sequence[str], right: str, order: str, init: bool):

    def next_arg(i, arg, offset):
        if offset == -1:
            return arg
        else:
            return arg.select(i % arg.shape(offset), offset)

    @core.inline
    def f_einsum(*args):
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
            length = functools.reduce(core.max, [
                arg.shape(offset)
                for arg, offset in zip(iter_args, iter_offsets)
            ])

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
                _einsum_(next_lefts, next_right, next_order, next_init)(*[
                    next_arg(i, arg, offset)
                    for arg, offset in zip(args, offsets)
                ])

    return f_einsum


def einsum_(format):
    lefts, right = format.split('->')
    lefts = lefts.split(',')
    order = right
    for left in lefts:
        for idx in left:
            if idx not in order:
                order += idx
    return _einsum_(lefts, right, order, True)


def einsum(format):
    lefts, right = format.split('->')
    lefts = lefts.split(',')

    @core.inline(verbose=True)
    def f_einsum(*args):
        shapes = []
        for v in right:
            offsets = [left.find(v) for left in lefts]
            iter_args, iter_offsets = zip(
                *filter(lambda x: x[1] != -1, zip(args, offsets)))
            assert len(iter_args) > 0
            shapes.append(iter_args[0].shape(iter_offsets[0]))

        # FIXME: compute dtype and mtype from every inputs
        Y = core.create_var(shapes, args[0].dtype, args[0].mtype)
        einsum_(format)(*args, Y)
        return Y

    return f_einsum


def _make_matmul_format(a_ndim, b_ndim):
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
    'nid: einsum'
    einsum_(_make_matmul_format(A.ndim, B.ndim))(A, B, Y)


@core.inline
def matmul(A, B):
    'nid: einsum'
    Y = einsum(_make_matmul_format(A.ndim, B.ndim))(A, B)
    return Y


def gemm_(has_bias: bool = False,
          trans_A: bool = False,
          trans_B: bool = False,
          alpha: float = 1.0,
          beta: float = 1.0):

    a_format = 'ki' if trans_A else 'ik'
    b_format = 'jk' if trans_B else 'kj'
    format = f"{a_format},{b_format}->ij"

    if not has_bias:

        @core.inline
        def f_gemm(A, B, Y):
            'nid: einsum'
            einsum_(format)(A, B, Y)
            'nid: mul_to'
            mul_to(Y, alpha)

    else:

        @core.inline
        def f_gemm(A, B, C, Y):
            'nid: einsum'
            einsum_(format)(A, B, Y)
            'nid: mul_to'
            mul_to(Y, alpha)
            'nid: add_to'
            add_to(Y, mul(beta, C))

    return f_gemm


def gemm(has_bias: bool = False,
         trans_A: bool = False,
         trans_B: bool = False,
         alpha: float = 1.0,
         beta: float = 1.0):

    def comp_shape(A, B):
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

    if not has_bias:

        @core.inline
        def f_gemm(A, B):
            Y = core.create_var(comp_shape(A, B),
                                core.up_cast(A.dtype, B.dtype),
                                core.same_mtype(A.mtype, B.mtype))
            'nid: recur'
            gemm_(has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    else:

        @core.inline
        def f_gemm(A, B, C):
            Y = core.create_var(
                comp_shape(A, B),
                core.up_cast(core.up_cast(A.dtype, B.dtype), C.dtype),
                core.same_mtype(core.same_mtype(A.mtype, B.mtype), C.mtype))
            'nid: recur'
            gemm_(has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    return f_gemm
