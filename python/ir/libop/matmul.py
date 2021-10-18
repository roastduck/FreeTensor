import itertools
from typing import Optional

from .. import core
from .assign import add_to, mul_to
from .element_wise import add, mul


def _einsum_(lefts: str, right: str, order: str, init: bool):
    params = [f"X{i}" for i in range(len(lefts))] + ["Y"]

    if len(order) == 0:
        Xs = [f"X{i}[()]" for i in range(len(lefts))]

        code = f'''
def f_einsum({','.join(params)}):
    Y[()] += {' * '.join(Xs)}
'''

    else:
        v = order[0]
        next_lefts = ["'" + left.replace(v, '') + "'" for left in lefts]
        next_right = "'" + right.replace(v, '') + "'"
        next_order = "'" + order[1:] + "'"
        # -1 = not found
        offsets = [left.find(v) for left in lefts] + [right.find(v)]
        arguments = [
            param if offset == -1 else f"{param}.select(i, {offset})"
            for param, offset in zip(params, offsets)
        ]
        for param, offset in zip(params, offsets):
            if offset != -1:
                length = f"{param}.shape({offset})"
        # TODO: Assert lengths of each arguments are consistent
        init_stmt = ''
        next_init = init
        if init and right == '':
            init_stmt = 'Y[()] = 0'
            next_init = False

        code = f'''
def f_einsum({', '.join(params)}):
    {init_stmt}
    for i in range({length}):
        _einsum_([{", ".join(next_lefts)}], {next_right}, {next_order}, {next_init})({', '.join(arguments)})
'''

    _locals = locals()
    exec(code, globals(), _locals)
    return core.inline(_locals['f_einsum'], code)


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
    params = [f"X{i}" for i in range(len(lefts))]
    shape = []
    for v in right:
        offsets = [left.find(v) for left in lefts]
        for param, offset in zip(params, offsets):
            if offset != -1:
                length = f"{param}.shape({offset})"
        shape.append(length)

    # FIXME: compute dtype and mtype from every inputs
    code = f'''
def f_einsum({', '.join(params)}):
    Y = core.create_var([{", ".join(shape)}], X0.dtype, "output", X0.mtype)
    einsum_('{format}')({', '.join(params)}, Y)
    return Y
'''

    _locals = locals()
    exec(code, globals(), _locals)
    return core.inline(_locals['f_einsum'], code)


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
            mul_to(Y, core.Tensor(alpha, Y.mtype))

    else:

        @core.inline
        def f_gemm(A, B, C, Y):
            'nid: einsum'
            einsum_(format)(A, B, Y)
            'nid: mul_to'
            mul_to(Y, core.Tensor(alpha, Y.mtype))
            'nid: add_to'
            add_to(Y, mul(core.Tensor(beta, Y.mtype), C))

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
                                core.up_cast(A.dtype, B.dtype), "output",
                                core.same_mtype(A.mtype, B.mtype))
            'nid: recur'
            gemm_(has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    else:

        @core.inline
        def f_gemm(A, B, C):
            Y = core.create_var(
                comp_shape(A, B),
                core.up_cast(core.up_cast(A.dtype, B.dtype), C.dtype), "output",
                core.same_mtype(core.same_mtype(A.mtype, B.mtype), C.mtype))
            'nid: recur'
            gemm_(has_bias, trans_A, trans_B, alpha, beta)(A, B, Y)
            return Y

    return f_gemm
