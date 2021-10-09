from .. import core
from .element_wise import exp, exp_, sub, sub_, div, div_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


def softmax_(io_mem, axis=-1, idx_dtype="int32"):

    @core.inline
    def f_softmax(x, y):
        'nid: max'
        maxval = reduce_max(io_mem,
                            axes=[axis],
                            keepdims=True,
                            idx_dtype=idx_dtype)(x)
        'nid: sub'
        corrected = sub(io_mem, idx_dtype)(x, maxval)
        'nid: exp'
        exponent = exp(io_mem, idx_dtype)(corrected)
        'nid: sum'
        summation = reduce_sum(io_mem,
                               axes=[axis],
                               keepdims=True,
                               idx_dtype=idx_dtype)(exponent)
        'nid: div'
        div_(io_mem)(exponent, summation, y)

    return f_softmax


def softmax(io_mem, axis=-1, idx_dtype="int32"):

    @core.inline
    def f_softmax(x):
        'nid: max'
        maxval = reduce_max(io_mem,
                            axes=[axis],
                            keepdims=True,
                            idx_dtype=idx_dtype)(x)
        'nid: sub'
        corrected = sub(io_mem, idx_dtype)(x, maxval)
        'nid: exp'
        exponent = exp(io_mem, idx_dtype)(corrected)
        'nid: sum'
        summation = reduce_sum(io_mem,
                               axes=[axis],
                               keepdims=True,
                               idx_dtype=idx_dtype)(exponent)
        'nid: div'
        out = div(io_mem, idx_dtype)(exponent, summation)
        return out

    return f_softmax
