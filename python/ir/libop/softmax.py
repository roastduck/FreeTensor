from .. import core
from .element_wise import exp, exp_, sub, sub_, div, div_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


def softmax_(axis=-1):

    @core.inline
    def f_softmax(x, y):
        'nid: max'
        maxval = reduce_max(axes=[axis], keepdims=True)(x)
        'nid: sub'
        corrected = sub(x, maxval)
        'nid: exp'
        exponent = exp(corrected)
        'nid: sum'
        summation = reduce_sum(axes=[axis], keepdims=True)(exponent)
        'nid: div'
        div_(exponent, summation, y)

    return f_softmax


def softmax(axis=-1):

    @core.inline
    def f_softmax(x):
        'nid: max'
        maxval = reduce_max(axes=[axis], keepdims=True)(x)
        'nid: sub'
        corrected = sub(x, maxval)
        'nid: exp'
        exponent = exp(corrected)
        'nid: sum'
        summation = reduce_sum(axes=[axis], keepdims=True)(exponent)
        'nid: div'
        out = div(exponent, summation)
        return out

    return f_softmax
