from .. import core
from .element_wise import exp, exp_, sub, sub_, truediv, truediv_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


@core.inline
def softmax_(x, y, axis=-1):
    'nid: max'
    maxval = reduce_max(x, axes=[axis], keepdims=True)
    'nid: sub'
    corrected = sub(x, maxval)
    'nid: exp'
    exponent = exp(corrected)
    'nid: sum'
    summation = reduce_sum(exponent, axes=[axis], keepdims=True)
    'nid: div'
    truediv_(exponent, summation, y)


@core.inline
def softmax(x, axis=-1):
    'nid: max'
    maxval = reduce_max(x, axes=[axis], keepdims=True)
    'nid: sub'
    corrected = sub(x, maxval)
    'nid: exp'
    exponent = exp(corrected)
    'nid: sum'
    summation = reduce_sum(exponent, axes=[axis], keepdims=True)
    'nid: div'
    out = truediv(exponent, summation)
    return out
