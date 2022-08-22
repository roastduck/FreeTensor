from .. import core
from .element_wise import exp, exp_, sub, sub_, truediv, truediv_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


@core.inline
def softmax_(x, y, axis: int = -1):
    '''
    Softmax of tensor `x` along an axis, and write to tensor `y`

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    axis : int (Optional)
        Axis that the softmax is performed along. Negative axis means
        count from the last dimension
    '''
    #! label: max
    maxval = reduce_max(x, axes=[axis], keepdims=True)
    #! label: sub
    corrected = sub(x, maxval)
    #! label: exp
    exponent = exp(corrected)
    #! label: sum
    summation = reduce_sum(exponent, axes=[axis], keepdims=True)
    #! label: div
    truediv_(exponent, summation, y)


@core.inline
def softmax(x, axis=-1):
    '''
    Softmax of tensor `x` along an axis and return the result

    Parameters
    ----------
    x : VarRef
        The input tensor
    axis : int (Optional)
        Axis that the softmax is performed along. Negative axis means
        count from the last dimension

    Returns
    -------
    VarRef :
        The result tensor
    '''
    #! label: max
    maxval = reduce_max(x, axes=[axis], keepdims=True)
    #! label: sub
    corrected = sub(x, maxval)
    #! label: exp
    exponent = exp(corrected)
    #! label: sum
    summation = reduce_sum(exponent, axes=[axis], keepdims=True)
    #! label: div
    out = truediv(exponent, summation)
    return out
