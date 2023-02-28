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
    with core.StmtRange() as rng:
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

        exponent_handle = core.push_for_backward(exponent)
        summation_handle = core.push_for_backward(summation)
        y_handle = core.push_for_backward(y)

    with core.UserGrad(x, y, stmt_range=rng) as (d_x, d_y):
        d_summation = -reduce_sum(d_y * y_handle, axes=[axis
                                                       ]) / summation_handle
        d_exponent = d_y / summation_handle + d_summation
        d_x[...] += d_exponent * exponent_handle


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
    out = core.empty(x.shape(), x.dtype, x.mtype)
    softmax_(x, out, axis)
    return out
