__all__ = ['logsumexp', 'logsumexp_']

from .. import core
from .reshape import squeeze
from .element_wise import sub, exp, ln, add_
from .reduction import reduce_max, reduce_sum, reduction_circular_axes, reduction_comp_shape


@core.inline
def logsumexp_(x: core.VarRef,
               y: core.VarRef,
               axis: int = -1,
               keepdims: bool = True):
    '''
    Compute `ln sum_i exp(x_i)`, where `i` is along an axis. Write to tensor `y`.

    The computation is numerically stabilized.

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    axis : int (Optional)
        Axis that the reduction is performed along. Negative axis means counting
        from the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True
    '''
    #! label: max
    maxval = reduce_max(x, axes=[axis], keepdims=True)
    #! label: sub
    corrected = sub(x, maxval)
    #! label: exp
    exponent = exp(corrected)
    #! label: sum
    summation = reduce_sum(exponent, axes=[axis], keepdims=keepdims)
    #! label: ln
    logged = ln(summation)
    #! label: add
    add_(logged, squeeze(maxval, [axis]), y)


@core.inline
def logsumexp(x: core.VarRef,
              axis: int = -1,
              keepdims: bool = True) -> core.VarRef:
    '''
    Compute `ln sum_i exp(x_i)`, where `i` is along an axis. Return the result.

    The computation is numerically stabilized.

    Parameters
    ----------
    x : VarRef
        The input tensor
    axis : int (Optional)
        Axis that the reduction is performed along. Negative axis means counting
        from the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True

    Returns
    -------
    VarRef :
        The result tensor
    '''
    out = core.empty(
        reduction_comp_shape(
            reduction_circular_axes([axis], core.ndim(x), keepdims), keepdims,
            x), core.dtype(x), core.mtype(x))
    logsumexp_(x, out, axis, keepdims)
    return out
