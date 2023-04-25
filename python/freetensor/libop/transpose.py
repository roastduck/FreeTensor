__all__ = ['transpose', 'transpose_']

from typing import Sequence

from .. import core
from .utils import all_minus_one, circular_axis


def _get_transpose_perm(perm, ndim):
    if perm is None:
        perm = list(reversed(core.static_range(ndim)))
    return [circular_axis(d, ndim) for d in perm]


@core.inline
def transpose_(x: core.VarRef, y: core.VarRef, perm: Sequence[int] = None):
    '''
    Transposition (in-place)

    The `perm[i]`-th dimension of the input becomes the `i`-th dimension of
    the output

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The output tensor
    perm : Sequence[int]
        Permutation of the dimensions. Negative values mean counting form the
        last dimension. By default reversing all dimensions
    '''

    perm = _get_transpose_perm(perm, core.ndim(x))
    if core.ndim(x) == 0:
        y[...] = x[...]
    else:
        for i in range(y.shape(0)):
            transpose_(x.select(i, dim=perm[0]), y[i], perm=all_minus_one(perm))


@core.inline
def transpose(x: core.VarRef, perm: Sequence[int] = None):
    '''
    Transposition (out-of-place)

    The `perm[i]`-th dimension of the input becomes the `i`-th dimension of
    the output

    Parameters
    ----------
    x : VarRef
        The input tensor
    perm : Sequence[int]
        Permutation of the dimensions. Negative values mean counting form the
        last dimension. By default reversing all dimensions

    Returns
    -------
    VarRef
        The output tensor
    '''

    perm = _get_transpose_perm(perm, core.ndim(x))
    y = core.empty([x.shape(perm[d]) for d in core.static_range(core.ndim(x))],
                   core.dtype(x), core.mtype(x))
    transpose_(x, y, perm)
    return y
