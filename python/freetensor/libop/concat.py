__all__ = ['concat', 'concat_']

from typing import Sequence

from .. import core
from .utils import circular_axis


@core.inline
def concat_(inputs: Sequence[core.VarRef], output: core.VarRef, axis: int = 0):
    '''
    Concatenate a list of tensors into a single tensor on an existing axis (in-place)

    All input tensors must have the same shape, except for the dimension size of the
    axis to concatenate on.

    All input tensors must have the same data type and memory type.

    Parameters
    ----------
    inputs: Sequence[VarRef]
        Tensors for concatenation
    output: VarRef
        Concatenation result
    axis: int
        Dimension number for concatenation. Negative axis means counting from the last
        dimension
    '''
    axis = circular_axis(axis, core.ndim(inputs[0]))
    off = 0
    for x in inputs:
        assert core.ndim(x) == core.ndim(inputs[0])
        output.select_slice(off, off + x.shape(axis), dim=axis)[...] = x
        off += x.shape(axis)


@core.inline
def concat(inputs: Sequence[core.VarRef], axis: int = 0):
    '''
    Concatenate a list of tensors into a single tensor on an existing axis (out-of-place)

    All input tensors must have the same shape, except for the dimension size of the
    axis to concatenate on.

    All input tensors must have the same data type and memory type.

    Parameters
    ----------
    inputs: Sequence[VarRef]
        Tensors for concatenation
    axis: int
        Dimension number for concatenation. Negative axis means counting from the last
        dimension

    Returns
    -------
    VarRef
        Concatenation result
    '''
    axis = circular_axis(axis, core.ndim(inputs[0]))
    shape = inputs[0].shape()
    shape[axis] = 0
    for x in inputs:
        assert core.dtype(x) == core.dtype(inputs[0])
        assert core.mtype(x) == core.mtype(inputs[0])
        shape[axis] += x.shape(axis)
    output = core.empty(shape, core.dtype(inputs[0]), core.mtype(inputs[0]))
    concat_(inputs, output, axis)
    return output
