__all__ = [
    'reduce_sum', 'reduce_sum_', 'reduce_prod', 'reduce_prod_', 'all', 'all_',
    'any', 'any_', 'reduce_min', 'reduce_min_', 'reduce_max', 'reduce_max_',
    'reduction_circular_axes', 'reduction_comp_shape'
]

from typing import Sequence, Optional
import functools

from .utils import begin_with_0, all_minus_one, circular_axis
from .. import core


def _named_partial(name: str, doc: str, f, *args, **kvs):
    ''' Similar to functools.partial, but it sets the returned function's __name__ and __doc__ '''

    # This function should be defined in the same file that uses it
    # https://github.com/mkdocstrings/pytkdocs/issues/143

    def g(*_args, **_kvs):
        return f(*args, *_args, **kvs, **_kvs)

    g.__name__ = name
    g.__doc__ = doc
    return g


def reduction_circular_axes(axes, x_ndim, keepdims):
    # ONNX >= 13 treats axes as a tensor, which we don't support for now

    # None for all dimensions
    if axes is None:
        return core.static_range(x_ndim)

    return sorted(map(functools.partial(circular_axis, ndim=x_ndim), axes))


@core.inline
def _init(neutral_val, y):
    if y.ndim == 0:
        #! label: exec
        y[()] = neutral_val
    else:
        #! label: L
        for i in range(y.shape(0)):
            #! label: recur
            _init(neutral_val, y[i])


@core.inline
def _reduce(op, axes, keepdims, x, y):
    if core.ndim(x) == 0:
        #! label: exec
        y[()] = op(y, x)
    else:
        #! label: L
        for i in range(x.shape(0)):
            if begin_with_0(axes):
                if keepdims:
                    assert y.shape(0) == 1
                    #! label: recur
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y[0])
                else:
                    #! label: recur
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y)
            else:
                assert y.shape(0) == x.shape(0)
                #! label: recur
                _reduce(op, all_minus_one(axes), keepdims, x[i], y[i])


@core.inline
def _general_reduce_(op,
                     neutral_val,
                     x,
                     y,
                     axes: Optional[Sequence[int]] = None,
                     keepdims: bool = True):
    #! label: init
    _init(neutral_val, y)
    #! label: reduce
    _reduce(op, reduction_circular_axes(axes, core.ndim(x), keepdims), keepdims,
            x, y)


def reduction_comp_shape(axes, keepdims, x):
    out_shape = []
    for i in range(core.ndim(x)):
        if len(axes) > 0 and axes[0] == i:
            if keepdims:
                out_shape.append(1)
            axes = axes[1:]
        else:
            out_shape.append(x.shape(i))
    return out_shape


@core.inline
def _general_reduce(op,
                    neutral_val,
                    x,
                    axes: Optional[Sequence[int]] = None,
                    keepdims: bool = True):
    #! label: y
    y = core.empty(
        reduction_comp_shape(
            reduction_circular_axes(axes, core.ndim(x), keepdims), keepdims, x),
        core.dtype(x), core.mtype(x))
    #! label: recur
    _general_reduce_(op, neutral_val, x, y,
                     reduction_circular_axes(axes, core.ndim(x), keepdims),
                     keepdims)
    return y


inplace_reduce_doc_template = '''
{} of a tensor through one or more dimensions. The result is written to another tensor

Parameters
----------
x : VarRef
    The input tensor
y : VarRef
    The result tensor
axes : Sequence[int] (Optional)
    Which dimensions to reduce through. Defaults to None, standing for all dimensions,
    i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
keepdims : bool (Optional)
    Keep the reduced dimensions as singleton dimensions. Defaults to True
'''

out_of_place_reduce_doc_template = '''
{} of a tensor through one or more dimensions and return the result

Parameters
----------
x : VarRef
    The input tensor
axes : Sequence[int] (Optional)
    Which dimensions to reduce through. Defaults to None, standing for all dimensions,
    i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
keepdims : bool (Optional)
    Keep the reduced dimensions as singleton dimensions. Defaults to True

Returns
-------
VarRef
    The result tensor
'''

reduce_sum_ = _named_partial("reduce_sum_",
                             inplace_reduce_doc_template.format("Sum"),
                             _general_reduce_, lambda x, y: x + y, 0)
reduce_sum = _named_partial("reduce_sum",
                            out_of_place_reduce_doc_template.format("Sum"),
                            _general_reduce, lambda x, y: x + y, 0)

reduce_prod_ = _named_partial("reduce_prod_",
                              inplace_reduce_doc_template.format("Product"),
                              _general_reduce_, lambda x, y: x * y, 1)
reduce_prod = _named_partial("reduce_prod",
                             out_of_place_reduce_doc_template.format("Product"),
                             _general_reduce, lambda x, y: x * y, 1)

all_ = _named_partial(
    "all_", inplace_reduce_doc_template.format("Reduction of logical and"),
    _general_reduce_, core.l_and, True)
all = _named_partial(
    "all", out_of_place_reduce_doc_template.format("Reduction of logical and"),
    _general_reduce, core.l_and, True)

any_ = _named_partial(
    "any_", inplace_reduce_doc_template.format("Reduction of logical or"),
    _general_reduce_, core.l_or, False)
any = _named_partial(
    "any", out_of_place_reduce_doc_template.format("Reduction of logical or"),
    _general_reduce, core.l_or, False)


@core.inline
def reduce_max_(x, y, axes: Sequence[int] = None, keepdims: bool = True):
    '''
    Maximum of a tensor through one or more dimensions. The result is written to another tensor

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    axes : Sequence[int] (Optional)
        Which dimensions to reduce through. Defaults to None, standing for all dimensions,
        i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True
    '''
    #! label: impl
    _general_reduce_(core.max, core.min_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_max(x, axes: Sequence[int] = None, keepdims: bool = True):
    '''
    Maximum of a tensor through one or more dimensions and return the result

    Parameters
    ----------
    x : VarRef
        The input tensor
    axes : Sequence[int] (Optional)
        Which dimensions to reduce through. Defaults to None, standing for all dimensions,
        i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True

    Returns
    -------
    VarRef
        The result tensor
    '''
    #! label: impl
    y = _general_reduce(core.max, core.min_value(core.dtype(x)), x, axes,
                        keepdims)
    return y


@core.inline
def reduce_min_(x, y, axes: Sequence[int] = None, keepdims: bool = True):
    '''
    Minimum of a tensor through one or more dimensions. The result is written to another tensor

    Parameters
    ----------
    x : VarRef
        The input tensor
    y : VarRef
        The result tensor
    axes : Sequence[int] (Optional)
        Which dimensions to reduce through. Defaults to None, standing for all dimensions,
        i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True
    '''
    #! label: impl
    _general_reduce_(core.min, core.max_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_min(x, axes: Sequence[int] = None, keepdims: bool = True):
    '''
    Minimum of a tensor through one or more dimensions and return the result

    Parameters
    ----------
    x : VarRef
        The input tensor
    axes : Sequence[int] (Optional)
        Which dimensions to reduce through. Defaults to None, standing for all dimensions,
        i.e., reduce the tensor to a scalar. Negative axis means counting form the last dimension
    keepdims : bool (Optional)
        Keep the reduced dimensions as singleton dimensions. Defaults to True

    Returns
    -------
    VarRef
        The result tensor
    '''
    #! label: impl
    y = _general_reduce(core.min, core.max_value(core.dtype(x)), x, axes,
                        keepdims)
    return y
