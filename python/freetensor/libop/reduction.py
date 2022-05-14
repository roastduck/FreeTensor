from typing import Sequence, Optional

from .utils import *
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


def _y_ndim(x_ndim, axes, keepdims):
    return x_ndim if keepdims else x_ndim - len(axes)


def _circular_axes(axes, x_ndim, keepdims):
    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    return sorted(
        map(lambda x: x
            if x >= 0 else _y_ndim(x_ndim, axes, keepdims) + x, axes))


@core.inline
def _init(neutral_val, y):
    if y.ndim == 0:
        #! nid: exec
        y[()] = neutral_val
    else:
        #! nid: L
        for i in range(y.shape(0)):
            #! nid: recur
            _init(neutral_val, y[i])


@core.inline
def _reduce(op, axes, keepdims, x, y):
    if core.ndim(x) == 0:
        #! nid: exec
        y[()] = op(y, x)
    else:
        #! nid: L
        for i in range(x.shape(0)):
            if begin_with_0(axes):
                if keepdims:
                    assert y.shape(0) == 1
                    #! nid: recur
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y[0])
                else:
                    #! nid: recur
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y)
            else:
                assert y.shape(0) == x.shape(0)
                #! nid: recur
                _reduce(op, all_minus_one(axes), keepdims, x[i], y[i])


@core.inline
def _general_reduce_(op,
                     neutral_val,
                     x,
                     y,
                     axes: Optional[Sequence[int]] = None,
                     keepdims: bool = True):
    #! nid: init
    _init(neutral_val, y)
    #! nid: reduce
    _reduce(op, _circular_axes(axes, core.ndim(x), keepdims), keepdims, x, y)


def _comp_shape(axes, keepdims, x):
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
    #! nid: y
    y = core.empty(
        _comp_shape(_circular_axes(axes, core.ndim(x), keepdims), keepdims, x),
        core.dtype(x), core.mtype(x))
    #! nid: recur
    _general_reduce_(op, neutral_val, x, y,
                     _circular_axes(axes, core.ndim(x), keepdims), keepdims)
    return y


implace_reduce_doc_template = '''
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
                             implace_reduce_doc_template.format("Sum"),
                             _general_reduce_, lambda x, y: x + y, 0)
reduce_sum = _named_partial("reduce_sum",
                            out_of_place_reduce_doc_template.format("Sum"),
                            _general_reduce, lambda x, y: x + y, 0)

reduce_prod_ = _named_partial("reduce_prod_",
                              implace_reduce_doc_template.format("Product"),
                              _general_reduce_, lambda x, y: x * y, 1)
reduce_prod = _named_partial("reduce_prod",
                             out_of_place_reduce_doc_template.format("Product"),
                             _general_reduce, lambda x, y: x * y, 1)

all_ = _named_partial(
    "all_", implace_reduce_doc_template.format("Reduction of logical and"),
    _general_reduce_, core.l_and, True)
all = _named_partial(
    "all", out_of_place_reduce_doc_template.format("Reduction of logical and"),
    _general_reduce, core.l_and, True)

any_ = _named_partial(
    "any_", implace_reduce_doc_template.format("Reduction of logical or"),
    _general_reduce_, core.l_or, False)
any = _named_partial(
    "any", out_of_place_reduce_doc_template.format("Reduction of logical or"),
    _general_reduce, core.l_or, False)


@core.inline
def reduce_max_(x, y, axes: Sequence[int], keepdims: bool = True):
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
    #! nid: impl
    _general_reduce_(core.max, core.min_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_max(x, axes: Sequence[int], keepdims: bool = True):
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
    #! nid: impl
    y = _general_reduce(core.max, core.min_value(core.dtype(x)), x, axes,
                        keepdims)
    return y


@core.inline
def reduce_min_(x, y, axes: Sequence[int], keepdims: bool = True):
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
    #! nid: impl
    _general_reduce_(core.min, core.max_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_min(x, axes: Sequence[int], keepdims: bool = True):
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
    #! nid: impl
    y = _general_reduce(core.min, core.max_value(core.dtype(x)), x, axes,
                        keepdims)
    return y
