from typing import Sequence, Optional

from .utils import *
from .. import core


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
        'nid: exec'
        y[()] = neutral_val
    else:
        'nid: L'
        for i in range(y.shape(0)):
            'nid: recur'
            _init(neutral_val, y[i])


@core.inline
def _reduce(op, axes, keepdims, x, y):
    if core.ndim(x) == 0:
        'nid: exec'
        y[()] = op(y, x)
    else:
        'nid: L'
        for i in range(x.shape(0)):
            if begin_with_0(axes):
                if keepdims:
                    assert y.shape(0) == 1
                    'nid: recur'
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y[0])
                else:
                    'nid: recur'
                    _reduce(op, all_minus_one(axes[1:]), keepdims, x[i], y)
            else:
                assert y.shape(0) == x.shape(0)
                'nid: recur'
                _reduce(op, all_minus_one(axes), keepdims, x[i], y[i])


@core.inline
def _general_reduce_(op,
                     neutral_val,
                     x,
                     y,
                     axes: Optional[Sequence[int]] = None,
                     keepdims: bool = True):
    'nid: init'
    _init(neutral_val, y)
    'nid: reduce'
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
    'nid: y'
    y = core.empty(
        _comp_shape(_circular_axes(axes, core.ndim(x), keepdims), keepdims, x),
        core.dtype(x), core.mtype(x))
    'nid: recur'
    _general_reduce_(op, neutral_val, x, y,
                     _circular_axes(axes, core.ndim(x), keepdims), keepdims)
    return y


reduce_sum_ = named_partial("reduce_sum_", _general_reduce_, lambda x, y: x + y,
                            0)
reduce_sum = named_partial("reduce_sum", _general_reduce, lambda x, y: x + y, 0)

reduce_mul_ = named_partial("reduce_mul_", _general_reduce_, lambda x, y: x * y,
                            1)
reduce_mul = named_partial("reduce_mul", _general_reduce, lambda x, y: x * y, 1)

reduce_l_and_ = named_partial("reduce_l_and_", _general_reduce_, core.l_and,
                              True)
reduce_l_and = named_partial("reduce_l_and", _general_reduce, core.l_and, True)
all_ = reduce_l_and_
all = reduce_l_and

reduce_l_or_ = named_partial("reduce_l_or_", _general_reduce_, core.l_or, False)
reduce_l_or = named_partial("reduce_l_or", _general_reduce, core.l_or, False)
any_ = reduce_l_or_
any = reduce_l_or


@core.inline
def reduce_max_(x, y, axes: Sequence[int], keepdims: bool = True):
    'nid: impl'
    _general_reduce_(core.max, core.min_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_max(x, axes: Sequence[int], keepdims: bool = True):
    'nid: impl'
    y = _general_reduce(core.max, core.min_value(core.dtype(x)), x, axes,
                        keepdims)
    return y


@core.inline
def reduce_min_(x, y, axes: Sequence[int], keepdims: bool = True):
    'nid: impl'
    _general_reduce_(core.min, core.max_value(core.dtype(x)), x, y, axes,
                     keepdims)


@core.inline
def reduce_min(x, axes: Sequence[int], keepdims: bool = True):
    'nid: impl'
    y = _general_reduce(core.min, core.max_value(core.dtype(x)), x, axes,
                        keepdims)
    return y
