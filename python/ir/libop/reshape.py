from typing import Sequence, Optional

from .. import core
from .utils import *
from .shape_utils import *


@core.inline
def _flatten_inner_(x, y):
    if core.ndim(x) == 0:
        y[0] = x
    else:
        'nid: L_inner'
        for i in range(x.shape(0)):
            'nid: recur'
            _flatten_inner_(
                x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                        (y.shape(0) // x.shape(0))])


@core.inline
def flatten_(x, y, axis=1):
    if axis == 0:
        'nid: recur'
        _flatten_inner_(x, y[0])
    else:
        'nid: L_outer'
        for i in range(x.shape(0)):
            'nid: recur'
            flatten_(
                x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                        (y.shape(0) // x.shape(0))], axis - 1)


def _flatten_comp_shape(x, axis):
    y_shape = [1, 1]
    for i in range(axis):
        y_shape[0] *= core.shape(x, i)
    for i in range(axis, core.ndim(x)):
        y_shape[1] *= core.shape(x, i)
    return y_shape


@core.inline
def flatten(x, axis=1):
    y = core.create_var(_flatten_comp_shape(x, axis), core.dtype(x),
                        core.mtype(x))
    'nid: recur'
    flatten_(x, y, axis)
    return y


def _circular_axes(axes, x_ndim):
    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    return sorted(map(lambda x: x if x >= 0 else x_ndim + len(axes) + x, axes))


@core.inline
def unsqueeze_(x, y, axes: Sequence[int]):
    axes = _circular_axes(axes, core.ndim(x))
    if y.ndim == 0:
        y[()] = x
    elif begin_with_0(axes):
        'nid: recur'
        unsqueeze_(x, y[0], all_minus_one(axes[1:]))
    else:
        'nid: L'
        for i in range(x.shape(0)):
            'nid: recur'
            unsqueeze_(x[i], y[i], all_minus_one(axes))


def _unsqueeze_comp_shape(axes, x):
    y_shape = copy_shape(x)
    for item in axes:
        y_shape.insert(item, 1)
    return y_shape


@core.inline
def unsqueeze(x, axes: Sequence[int]):
    y = core.create_var(
        _unsqueeze_comp_shape(_circular_axes(axes, core.ndim(x)), x),
        core.dtype(x), core.mtype(x))
    'nid: recur'
    unsqueeze_(x, y, axes)
    return y


@core.inline
def expand_(a, out):
    if out.ndim == 0:
        out[()] = a
    else:
        'nid: L_elem'
        for i in range(out.shape(0)):
            if core.ndim(a) < out.ndim:
                'nid: recur'
                expand_(a, out[i])
            else:
                'nid: recur'
                expand_(a[i % a.shape(0)], out[i])


@core.inline
def expand(a, expand_shape):
    # FIXME: out_shape = broadcast(a.shape, expand_shape)
    out = core.create_var(expand_shape, core.dtype(a), core.mtype(a))
    'nid: recur'
    expand_(a, out)
    return out
