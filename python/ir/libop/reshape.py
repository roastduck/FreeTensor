from typing import Sequence, Optional

from .. import core
from .shape_utils import *


def _flatten_inner_(io_mem):

    @core.inline
    def f_flatten(x, y):
        if x.ndim == 0:
            y[0] = x[()]
        else:
            'nid: L_inner'
            for i in range(x.shape(0)):
                'nid: recur'
                _flatten_inner_(io_mem)(
                    x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                            (y.shape(0) // x.shape(0))])

    return f_flatten


def flatten_(io_mem, axis=1):

    @core.inline
    def f_flatten(x, y):
        if axis == 0:
            'nid: recur'
            _flatten_inner_(io_mem)(x, y[0])
        else:
            'nid: L_outer'
            for i in range(x.shape(0)):
                'nid: recur'
                flatten_(io_mem,
                         axis - 1)(x[i],
                                   y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                                     (y.shape(0) // x.shape(0))])

    return f_flatten


def flatten(io_mem, axis=1):

    def comp_shape(x, axis):
        y_shape = [1, 1]
        for i in range(axis):
            y_shape[0] *= x.shape(i)
        for i in range(axis, x.ndim):
            y_shape[1] *= x.shape(i)
        return y_shape

    @core.inline
    def f_flatten(x):
        y = core.create_var(comp_shape(x, axis), x.dtype, "output", io_mem)
        'nid: recur'
        flatten_(io_mem, axis)(x, y)
        return y

    return f_flatten


def _unsqueeze_(io_mem, axes: Sequence[int]):

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    @core.inline
    def f_unsqueeze(x, y):
        if y.ndim == 0:
            y[()] = x[()]
        elif begin_with_0(axes):
            'nid: recur'
            unsqueeze_(io_mem, all_minus_one(axes[1:]))(x, y[0])
        else:
            'nid: L'
            for i in range(x.shape(0)):
                'nid: recur'
                unsqueeze_(io_mem, all_minus_one(axes))(x[i], y[i])

    return f_unsqueeze


def unsqueeze_(io_mem, axes: Sequence[int]):

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else x.ndim + len(axes) + x, axes))

    @core.inline
    def f_unsqueeze(x, y):
        'nid: impl'
        _unsqueeze_(io_mem, circular_axes(axes, x.ndim))(x, y)

    return f_unsqueeze


def unsqueeze(io_mem, axes: Sequence[int]):

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else x.ndim + len(axes) + x, axes))

    def comp_shape(axes, x):
        y_shape = copy_shape(x)
        for item in axes:
            y_shape.insert(item, 1)
        return y_shape

    @core.inline
    def f_unsqueeze(x):
        y = core.create_var(comp_shape(circular_axes(axes, x.ndim), x), x.dtype,
                            "output", io_mem)
        'nid: recur'
        _unsqueeze_(io_mem, circular_axes(axes, x.ndim))(x, y)
        return y

    return f_unsqueeze


def expand_(io_mem):

    @core.inline
    def f_expand(a, out):
        if out.ndim == 0:
            out[()] = a[()]
        else:
            'nid: L_elem'
            for i in range(out.shape(0)):
                if a.ndim < out.ndim:
                    'nid: recur'
                    expand_(io_mem)(a, out[i])
                else:
                    'nid: recur'
                    expand_(io_mem,)(a[i % a.shape(0)], out[i])

    return f_expand


def expand(io_mem):

    @core.inline
    def f_expand(a, expand_shape):
        # FIXME: out_shape = broadcast(a.shape, expand_shape)
        out = core.create_var(expand_shape, a.dtype, "output", io_mem)
        'nid: recur'
        expand_(io_mem)(a, out)
        return out

    return f_expand
