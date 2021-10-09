from typing import Sequence, Optional
import builtins

from .. import core
from .shape_utils import *


def _flatten_inner_(io_mem, idx_dtype="int32"):

    @core.inline
    def f_flatten(x, y):
        if x.ndim == 0:
            y[0] = x[()]
        else:
            'nid: L_inner'
            for i in range(x.shape(0)):
                'nid: recur'
                _flatten_inner_(io_mem, idx_dtype)(
                    x[i], y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                            (y.shape(0) // x.shape(0))])

    return f_flatten


def flatten_(io_mem, idx_dtype="int32", axis=1):

    @core.inline
    def f_flatten(x, y):
        if axis == 0:
            'nid: recur'
            _flatten_inner_(io_mem, idx_dtype)(x, y[0])
        else:
            'nid: L_outer'
            for i in range(x.shape(0)):
                'nid: recur'
                flatten_(io_mem, idx_dtype,
                         axis - 1)(x[i],
                                   y[i * (y.shape(0) // x.shape(0)):(i + 1) *
                                     (y.shape(0) // x.shape(0))])

    return f_flatten


def flatten(io_mem, idx_dtype="int32", axis=1):

    @core.inline
    def f_flatten(x):
        y_shape = core.create_var((2,), idx_dtype, "output", io_mem)
        y_shape[0] = 1
        'nid: L_shape_0'
        for i in range(axis):
            y_shape[0] *= x.shape(i)
        y_shape[1] = 1
        'nid: L_shape_1'
        for i in range(axis, x.ndim):
            y_shape[1] *= x.shape(i)
        y = core.create_var(y_shape, x.dtype, "output", io_mem)
        'nid: recur'
        flatten_(io_mem, idx_dtype, axis)(x, y)
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
        _unsqueeze_(io_mem, circular_axes(axes, x.ndim))(x, y)

    return f_unsqueeze


def _unsqueeze(io_mem, axes: Sequence[int], idx_dtype="int32"):

    def y_ndim(x_ndim, axes):
        return x_ndim + len(axes)

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    def comp_shape(axes):

        @core.inline
        def f_shape(x, y_shape):
            if y_ndim(x.ndim, axes) > 0:
                if begin_with_0(axes):
                    y_shape[0] = 1
                    comp_shape(all_minus_one(axes[1:]))(x, y_shape[1:])
                else:
                    y_shape[0] = x.shape(0)
                    comp_shape(all_minus_one(axes))(x[0], y_shape[1:])

        return f_shape

    @core.inline
    def f_unsqueeze(x):
        y_shape = core.create_var((x.ndim + builtins.len(axes),), idx_dtype,
                                  "output", io_mem)
        'nid: shape'
        comp_shape(axes)(x, y_shape)
        y = core.create_var(y_shape, x.dtype, "output", io_mem)
        'nid: recur'
        _unsqueeze_(io_mem, axes)(x, y)
        return y

    return f_unsqueeze


def unsqueeze(io_mem, axes: Sequence[int], idx_dtype="int32"):

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else x.ndim + len(axes) + x, axes))

    @core.inline
    def f_unsqueeze(y):
        y = _unsqueeze(io_mem, circular_axes(axes, x.ndim), idx_dtype)(y)
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


def expand(io_mem, idx_dtype="int32"):

    @core.inline
    def f_expand(a, expand_shape):
        # FIXME: out_shape = broadcast(a.shape, expand_shape)
        out = core.create_var(expand_shape, a.dtype, "output", io_mem)
        'nid: recur'
        expand_(io_mem)(a, out)
        return out

    return f_expand
