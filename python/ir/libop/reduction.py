from typing import Sequence, Optional

from .. import core


def _general_reduce_(io_mem,
                     op,
                     neutral_val,
                     axes: Optional[Sequence[int]] = None,
                     keepdims: bool = True):

    def y_ndim(x_ndim, axes):
        return x_ndim if keepdims else x_ndim - len(axes)

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else y_ndim(x_ndim, axes) + x, axes))

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    def init():

        @core.inline
        def f_init(y):
            if y.ndim == 0:
                'nid: exec'
                y[()] = neutral_val
            else:
                'nid: L'
                for i in range(y.shape(0)):
                    'nid: recur'
                    init()(y[i])

        return f_init

    def reduce(axes):

        @core.inline
        def f_reduce(x, y):
            if x.ndim == 0:
                'nid: exec'
                y[()] = op(y[()], x[()])
            else:
                'nid: L'
                for i in range(x.shape(0)):
                    if begin_with_0(axes):
                        if keepdims:
                            'nid: recur'
                            reduce(all_minus_one(axes[1:]))(x[i], y[0])
                        else:
                            'nid: recur'
                            reduce(all_minus_one(axes[1:]))(x[i], y)
                    else:
                        'nid: recur'
                        reduce(all_minus_one(axes))(x[i], y[i])

        return f_reduce

    @core.inline
    def f_reduce(x, y):
        'nid: init'
        init()(y)
        'nid: reduce'
        reduce(circular_axes(axes, x.ndim))(x, y)

    return f_reduce


def _general_reduce(io_mem,
                    op,
                    neutral_val,
                    axes: Optional[Sequence[int]] = None,
                    keepdims: bool = True,
                    idx_dtype="int32"):

    def y_ndim(x_ndim, axes):
        return x_ndim if keepdims else x_ndim - len(axes)

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else y_ndim(x_ndim, axes) + x, axes))

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    def comp_shape(axes):

        @core.inline
        def f_shape(x, y_shape):
            if y_ndim(x.ndim, axes) > 0:
                if begin_with_0(axes):
                    if keepdims:
                        y_shape[0] = 1
                        comp_shape(all_minus_one(axes[1:]))(x[0], y_shape[1:])
                    else:
                        comp_shape(all_minus_one(axes[1:]))(x[0], y_shape)
                else:
                    y_shape[0] = x.shape(0)
                    comp_shape(all_minus_one(axes))(x[0], y_shape[1:])

        return f_shape

    @core.inline
    def f_reduce(x):
        y_shape = core.create_var((y_ndim(x.ndim, axes),), idx_dtype, "output",
                                  io_mem)
        'nid: shape'
        comp_shape(circular_axes(axes, x.ndim))(x, y_shape)
        y = core.create_var(y_shape, x.dtype, "output", io_mem)
        'nid: recur'
        _general_reduce_(io_mem, op, neutral_val, circular_axes(axes, x.ndim),
                         keepdims)(x, y)
        return y

    return f_reduce


def reduce_sum_(io_mem, axes: Sequence[int], keepdims: bool = True):
    return _general_reduce_(io_mem, lambda x, y: x + y, 0, axes, keepdims)


def reduce_sum(io_mem,
               axes: Sequence[int],
               keepdims: bool = True,
               idx_dtype="int32"):
    return _general_reduce(io_mem, lambda x, y: x + y, 0, axes, keepdims)


def reduce_max_(io_mem, axes: Sequence[int], keepdims: bool = True):
    op = lambda x, y: core.max(x, y)

    @core.inline
    def f(x, y):
        'nid: impl'
        _general_reduce_(io_mem, op, core.min_value(x.dtype), axes, keepdims)(x,
                                                                              y)

    return f


def reduce_max(io_mem,
               axes: Sequence[int],
               keepdims: bool = True,
               idx_dtype="int32"):
    op = lambda x, y: core.max(x, y)

    @core.inline
    def f(x):
        'nid: impl'
        y = _general_reduce(io_mem, op, core.min_value(x.dtype), axes,
                            keepdims)(x)
        return y

    return f
