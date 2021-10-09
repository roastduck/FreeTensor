from typing import Sequence, Optional

from .. import core


def _general_reduce_(op,
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


def _general_reduce(op,
                    neutral_val,
                    axes: Optional[Sequence[int]] = None,
                    keepdims: bool = True):

    def y_ndim(x_ndim, axes):
        return x_ndim if keepdims else x_ndim - len(axes)

    def circular_axes(axes, x_ndim):
        # ONNX >= 13 treats axes as a tensor, which we don't support for now
        return sorted(
            map(lambda x: x if x >= 0 else y_ndim(x_ndim, axes) + x, axes))

    def comp_shape(axes, x):
        out_shape = []
        for i in range(x.ndim):
            if len(axes) > 0 and axes[0] == i:
                if keepdims:
                    out_shape.append(1)
                axes = axes[1:]
            else:
                out_shape.append(x.shape(i))
        return out_shape

    @core.inline
    def f_reduce(x):
        y = core.create_var(comp_shape(circular_axes(axes, x.ndim), x), x.dtype,
                            "output", x.mtype)
        'nid: recur'
        _general_reduce_(op, neutral_val, circular_axes(axes, x.ndim),
                         keepdims)(x, y)
        return y

    return f_reduce


def reduce_sum_(axes: Sequence[int], keepdims: bool = True):
    return _general_reduce_(lambda x, y: x + y, 0, axes, keepdims)


def reduce_sum(axes: Sequence[int], keepdims: bool = True):
    return _general_reduce(lambda x, y: x + y, 0, axes, keepdims)


def reduce_max_(axes: Sequence[int], keepdims: bool = True):
    op = lambda x, y: core.max(x, y)

    @core.inline
    def f(x, y):
        'nid: impl'
        _general_reduce_(op, core.min_value(x.dtype), axes, keepdims)(x, y)

    return f


def reduce_max(axes: Sequence[int], keepdims: bool = True):
    op = lambda x, y: core.max(x, y)

    @core.inline
    def f(x):
        'nid: impl'
        y = _general_reduce(op, core.min_value(x.dtype), axes, keepdims)(x)
        return y

    return f
