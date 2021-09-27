from typing import Sequence, Optional

from .. import core
from .common import StaticType as StaticType


def _general_reduce_(t_x: StaticType,
                     t_y: StaticType,
                     io_mem,
                     op,
                     neutral_val,
                     idx_dtype="int32",
                     axes: Optional[Sequence[int]] = None,
                     keepdims: bool = True):

    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    assert axes is not None, "Currently only unsqueeze for ONNX < 13 is supported"
    axes = sorted(map(lambda x: x if x >= 0 else t_y.ndim + x, axes))

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    def init(t_y: StaticType):

        @core.transform
        def f_init(y_shape, y):
            'nid: V_y_shape'
            core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
            'nid: V_y'
            core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

            if t_y.ndim == 0:
                'nid: exec'
                y[()] = neutral_val
            else:
                'nid: L'
                for i in range(y_shape[0]):
                    'nid: recur'
                    init(t_y.one_less_dim())(y_shape[1:], y[i])

        return f_init

    def reduce(t_x: StaticType, t_y: StaticType, axes):

        @core.transform
        def f_reduce(x_shape, y_shape, x, y):
            'nid: V_x_shape'
            core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
            'nid: V_y_shape'
            core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
            'nid: V_x'
            core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
            'nid: V_y'
            core.declare_var(y, y_shape, t_x.elem_type, "inout", io_mem)

            if t_x.ndim == 0:
                'nid: exec'
                y[()] = op(y[()], x[()])
            else:
                'nid: L'
                for i in range(x_shape[0]):
                    if begin_with_0(axes):
                        if keepdims:
                            'nid: recur'
                            reduce(t_x.one_less_dim(), t_y.one_less_dim(),
                                   all_minus_one(axes[1:]))(x_shape[1:],
                                                            y_shape[1:], x[i],
                                                            y[0])
                        else:
                            'nid: recur'
                            reduce(t_x.one_less_dim(), t_y,
                                   all_minus_one(axes[1:]))(x_shape[1:],
                                                            y_shape, x[i], y)
                    else:
                        'nid: recur'
                        reduce(t_x.one_less_dim(), t_y.one_less_dim(),
                               all_minus_one(axes))(x_shape[1:], y_shape[1:],
                                                    x[i], y[i])

        return f_reduce

    @core.transform
    def f_reduce(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        'nid: init'
        init(t_y)(y_shape, y)
        'nid: reduce'
        reduce(t_x, t_y, axes)(x_shape, y_shape, x, y)

    return f_reduce


def _general_reduce(t_x: StaticType,
                    t_y: StaticType,
                    io_mem,
                    op,
                    neutral_val,
                    idx_dtype="int32",
                    axes: Optional[Sequence[int]] = None,
                    keepdims: bool = True):

    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    assert axes is not None, "Currently only unsqueeze for ONNX < 13 is supported"
    axes = sorted(map(lambda x: x if x >= 0 else t_y.ndim + x, axes))

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    def comp_shape(t_x: StaticType, t_y: StaticType, axes):

        @core.transform
        def f_shape(x_shape, y_shape):
            'nid: V_x_shape'
            core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
            'nid: V_y_shape'
            core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "output", io_mem)

            if t_y.ndim > 0:
                if begin_with_0(axes):
                    if keepdims:
                        y_shape[0] = 1
                        comp_shape(t_x.one_less_dim(), t_y.one_less_dim(),
                                   all_minus_one(axes[1:]))(x_shape[1:],
                                                            y_shape[1:])
                    else:
                        comp_shape(t_x.one_less_dim(), t_y,
                                   all_minus_one(axes[1:]))(x_shape[1:],
                                                            y_shape)
                else:
                    y_shape[0] = x_shape[0]
                    comp_shape(t_x.one_less_dim(), t_y.one_less_dim(),
                               all_minus_one(axes))(x_shape[1:], y_shape[1:])

        return f_shape

    @core.transform
    def f_reduce(x_shape, x):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y_shape'
        y_shape = core.create_var((t_y.ndim,), idx_dtype, "output", io_mem)
        'nid: shape'
        comp_shape(t_x, t_y, axes)(x_shape, y_shape)
        'nid: V_y'
        y = core.create_var(y_shape, t_x.elem_type, "output", io_mem)
        'nid: recur'
        _general_reduce_(t_x, t_y, io_mem, op, neutral_val, idx_dtype, axes,
                         keepdims)(x_shape, y_shape, x, y)
        return y

    return f_reduce


def reduce_sum_(t_x: StaticType,
                t_y: StaticType,
                io_mem,
                idx_dtype="int32",
                axes: Optional[Sequence[int]] = None,
                keepdims: bool = True):
    return _general_reduce_(t_x, t_y, io_mem, lambda x, y: x + y, 0, idx_dtype,
                            axes, keepdims)


def reduce_sum(t_x: StaticType,
               t_y: StaticType,
               io_mem,
               idx_dtype="int32",
               axes: Optional[Sequence[int]] = None,
               keepdims: bool = True):
    return _general_reduce(t_x, t_y, io_mem, lambda x, y: x + y, 0, idx_dtype,
                           axes, keepdims)


def reduce_max_(t_x: StaticType,
                t_y: StaticType,
                io_mem,
                idx_dtype="int32",
                axes: Optional[Sequence[int]] = None,
                keepdims: bool = True):
    return _general_reduce_(t_x, t_y, io_mem, lambda x, y: core.max(x, y),
                            core.min_value(t_x.elem_type), idx_dtype, axes,
                            keepdims)


def reduce_max(t_x: StaticType,
               t_y: StaticType,
               io_mem,
               idx_dtype="int32",
               axes: Optional[Sequence[int]] = None,
               keepdims: bool = True):
    return _general_reduce(t_x, t_y, io_mem, lambda x, y: core.max(x, y),
                           core.min_value(t_x.elem_type), idx_dtype, axes,
                           keepdims)
