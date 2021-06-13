from typing import Sequence, Optional

from .. import core
from .common import StaticType as StaticType


def _flatten_inner_(t_x: StaticType,
                    t_y: StaticType,
                    io_mem,
                    idx_dtype="int32"):

    @core.transform
    def f_flatten(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (1,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        if t_x.ndim == 0:
            y[0] = x[()]
        else:
            'nid: L_inner'
            for i in range(x_shape[0]):
                'nid: V_recur_y_shape'
                recur_y_shape = core.create_var((1,), idx_dtype, "cache",
                                                io_mem)
                recur_y_shape[0] = y_shape[0] // x_shape[0]
                'nid: recur'
                _flatten_inner_(t_x.one_less_dim(), t_y, io_mem, idx_dtype)(
                    x_shape[1:], recur_y_shape, x[i],
                    y[i * recur_y_shape[0]:(i + 1) * recur_y_shape[0]])

    return f_flatten


def flatten_(t_x: StaticType,
             t_y: StaticType,
             io_mem,
             idx_dtype="int32",
             axis=1):

    @core.transform
    def f_flatten(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (2,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        if axis == 0:
            'nid: recur'
            _flatten_inner_(t_x, t_y.one_less_dim(), io_mem,
                            idx_dtype)(x_shape, y_shape[1:], x, y[0])
        else:
            'nid: V_recur_y_shape'
            recur_y_shape = core.create_var((2,), idx_dtype, "cache", io_mem)
            recur_y_shape[0] = y_shape[0] // x_shape[0]
            recur_y_shape[1] = y_shape[1]
            'nid: L_outer'
            for i in range(x_shape[0]):
                'nid: recur'
                flatten_(t_x.one_less_dim(), t_y, io_mem, idx_dtype, axis -
                         1)(x_shape[1:], recur_y_shape, x[i],
                            y[i * recur_y_shape[0]:(i + 1) * recur_y_shape[0]])

    return f_flatten


def flatten(t_x: StaticType,
            t_y: StaticType,
            io_mem,
            idx_dtype="int32",
            axis=1):

    @core.transform
    def f_flatten(x_shape, x):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y_shape'
        y_shape = core.create_var((2,), idx_dtype, "output", io_mem)
        y_shape[0] = 1
        'nid: L_shape_0'
        for i in range(axis):
            y_shape[0] *= x_shape[i]
        y_shape[1] = 1
        'nid: L_shape_1'
        for i in range(axis, t_x.ndim):
            y_shape[1] *= x_shape[i]
        'nid: V_y'
        y = core.create_var(y_shape, t_x.elem_type, "output", io_mem)
        'nid: recur'
        flatten_(t_x, t_y, io_mem, idx_dtype, axis)(x_shape, y_shape, x, y)
        return y_shape, y

    return f_flatten


def unsqueeze_(t_x: StaticType,
               t_y: StaticType,
               io_mem,
               idx_dtype="int32",
               axes: Optional[Sequence[int]] = None):

    # ONNX >= 13 treats axes as a tensor, which we don't support for now
    assert axes is not None, "Currently only unsqueeze for ONNX < 13 is supported"
    axes = sorted(map(lambda x: x if x >= 0 else t_y.ndim + x, axes))

    def begin_with_0(lst):
        return len(lst) > 0 and lst[0] == 0

    def all_minus_one(lst):
        return list(map(lambda x: x - 1, lst))

    @core.transform
    def f_unsqueeze(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        if t_y.ndim == 0:
            y[()] = x[()]
        elif begin_with_0(axes):
            'nid: recur'
            unsqueeze_(t_x, t_y.one_less_dim(), io_mem, idx_dtype,
                       all_minus_one(axes[1:]))(x_shape, y_shape[1:], x, y[0])
        else:
            'nid: L'
            for i in range(x_shape[0]):
                'nid: recur'
                unsqueeze_(t_x.one_less_dim(),
                           t_y.one_less_dim(), io_mem, idx_dtype,
                           all_minus_one(axes))(x_shape[1:], y_shape[1:], x[i],
                                                y[i])

    return f_unsqueeze


def unsqueeze(t_x: StaticType,
              t_y: StaticType,
              io_mem,
              idx_dtype="int32",
              axes: Optional[Sequence[int]] = None):

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
                    y_shape[0] = 1
                    comp_shape(t_x, t_y.one_less_dim(),
                               all_minus_one(axes[1:]))(x_shape, y_shape[1:])
                else:
                    y_shape[0] = x_shape[0]
                    comp_shape(t_x.one_less_dim(), t_y.one_less_dim(),
                               all_minus_one(axes))(x_shape[1:], y_shape[1:])

        return f_shape

    @core.transform
    def f_unsqueeze(x_shape, x):
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
        unsqueeze_(t_x, t_y, io_mem, idx_dtype, axes)(x_shape, y_shape, x, y)
        return y_shape, y

    return f_unsqueeze
