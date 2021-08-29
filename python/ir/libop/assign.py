from .. import core
from .shape_utils import *
from .common import StaticType


def _assign_op(t_y: StaticType, t_x: StaticType, io_mem, op, idx_dtype="int32"):

    @core.transform
    def f_binary_op(y_shape, x_shape, y, x):
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_y.elem_type, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)

        if t_y.ndim == 0:
            y[()] = op(y[()], x[()])
        else:
            'nid: L_elem'
            for i in range(y_shape[0]):
                if t_x.ndim < t_y.ndim:
                    'nid: recur'
                    _assign_op(t_y.one_less_dim(), t_x, io_mem, op,
                               idx_dtype)(y_shape[1:], x_shape, y[i], x)
                else:
                    'nid: recur'
                    _assign_op(t_y.one_less_dim(), t_x.one_less_dim(), io_mem,
                               op, idx_dtype)(y_shape[1:], x_shape[1:], y[i],
                                              x[i % x_shape[0]])

    return f_binary_op


def assign(t_y: StaticType, t_x: StaticType, io_mem, idx_dtype="int32"):
    return _assign_op(t_y, t_x, io_mem, lambda y, x: x, idx_dtype)


def add_to(t_y: StaticType, t_x: StaticType, io_mem, idx_dtype="int32"):
    return _assign_op(t_y, t_x, io_mem, lambda y, x: x + y, idx_dtype)
