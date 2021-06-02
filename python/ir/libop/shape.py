from .. import core
from .common import StaticType as T


def _flatten_inner(t_x: T, io_mem, idx_dtype="int32"):

    assert t_x.ndim >= 1

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

        'nid: L_inner'
        for i in range(x_shape[0]):
            if t_x.ndim == 1:
                y[i] = x[i]
            else:
                'nid: V_recur_y_shape'
                recur_y_shape = core.create_var((1,), idx_dtype, "cache",
                                                io_mem)
                recur_y_shape[0] = y_shape[0] // x_shape[0]
                'nid: recur'
                _flatten_inner(T(t_x.elem_type, t_x.ndim - 1), io_mem,
                               idx_dtype)(x_shape[1:], recur_y_shape, x[i],
                                          y[i * recur_y_shape[0]:(i + 1) *
                                            recur_y_shape[0]])

    return f_flatten


def flatten(t_x: T, io_mem, idx_dtype="int32"):

    recur = _flatten_inner(T(t_x.elem_type, t_x.ndim - 1), io_mem, idx_dtype)

    @core.transform
    def f_flatten(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (2,), idx_dtype, "output", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        y_shape[0] = x_shape[0]
        y_shape[1] = 1
        'nid: L_shape'
        for i in range(1, t_x.ndim):
            y_shape[1] *= x_shape[i]

        'nid: L_outer'
        for i in range(x_shape[0]):
            'nid: recur'
            recur(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_flatten
