from .. import core
from .shape_utils import *
from .common import StaticType


def _binary_op_(t_a: StaticType,
                t_b: StaticType,
                t_out: StaticType,
                io_mem,
                op,
                idx_dtype="int32"):

    @core.transform
    def f_binary_op(a_shape, b_shape, out_shape, a, b, out):
        'nid: V_a_shape'
        core.declare_var(a_shape, (t_a.ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (t_b.ndim,), idx_dtype, "input", io_mem)
        'nid: V_out_shape'
        core.declare_var(out_shape, (t_out.ndim,), idx_dtype, "input", io_mem)
        'nid: V_a'
        core.declare_var(a, a_shape, t_a.elem_type, "input", io_mem)
        'nid: V_b'
        core.declare_var(b, b_shape, t_b.elem_type, "input", io_mem)
        'nid: V_out'
        core.declare_var(out, out_shape, t_out.elem_type, "output", io_mem)

        if t_out.ndim == 0:
            out[()] = op(a[()], b[()])
        else:
            'nid: L_elem'
            for i in range(out_shape[0]):
                if t_a.ndim < t_out.ndim:
                    'nid: recur'
                    add_(t_a, t_b.one_less_dim(), t_out.one_less_dim(), io_mem,
                         idx_dtype)(a_shape, b_shape[1:], out_shape[1:], a,
                                    b[i], out[i])
                elif t_b.ndim < t_out.ndim:
                    'nid: recur'
                    add_(t_a.one_less_dim(), t_b, t_out.one_less_dim(), io_mem,
                         idx_dtype)(a_shape[1:], b_shape, out_shape[1:], a[i],
                                    b, out[i])
                else:
                    'nid: recur'
                    add_(t_a.one_less_dim(), t_b.one_less_dim(),
                         t_out.one_less_dim(), io_mem,
                         idx_dtype)(a_shape[1:], b_shape[1:], out_shape[1:],
                                    a[i % a_shape[0]], b[i % b_shape[0]],
                                    out[i])

    return f_binary_op


def _binary_op(t_a: StaticType,
               t_b: StaticType,
               t_out: StaticType,
               io_mem,
               op,
               idx_dtype="int32"):

    @core.transform
    def f_binary_op(a_shape, b_shape, a, b):
        'nid: V_a_shape'
        core.declare_var(a_shape, (t_a.ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (t_b.ndim,), idx_dtype, "input", io_mem)
        'nid: V_a'
        core.declare_var(a, a_shape, t_a.elem_type, "input", io_mem)
        'nid: V_b'
        core.declare_var(b, b_shape, t_b.elem_type, "input", io_mem)
        'nid: broadcast_shape'
        out_shape = broadcast_shape(t_a, t_b, t_out, io_mem, idx_dtype)(a_shape,
                                                                        b_shape)
        'nid: V_out'
        out = core.create_var(out_shape, t_out.elem_type, "output", io_mem)
        'nid: recur'
        _binary_op_(t_a, t_b, t_out, io_mem, op,
                    idx_dtype)(a_shape, b_shape, out_shape, a, b, out)
        return out_shape, out

    return f_binary_op


def add_(t_a: StaticType,
         t_b: StaticType,
         t_out: StaticType,
         io_mem,
         idx_dtype="int32"):
    return _binary_op_(t_a, t_b, t_out, io_mem, lambda x, y: x + y, idx_dtype)


def add(t_a: StaticType,
        t_b: StaticType,
        t_out: StaticType,
        io_mem,
        idx_dtype="int32"):
    return _binary_op(t_a, t_b, t_out, io_mem, lambda x, y: x + y, idx_dtype)


def sub_(t_a: StaticType,
         t_b: StaticType,
         t_out: StaticType,
         io_mem,
         idx_dtype="int32"):
    return _binary_op_(t_a, t_b, t_out, io_mem, lambda x, y: x - y, idx_dtype)


def sub(t_a: StaticType,
        t_b: StaticType,
        t_out: StaticType,
        io_mem,
        idx_dtype="int32"):
    return _binary_op(t_a, t_b, t_out, io_mem, lambda x, y: x - y, idx_dtype)


def mul_(t_a: StaticType,
         t_b: StaticType,
         t_out: StaticType,
         io_mem,
         idx_dtype="int32"):
    return _binary_op_(t_a, t_b, t_out, io_mem, lambda x, y: x * y, idx_dtype)


def mul(t_a: StaticType,
        t_b: StaticType,
        t_out: StaticType,
        io_mem,
        idx_dtype="int32"):
    return _binary_op(t_a, t_b, t_out, io_mem, lambda x, y: x * y, idx_dtype)


def div_(t_a: StaticType,
         t_b: StaticType,
         t_out: StaticType,
         io_mem,
         idx_dtype="int32"):
    return _binary_op_(t_a, t_b, t_out, io_mem, lambda x, y: x / y, idx_dtype)


def div(t_a: StaticType,
        t_b: StaticType,
        t_out: StaticType,
        io_mem,
        idx_dtype="int32"):
    return _binary_op(t_a, t_b, t_out, io_mem, lambda x, y: x / y, idx_dtype)


def relu_(t_x: StaticType, t_y: StaticType, io_mem, idx_dtype="int32"):

    @core.transform
    def f_relu(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        if t_x.ndim == 0:
            if x[()] > 0:
                y[()] = x[()]
            else:
                y[()] = 0
        else:
            'nid: L_elem'
            for i in range(x_shape[0]):
                'nid: recur'
                relu_(t_x.one_less_dim(), t_y.one_less_dim(), io_mem,
                      idx_dtype)(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_relu


def relu(t_x: StaticType, t_y: StaticType, io_mem, idx_dtype="int32"):

    @core.transform
    def f_relu(x_shape, x):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: copy_shape'
        y_shape = copy_shape(t_x, t_y, io_mem, idx_dtype)(x_shape)
        'nid: V_y'
        y = core.create_var(y_shape, t_y.elem_type, "output", io_mem)
        'nid: recur'
        relu_(t_x, t_y, io_mem, idx_dtype)(x_shape, y_shape, x, y)
        return y_shape, y

    return f_relu


def sqrt_(t_x: StaticType, t_y: StaticType, io_mem, idx_dtype="int32"):

    @core.transform
    def f_sqrt(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_x.elem_type, "output", io_mem)

        if t_x.ndim == 0:
            y[()] = core.sqrt(x[()])
        else:
            'nid: L_elem'
            for i in range(x_shape[0]):
                'nid: recur'
                sqrt_(t_x.one_less_dim(), t_y.one_less_dim(), io_mem,
                      idx_dtype)(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_sqrt


def sqrt(t_x: StaticType, t_y: StaticType, io_mem, idx_dtype="int32"):

    @core.transform
    def f_sqrt(x_shape, x):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: copy_shape'
        y_shape = copy_shape(t_x, t_y, io_mem, idx_dtype)(x_shape)
        'nid: V_y'
        y = core.create_var(y_shape, t_y.elem_type, "output", io_mem)
        'nid: recur'
        sqrt_(t_x, t_y, io_mem, idx_dtype)(x_shape, y_shape, x, y)
        return y_shape, y

    return f_sqrt
