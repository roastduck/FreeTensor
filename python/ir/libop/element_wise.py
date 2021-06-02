from .. import core
from .common import StaticType as T


def add(t_a: T, t_b: T, t_out: T, io_mem, idx_dtype="int32"):

    @core.transform
    def f_add(a_shape, b_shape, out_shape, a, b, out):
        'nid: V_a_shape'
        core.declare_var(a_shape, (t_a.ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (t_b.ndim,), idx_dtype, "input", io_mem)
        'nid: V_out_shape'
        core.declare_var(out_shape, (t_out.ndim,), idx_dtype, "output", io_mem)
        'nid: V_a'
        core.declare_var(a, a_shape, t_a.elem_type, "input", io_mem)
        'nid: V_b'
        core.declare_var(b, b_shape, t_b.elem_type, "input", io_mem)
        'nid: V_out'
        core.declare_var(out, out_shape, t_out.elem_type, "output", io_mem)

        if t_out.ndim == 0:
            out[()] = a[()] + b[()]
        else:
            out_shape[0] = core.max(a_shape[0], b_shape[0])

            'nid: L_elem'
            for i in range(out_shape[0]):
                if t_a.ndim < t_out.ndim:
                    'nid: recur'
                    add(T(t_a.elem_type, t_a.ndim),
                        T(t_b.elem_type, t_b.ndim - 1),
                        T(t_out.elem_type, t_out.ndim - 1), io_mem,
                        idx_dtype)(a_shape, b_shape[1:], out_shape[1:], a, b[i],
                                   out[i])
                elif t_b.ndim < t_out.ndim:
                    'nid: recur'
                    add(T(t_a.elem_type, t_a.ndim - 1),
                        T(t_b.elem_type, t_b.ndim),
                        T(t_out.elem_type, t_out.ndim - 1), io_mem,
                        idx_dtype)(a_shape[1:], b_shape, out_shape[1:], a[i], b,
                                   out[i])
                else:
                    'nid: recur'
                    add(T(t_a.elem_type, t_a.ndim - 1),
                        T(t_b.elem_type, t_b.ndim - 1),
                        T(t_out.elem_type, t_out.ndim - 1), io_mem,
                        idx_dtype)(a_shape[1:], b_shape[1:], out_shape[1:],
                                   a[i % a_shape[0]], b[i % b_shape[0]], out[i])

    return f_add


def relu(t_x: T, io_mem, idx_dtype="int32"):

    @core.transform
    def f_relu(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_x.ndim,), idx_dtype, "output", io_mem)
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
            y_shape[0] = x_shape[0]

            'nid: L_elem'
            for i in range(x_shape[0]):
                'nid: recur'
                relu(T(t_x.elem_type, t_x.ndim - 1), io_mem,
                     idx_dtype)(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_relu
