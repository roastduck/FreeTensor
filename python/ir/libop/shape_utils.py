from .. import core
from .common import StaticType as T


def broadcast_shape_(t_a: T, t_b: T, t_out: T, io_mem, idx_dtype="int32"):

    @core.inline
    def f_broadcast_shape(a_shape, b_shape, out_shape):
        'nid: V_a_shape'
        core.declare_var(a_shape, (t_a.ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (t_b.ndim,), idx_dtype, "input", io_mem)
        'nid: V_out_shape'
        core.declare_var(out_shape, (t_out.ndim,), idx_dtype, "output", io_mem)

        if t_a.ndim < t_b.ndim:
            out_shape[0] = b_shape[0]
            broadcast_shape_(t_a, t_b.one_less_dim(), t_out.one_less_dim(),
                             io_mem, idx_dtype)(a_shape, b_shape[1:],
                                                out_shape[1:])
        elif t_a.ndim > t_b.ndim:
            out_shape[0] = a_shape[0]
            broadcast_shape_(t_a.one_less_dim(), t_b, t_out.one_less_dim(),
                             io_mem, idx_dtype)(a_shape[1:], b_shape,
                                                out_shape[1:])
        elif t_out.ndim > 0:
            out_shape[0] = core.max(a_shape[0], b_shape[0])
            broadcast_shape_(t_a.one_less_dim(), t_b.one_less_dim(),
                             t_out.one_less_dim(), io_mem,
                             idx_dtype)(a_shape[1:], b_shape[1:], out_shape[1:])

    return f_broadcast_shape


def broadcast_shape(t_a: T, t_b: T, t_out: T, io_mem, idx_dtype="int32"):

    @core.inline
    def f_broadcast_shape(a_shape, b_shape):
        'nid: V_a_shape'
        core.declare_var(a_shape, (t_a.ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (t_b.ndim,), idx_dtype, "input", io_mem)
        'nid: V_out_shape'
        out_shape = core.create_var((t_out.ndim,), idx_dtype, "output", io_mem)
        'nid: recur'
        broadcast_shape_(t_a, t_b, t_out, io_mem, idx_dtype)(a_shape, b_shape,
                                                             out_shape)
        return out_shape

    return f_broadcast_shape
