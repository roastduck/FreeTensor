import builtins

from .. import core


def broadcast_shape_():

    @core.inline
    def f_broadcast_shape(a, b, out_shape):
        if a.ndim < b.ndim:
            out_shape[0] = b.shape(0)
            broadcast_shape_()(a, b[0], out_shape[1:])
        elif a.ndim > b.ndim:
            out_shape[0] = a.shape(0)
            broadcast_shape_(a_ndim - 1, b_ndim, out_ndim - 1,
                             io_mem)(a[0], b, out_shape[1:])
        elif a.ndim > 0:
            out_shape[0] = core.max(a.shape(0), b.shape(0))
            broadcast_shape_()(a[0], b[0], out_shape[1:])

    return f_broadcast_shape


def broadcast_shape(io_mem, idx_dtype="int32"):

    @core.inline
    def f_broadcast_shape(a, b):
        out_shape = core.create_var((builtins.max(a.ndim, b.ndim),), idx_dtype,
                                    "output", io_mem)
        'nid: recur'
        broadcast_shape_()(a, b, out_shape)
        return out_shape

    return f_broadcast_shape


def copy_shape_():

    @core.inline
    def f_copy_shape(x, y_shape):
        if x.ndim > 0:
            y_shape[0] = x.shape(0)
            copy_shape_()(x[0], y_shape[1:])

    return f_copy_shape


def copy_shape(io_mem, idx_dtype="int32"):

    @core.inline
    def f_copy_shape(x):
        y_shape = core.create_var((x.ndim,), idx_dtype, "output", io_mem)
        'nid: recur'
        copy_shape_()(x, y_shape)
        return y_shape

    return f_copy_shape
