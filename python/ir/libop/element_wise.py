from .. import core


def add(a_ndim: int,
        b_ndim: int,
        out_ndim: int,
        io_mem,
        data_dtype="float32",
        idx_dtype="int32"):

    @core.transform
    def f_add(a_shape, b_shape, out_shape, a, b, out):
        'nid: V_a_shape'
        core.declare_var(a_shape, (a_ndim,), idx_dtype, "input", io_mem)
        'nid: V_b_shape'
        core.declare_var(b_shape, (b_ndim,), idx_dtype, "input", io_mem)
        'nid: V_out_shape'
        core.declare_var(out_shape, (out_ndim,), idx_dtype, "output", io_mem)
        'nid: V_a'
        core.declare_var(a, a_shape, data_dtype, "input", io_mem)
        'nid: V_b'
        core.declare_var(b, b_shape, data_dtype, "input", io_mem)
        'nid: V_out'
        core.declare_var(out, out_shape, data_dtype, "output", io_mem)

        if out_ndim == 0:
            out[()] = a[()] + b[()]
        else:
            out_shape[0] = core.max(a_shape[0], b_shape[0])

            'nid: L_elem'
            for i in range(out_shape[0]):
                if a_ndim < out_ndim:
                    'nid: recur'
                    add(a_ndim, b_ndim - 1, out_ndim - 1, io_mem, data_dtype,
                        idx_dtype)(a_shape, b_shape[1:], out_shape[1:], a, b[i],
                                   out[i])
                elif b_ndim < out_ndim:
                    'nid: recur'
                    add(a_ndim - 1, b_ndim, out_ndim - 1, io_mem, data_dtype,
                        idx_dtype)(a_shape[1:], b_shape, out_shape[1:], a[i], b,
                                   out[i])
                else:
                    'nid: recur'
                    add(a_ndim - 1, b_ndim - 1, out_ndim - 1, io_mem,
                        data_dtype, idx_dtype)(a_shape[1:], b_shape[1:],
                                               out_shape[1:], a[i % a_shape[0]],
                                               b[i % b_shape[0]], out[i])

    return f_add


def relu(ndim: int, io_mem, data_dtype="float32", idx_dtype="int32"):

    @core.transform
    def f_relu(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (ndim,), idx_dtype, "output", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, data_dtype, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, data_dtype, "output", io_mem)

        if ndim == 0:
            if x[()] > 0:
                y[()] = x[()]
            else:
                y[()] = 0
        else:
            y_shape[0] = x_shape[0]

            'nid: L_elem'
            for i in range(x_shape[0]):
                'nid: recur'
                relu(ndim - 1, io_mem, data_dtype,
                     idx_dtype)(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_relu
