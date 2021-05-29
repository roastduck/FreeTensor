from .. import core


def add(a_ndim: int,
        b_ndim: int,
        out_ndim: int,
        io_mem,
        data_dtype="float32",
        idx_dtype="int32"):

    if out_ndim == 0:

        @core.transform
        def f_add(a_shape, b_shape, out_shape, a, b, out):
            'nid: V_a_shape'
            core.declare_var(a_shape, (a_ndim,), idx_dtype, "input", io_mem)
            'nid: V_b_shape'
            core.declare_var(b_shape, (b_ndim,), idx_dtype, "input", io_mem)
            'nid: V_out_shape'
            core.declare_var(out_shape, (out_ndim,), idx_dtype, "output",
                             io_mem)
            'nid: V_a'
            core.declare_var(a, (), data_dtype, "input", io_mem)
            'nid: V_b'
            core.declare_var(b, (), data_dtype, "input", io_mem)
            'nid: V_out'
            core.declare_var(out, (), data_dtype, "output", io_mem)

            out[()] = a[()] + b[()]

    elif a_ndim < out_ndim:
        recur = add(a_ndim, b_ndim - 1, out_ndim - 1, io_mem, data_dtype,
                    idx_dtype)

        @core.transform
        def f_add(a_shape, b_shape, out_shape, a, b, out):
            'nid: V_a_shape'
            core.declare_var(a_shape, (a_ndim,), idx_dtype, "input", io_mem)
            'nid: V_b_shape'
            core.declare_var(b_shape, (b_ndim,), idx_dtype, "input", io_mem)
            'nid: V_out_shape'
            core.declare_var(out_shape, (out_ndim,), idx_dtype, "output",
                             io_mem)
            'nid: V_a'
            core.declare_var(a, a_shape, data_dtype, "input", io_mem)
            'nid: V_b'
            core.declare_var(b, b_shape, data_dtype, "input", io_mem)
            'nid: V_out'
            core.declare_var(out, out_shape, data_dtype, "output", io_mem)

            out_shape[0] = b_shape[0]

            'nid: L_elem'
            for i in range(b_shape[0]):
                'nid: recur'
                recur(a_shape, b_shape[1:], out_shape[1:], a, b[i], out[i])

    elif a_ndim > out_ndim:
        recur = add(a_ndim - 1, b_ndim, out_ndim - 1, io_mem, data_dtype,
                    idx_dtype)

        @core.transform
        def f_add(a_shape, b_shape, out_shape, a, b, out):
            'nid: V_a_shape'
            core.declare_var(a_shape, (a_ndim,), idx_dtype, "input", io_mem)
            'nid: V_b_shape'
            core.declare_var(b_shape, (b_ndim,), idx_dtype, "input", io_mem)
            'nid: V_out_shape'
            core.declare_var(out_shape, (out_ndim,), idx_dtype, "output",
                             io_mem)
            'nid: V_a'
            core.declare_var(a, a_shape, data_dtype, "input", io_mem)
            'nid: V_b'
            core.declare_var(b, b_shape, data_dtype, "input", io_mem)
            'nid: V_out'
            core.declare_var(out, out_shape, data_dtype, "output", io_mem)

            out_shape[0] = a_shape[0]

            'nid: L_elem'
            for i in range(a_shape[0]):
                'nid: recur'
                recur(a_shape[1:], b_shape, out_shape[1:], a[i], b, out[i])

    else:
        recur = add(a_ndim - 1, b_ndim - 1, out_ndim - 1, io_mem, data_dtype,
                    idx_dtype)

        @core.transform
        def f_add(a_shape, b_shape, out_shape, a, b, out):
            'nid: V_a_shape'
            core.declare_var(a_shape, (a_ndim,), idx_dtype, "input", io_mem)
            'nid: V_b_shape'
            core.declare_var(b_shape, (b_ndim,), idx_dtype, "input", io_mem)
            'nid: V_out_shape'
            core.declare_var(out_shape, (out_ndim,), idx_dtype, "output",
                             io_mem)
            'nid: V_a'
            core.declare_var(a, a_shape, data_dtype, "input", io_mem)
            'nid: V_b'
            core.declare_var(b, b_shape, data_dtype, "input", io_mem)
            'nid: V_out'
            core.declare_var(out, out_shape, data_dtype, "output", io_mem)

            out_shape[0] = core.max(a_shape[0], b_shape[0])

            'nid: L_elem'
            for i in range(out_shape[0]):
                if a_shape[0] == 1:
                    'nid: recur_0'
                    recur(a_shape[1:], b_shape[1:], out_shape[1:], a[0], b[i],
                          out[i])
                elif b_shape[0] == 1:
                    'nid: recur_1'
                    recur(a_shape[1:], b_shape[1:], out_shape[1:], a[i], b[0],
                          out[i])
                else:
                    'nid: recur_2'
                    recur(a_shape[1:], b_shape[1:], out_shape[1:], a[i], b[i],
                          out[i])

    return f_add
