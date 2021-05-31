from .. import core


def _flatten_inner(ndim: int, io_mem, data_dtype="float32", idx_dtype="int32"):

    assert ndim >= 1
    if ndim == 1:

        @core.transform
        def f_flatten(x_shape, y_shape, x, y):
            'nid: V_x_shape'
            core.declare_var(x_shape, (ndim,), idx_dtype, "input", io_mem)
            'nid: V_y_shape'
            core.declare_var(y_shape, (1,), idx_dtype, "input", io_mem)
            'nid: V_x'
            core.declare_var(x, x_shape, data_dtype, "input", io_mem)
            'nid: V_y'
            core.declare_var(y, y_shape, data_dtype, "output", io_mem)

            # TODO: Assert x_shape[0] == y_shape[0]

            'nid: L_inner'
            for i in range(x_shape[0]):
                y[i] = x[i]

    else:

        recur = _flatten_inner(ndim - 1, io_mem, data_dtype, idx_dtype)

        @core.transform
        def f_flatten(x_shape, y_shape, x, y):
            'nid: V_x_shape'
            core.declare_var(x_shape, (ndim,), idx_dtype, "input", io_mem)
            'nid: V_y_shape'
            core.declare_var(y_shape, (1,), idx_dtype, "input", io_mem)
            'nid: V_x'
            core.declare_var(x, x_shape, data_dtype, "input", io_mem)
            'nid: V_y'
            core.declare_var(y, y_shape, data_dtype, "output", io_mem)

            'nid: L_inner'
            for i in range(x_shape[0]):
                'nid: V_recur_y_shape'
                recur_y_shape = core.create_var((1,), idx_dtype, "cache",
                                                io_mem)
                recur_y_shape[0] = y_shape[0] // x_shape[0]
                'nid: recur'
                recur(x_shape[1:], recur_y_shape, x[i],
                      y[i * recur_y_shape[0]:(i + 1) * recur_y_shape[0]])

    return f_flatten


def flatten(x_ndim: int, io_mem, data_dtype="float32", idx_dtype="int32"):

    recur = _flatten_inner(x_ndim - 1, io_mem, data_dtype, idx_dtype)

    @core.transform
    def f_flatten(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (x_ndim,), idx_dtype, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (2,), idx_dtype, "output", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, data_dtype, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, data_dtype, "output", io_mem)

        y_shape[0] = x_shape[0]
        y_shape[1] = 1
        'nid: L_shape'
        for i in range(1, x_ndim):
            y_shape[1] *= x_shape[i]

        'nid: L_outer'
        for i in range(x_shape[0]):
            'nid: recur'
            recur(x_shape[1:], y_shape[1:], x[i], y[i])

    return f_flatten
