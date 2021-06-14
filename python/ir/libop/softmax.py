from .. import core
from .common import StaticType as StaticType
from .element_wise import exp, exp_, sub, sub_, div, div_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


def softmax_(t_x: StaticType,
             t_y: StaticType,
             io_mem,
             idx_dtype="int32",
             axis=-1):

    @core.transform
    def f_softmax(x_shape, y_shape, x, y):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)
        'nid: V_y_shape'
        core.declare_var(y_shape, (t_y.ndim,), idx_dtype, "input", io_mem)
        'nid: V_y'
        core.declare_var(y, y_shape, t_y.elem_type, "output", io_mem)

        'nid: max'
        maxval = reduce_max(t_x,
                            t_x,
                            io_mem,
                            idx_dtype,
                            axes=[axis],
                            keepdims=True)(x_shape, x)
        'nid: sub'
        corrected = sub(t_x, t_x, t_x, io_mem, idx_dtype)(x_shape, maxval.shape,
                                                          x, maxval)
        'nid: exp'
        exponent = exp(t_x, t_x, io_mem, idx_dtype)(corrected.shape, corrected)
        'nid: sum'
        summation = reduce_sum(t_x,
                               t_x,
                               io_mem,
                               idx_dtype,
                               axes=[axis],
                               keepdims=True)(exponent.shape, exponent)
        'nid: div'
        div_(t_x, t_x, t_x, io_mem, idx_dtype)(exponent.shape, summation.shape,
                                               y_shape, exponent, summation, y)

    return f_softmax


def softmax(t_x: StaticType,
            t_y: StaticType,
            io_mem,
            idx_dtype="int32",
            axis=-1):

    @core.transform
    def f_softmax(x_shape, x):
        'nid: V_x_shape'
        core.declare_var(x_shape, (t_x.ndim,), idx_dtype, "input", io_mem)
        'nid: V_x'
        core.declare_var(x, x_shape, t_x.elem_type, "input", io_mem)

        'nid: max'
        maxval = reduce_max(t_x,
                            t_x,
                            io_mem,
                            idx_dtype,
                            axes=[axis],
                            keepdims=True)(x_shape, x)
        'nid: sub'
        corrected = sub(t_x, t_x, t_x, io_mem, idx_dtype)(x_shape, maxval.shape,
                                                          x, maxval)
        'nid: exp'
        exponent = exp(t_x, t_x, io_mem, idx_dtype)(corrected.shape, corrected)
        'nid: sum'
        summation = reduce_sum(t_x,
                               t_x,
                               io_mem,
                               idx_dtype,
                               axes=[axis],
                               keepdims=True)(exponent.shape, exponent)
        'nid: div'
        out = div(t_x, t_x, t_x, io_mem,
                  idx_dtype)(exponent.shape, summation.shape, exponent,
                             summation)
        return out

    return f_softmax
