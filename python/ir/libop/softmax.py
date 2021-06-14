from .. import core
from .common import StaticType as StaticType
from .element_wise import exp, exp_, sub, sub_, div, div_
from .reduction import reduce_max, reduce_max_, reduce_sum, reduce_sum_


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

        maxval = reduce_max(t_x,
                            t_x,
                            io_mem,
                            idx_dtype,
                            axes=[axis],
                            keepdims=True)(x_shape, x)
        corrected = sub(t_x, t_x, t_x, io_mem, idx_dtype)(x_shape, maxval.shape,
                                                          x, maxval)
        exponent = exp(t_x, t_x, io_mem, idx_dtype)(corrected.shape, corrected)
        summation = reduce_sum(t_x,
                               t_x,
                               io_mem,
                               idx_dtype,
                               axes=[axis],
                               keepdims=True)(exponent.shape, exponent)
        out = div(t_x, t_x, t_x, io_mem,
                  idx_dtype)(exponent.shape, summation.shape, exponent,
                             summation)
        return out

    return f_softmax
