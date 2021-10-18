from .. import core


def broadcast_shape(a, b):
    out_shape = []
    out_ndim = max(a.ndim, b.ndim)
    for i in range(out_ndim):
        if i - out_ndim + a.ndim < 0:
            out_shape.append(b.shape(i))
        elif i - out_ndim + b.ndim < 0:
            out_shape.append(a.shape(i))
        else:
            out_shape.append(
                core.max(a.shape(i - out_ndim + a.ndim),
                         b.shape(i - out_ndim + b.ndim)))
    return out_shape


def copy_shape(a):
    shape = []
    for i in range(a.ndim):
        shape.append(a.shape(i))
    return shape
