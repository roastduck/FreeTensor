from .. import core


def broadcast_shape(a, b):
    out_shape = []
    out_ndim = max(core.ndim(a), core.ndim(b))
    for i in range(out_ndim):
        if i - out_ndim + core.ndim(a) < 0:
            out_shape.append(core.shape(b, i))
        elif i - out_ndim + core.ndim(b) < 0:
            out_shape.append(core.shape(a, i))
        else:
            out_shape.append(
                core.max(core.shape(a, i - out_ndim + core.ndim(a)),
                         core.shape(b, i - out_ndim + core.ndim(b))))
    return out_shape


def copy_shape(a):
    shape = []
    for i in range(core.ndim(a)):
        shape.append(core.shape(a, i))
    return shape
