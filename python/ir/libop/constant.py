from .. import core
from .assign import assign


def zeros_():

    @core.inline
    def f_zeros(y):
        if y.ndim == 0:
            y[()] = 0
        else:
            'nid: L_elem'
            for i in range(y.shape(0)):
                'nid: recur'
                zeros_()(y[i])

    return f_zeros


def zeros(shape, dtype, mtype):

    @core.inline
    def f_zeros():
        y = core.create_var(shape, dtype, "cache", mtype)
        'nid: recur'
        zeros_()(y)
        return y

    return f_zeros
