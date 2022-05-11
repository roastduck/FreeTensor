from .. import core
from .utils import *
from .shape_utils import *


@core.inline
def _assign_op(op, y, x):
    if core.ndim(y) == 0:
        y[()] = op(y, x)
    else:
        'nid: L_elem'
        for i in range(core.shape(y, 0)):
            if core.ndim(x) < core.ndim(y):
                'nid: recur'
                _assign_op(op, y[i], x)
            else:
                assert core.shape(x, 0) == core.shape(y, 0) or core.shape(
                    x, 0) == 1
                'nid: recur'
                _assign_op(op, y[i], x[i % x.shape(0)])


assign = named_partial("assign", _assign_op, lambda y, x: x)
add_to = named_partial("add_to", _assign_op, lambda y, x: y + x)
sub_to = named_partial("sub_to", _assign_op, lambda y, x: y - x)
mul_to = named_partial("mul_to", _assign_op, lambda y, x: y * x)
truediv_to = named_partial("truediv_to", _assign_op, lambda y, x: y / x)
floordiv_to = named_partial("floordiv_to", _assign_op, lambda y, x: y // x)
mod_to = named_partial("mod_to", _assign_op, lambda y, x: y % x)
