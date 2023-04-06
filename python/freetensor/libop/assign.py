__all__ = [
    'assign', 'add_to', 'sub_to', 'mul_to', 'truediv_to', 'floordiv_to',
    'mod_to'
]

from .. import core
from .utils import *
from .shape_utils import *


def _named_partial(name: str, doc: str, f, *args, **kvs):
    ''' Similar to functools.partial, but it sets the returned function's __name__ and __doc__ '''

    # This function should be defined in the same file that uses it
    # https://github.com/mkdocstrings/pytkdocs/issues/143

    def g(*_args, **_kvs):
        return f(*args, *_args, **kvs, **_kvs)

    g.__name__ = name
    g.__doc__ = doc
    return g


@core.inline
def _assign_op(op, y, x):
    if core.ndim(y) == 0:
        y[()] = op(y, x)
    else:
        #! label: L_elem
        for i in range(core.shape(y, 0)):
            if core.ndim(x) < core.ndim(y):
                #! label: recur
                _assign_op(op, y[i], x)
            else:
                assert core.shape(x, 0) == core.shape(y, 0) or core.shape(
                    x, 0) == 1
                #! label: recur
                _assign_op(op, y[i], x[i % x.shape(0)])


assign_doc_template = '''
(Broadcasted) {} a tensor two another tensor

Parameters
----------
y : VarRef
    The target tensor
x : VarRef
    The source tensor
'''

assign = _named_partial("assign", assign_doc_template.format('assign to'),
                        _assign_op, lambda y, x: x)
add_to = _named_partial("add_to", assign_doc_template.format('add to'),
                        _assign_op, lambda y, x: y + x)
sub_to = _named_partial("sub_to", assign_doc_template.format('subtract from'),
                        _assign_op, lambda y, x: y - x)
mul_to = _named_partial("mul_to", assign_doc_template.format('multiply to'),
                        _assign_op, lambda y, x: y * x)
truediv_to = _named_partial(
    "truediv_to", assign_doc_template.format('floating-point division from'),
    _assign_op, lambda y, x: y / x)
floordiv_to = _named_partial(
    "floordiv_to",
    assign_doc_template.format(
        'rounding-towards-negative-infinity integer division (following Python convention, but not C) from'
    ), _assign_op, lambda y, x: y // x)
mod_to = _named_partial(
    "mod_to",
    assign_doc_template.format(
        'modulo (results are non-negative, following Python convention, but not C) from'
    ), _assign_op, lambda y, x: y % x)
