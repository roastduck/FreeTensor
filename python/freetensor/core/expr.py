'''
Facility to build AST expressions

Classes and functions in this module are not only used internally for constructing AST nodes,
and also exposed to users via multi-stage programming
'''

import collections
import builtins
import math
from numbers import Number
from typing import Sequence

import freetensor_ffi as ffi

from .context import ctx_stack


class VarRef(ffi.FrontendVar):
    '''
    Variable of FreeTensor

    All variables in FreeTensor DSL (declared via `Var`, created by `empty` or `var`,
    returned by `libop`, etc.), and their slices, are `VarRef` objects. Operations
    on `VarRef` objects generates AST nodes
    '''

    def __init__(self,
                 name: str,
                 vardef,
                 full_shape: Sequence,
                 dtype: ffi.DataType,
                 mtype: ffi.MemType,
                 indices: Sequence = []):
        super(VarRef, self).__init__(name, full_shape, dtype, mtype, indices)
        self.vardef = vardef

        from .stmt import find_borrowed_vardefs
        self.borrowed_vardefs = find_borrowed_vardefs(indices)
        for item in self.borrowed_vardefs:
            item.lend_out()

    def __del__(self):
        for item in self.borrowed_vardefs:
            item.reclaim()

    def __getitem__(self, key):
        return VarRef(self.name, self.vardef, self.full_shape, self.dtype,
                      self.mtype, self.chain_indices(self._parse_key(key)))

    def __setitem__(self, key, value):
        var = VarRef(self.name, self.vardef, self.full_shape, self.dtype,
                     self.mtype, self.chain_indices(self._parse_key(key)))
        if var.ndim > 0:
            if value is not None:
                # In standard Python data model, functions like __iadd__
                # returns the modified self, and __setitem__ does a self-
                # assignment. We do the augmenting assignment directly
                # in __iadd__ and return None, so we do not have to do
                # it again here
                from .. import libop
                libop.assign(var, value)
            return
        if var.vardef.atype == ffi.AccessType("input"):
            raise ffi.InvalidProgram("Cannot modify an \"input\" tensor `" +
                                     self.name + "`")
        if var.vardef.borrower_cnt > 0:
            raise ffi.InvalidProgram(
                "Cannot modify tensor `" + self.name +
                "` becuase it has been borrowed in another tensor's shape, "
                "a tensor slice, or a range of a loop")
        top = ctx_stack.top()
        top.append_stmt(var.as_store(top.get_metadata(), value))

    def select(self, idx, dim):
        assert isinstance(dim, int)
        assert dim >= 0 and dim < self.ndim
        indices = [
            slice(None, None) if d != dim else idx for d in range(self.ndim)
        ]
        return self[indices]

    def shape(self, dim=None):
        '''
        Return lengths of all dimensions or the length of one dimension

        `.shape()` -> list of lengths of all dimensions

        `.shape(dim)` -> length of dimension `dim`, where `dim` can be `int`
        or `Expr`

        All lengths can be `Expr` (if the length is dynamically decided) or
        `int` (if statically decided)
        '''
        intOrExpr = lambda x: x.val if isinstance(x, ffi.IntConst) else x
        if dim is None:
            return [intOrExpr(d) for d in super(VarRef, self).shape()]
        else:
            return intOrExpr(super(VarRef, self).shape(dim))

    def _parse_key(self, key):
        if key is None or key is ...:
            key = ()
        if not isinstance(key, collections.abc.Sequence):
            key = (key,)
        ffiIdx = []
        for idx, length in zip(key, self.shape()):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else length
                assert idx.step is None or idx.step == 1
                ffiIdx.append(ffi.FrontendVarIdx(start, stop))
            elif isinstance(idx, VarRef):
                if len(idx.full_shape) == len(idx.indices):
                    ffiIdx.append(ffi.FrontendVarIdx(idx.as_load()))
                else:
                    assert len(
                        key
                    ) == 1, f"Shape of an index of {self.name} should be 1-D, instead of {idx.name}"
                    assert type(
                        idx.full_shape[0]
                    ) is ffi.IntConst, "Dynamic number of dimensions is not supported"
                    ndim = idx.full_shape[0].val
                    ffiIdx += [
                        ffi.FrontendVarIdx(idx[i].as_load())
                        for i in range(ndim)
                    ]
            else:
                ffiIdx.append(ffi.FrontendVarIdx(idx))
        return ffiIdx

    def __add__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.add(self, other)
        return self.as_load() + other

    def __radd__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.add(other, self)
        return other + self.as_load()

    def __iadd__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.add_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __sub__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.sub(self, other)
        return self.as_load() - other

    def __rsub__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.sub(other, self)
        return other - self.as_load()

    def __isub__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.sub_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __mul__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.mul(self, other)
        return self.as_load() * other

    def __rmul__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.mul(other, self)
        return other * self.as_load()

    def __imul__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.mul_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __truediv__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.truediv(self, other)
        return self.as_load() / other

    def __rtruediv__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.truediv(other, self)
        return other / self.as_load()

    def __itruediv__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.truediv_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __floordiv__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.floordiv(self, other)
        return self.as_load() // other

    def __rfloordiv__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.floordiv(other, self)
        return other // self.as_load()

    def __ifloordiv__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.floordiv_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __mod__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.mod(self, other)
        return self.as_load() % other

    def __rmod__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.mod(other, self)
        return other % self.as_load()

    def __imod__(self, other):
        if self.ndim > 0:
            from .. import libop
            libop.mod_to(self, other)
            return  # Don't return self. See __setitem__
        return NotImplemented

    def __lt__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.lt(self, other)
        return self.as_load() < other

    def __le__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.le(self, other)
        return self.as_load() <= other

    def __gt__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.gt(self, other)
        return self.as_load() > other

    def __ge__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.ge(self, other)
        return self.as_load() >= other

    def __eq__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.eq(self, other)
        return self.as_load() == other

    def __ne__(self, other):
        if self.ndim > 0:
            from .. import libop
            return libop.ne(self, other)
        return self.as_load() != other

    def __neg__(self):
        if self.ndim > 0:
            from .. import libop
            return libop.neg(self)
        return 0 - self.as_load()

    def __matmul__(self, other):
        from .. import libop
        return libop.matmul(self, other)

    def __rmatmul__(self, other):
        from .. import libop
        return libop.matmul(other, self)


def _istensor(x):
    return type(x) is VarRef and x.ndim > 0


######################################
# Binary Operators


def add(lhs, rhs):
    '''
    `lhs + rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.add

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The sum
    '''
    return lhs + rhs


def sub(lhs, rhs):
    '''
    `lhs - rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.sub

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The difference
    '''
    return lhs - rhs


def mul(lhs, rhs):
    '''
    `lhs * rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.mul

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The product
    '''
    return lhs * rhs


def truediv(lhs, rhs):
    '''
    Floating point division of `lhs` dividing by `rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.truediv

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The quotient
    '''
    return lhs / rhs


def floordiv(lhs, rhs):
    '''
    Floored integer division of `lhs` dividing by `rhs`

    The result rounds towards negative infinity (following Python convention, instead of C)
    This function is recommended over `round_towards_0_div`, as it enjoys more optimizations

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.floordiv

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The quotient
    '''
    return lhs // rhs


def ceildiv(lhs, rhs):
    '''
    Ceiling integer division of `lhs` dividing by `rhs`

    The result rounds towards positive infinity

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.ceildiv

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The quotient
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.ceildiv(lhs, rhs)
    if type(lhs) is int and type(rhs) is int:
        return lhs // rhs + (lhs % rhs > 0)
    return ffi.makeCeilDiv(lhs, rhs)


def round_towards_0_div(lhs, rhs):
    '''
    C-style integer division of `lhs` dividing by `rhs`

    The result rounds towards 0 (following C convention, instead of Python)
    End users are encouraged to use `lhs // rhs` instead, which follows Python convetion,
    and enjoys better optimization in FreeTensor

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.round_towards_0_div

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The quotient
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.round_towards_0_div(lhs, rhs)
    return ffi.makeRoundTowards0Div(lhs, rhs)


def mod(lhs, rhs):
    '''
    `lhs` modulus `rhs`

    The result is always non-negative (following Python convention, instead of C).
    This function is recommended over `remainder`, as it enjoys more optimizations

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.mod

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The modulo
    '''
    return lhs % rhs


def remainder(lhs, rhs):
    '''
    Remainder of `lhs` dividing `rhs`

    The result can be positive or negative (following C convention, instead of Python).
    End users are encouraged to use `lhs % rhs` instead, which follows Python convetion,
    and enjoys better optimization in FreeTensor

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.remainder

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The remainder
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.remainder(lhs, rhs)
    return ffi.makeRemainder(lhs, rhs)


def min(lhs, rhs):
    '''
    Minimum of `lhs` and `rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.min

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The minimum
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.min(lhs, rhs)
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return builtins.min(lhs, rhs)
    return ffi.makeMin(lhs, rhs)


def max(lhs, rhs):
    '''
    Maximum of `lhs` and `rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.max

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The maximum
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.max(lhs, rhs)
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return builtins.max(lhs, rhs)
    return ffi.makeMax(lhs, rhs)


def l_and(lhs, rhs):
    '''
    Logical and of `lhs` and `rhs`

    NOTE: Short-circuit evaluation is NOT supported

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.l_and

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The logical and
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.l_and(lhs, rhs)
    if type(lhs) is bool and type(rhs) is bool:
        return lhs and rhs
    else:
        return ffi.makeLAnd(lhs, rhs)


def l_or(lhs, rhs):
    '''
    Logical or of `lhs` and `rhs`

    NOTE: Short-circuit evaluation is NOT supported

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.l_or

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The logical or
    '''
    if _istensor(lhs) or _istensor(rhs):
        from .. import libop
        return libop.l_or(lhs, rhs)
    if type(lhs) is bool and type(rhs) is bool:
        return lhs or rhs
    else:
        return ffi.makeLOr(lhs, rhs)


def lt(lhs, rhs):
    '''
    `lhs < rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.lt

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs < rhs


def le(lhs, rhs):
    '''
    `lhs <= rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.le

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs <= rhs


def gt(lhs, rhs):
    '''
    `lhs > rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.gt

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs > rhs


def ge(lhs, rhs):
    '''
    `lhs >= rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.ge

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs >= rhs


def eq(lhs, rhs):
    '''
    `lhs == rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.eq

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs == rhs


def ne(lhs, rhs):
    '''
    `lhs != rhs`

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.ne

    Parameters
    ----------
    lhs : VarRef or Number
        The left-hand-side operand
    rhs : VarRef or Number
        The right-hand-side operand

    Returns
    -------
    VarRef or Number
        The comparison
    '''
    return lhs != rhs


######################################
# Unary Operators


def l_not(expr):
    '''
    Logical not

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.l_not

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The logical not
    '''
    if _istensor(expr):
        from .. import libop
        return libop.l_not(expr)
    if type(expr) is bool:
        return not expr
    else:
        return ffi.makeLNot(expr)


def abs(expr):
    '''
    Absolute value

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.abs

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The absolute value
    '''
    if _istensor(expr):
        from .. import libop
        return libop.abs(expr)
    if isinstance(expr, Number):
        return builtins.abs(expr)
    return ffi.makeAbs(expr)


def sqrt(expr):
    '''
    Square root

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.sqrt

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The square root
    '''
    if _istensor(expr):
        from .. import libop
        return libop.sqrt(expr)
    if isinstance(expr, Number):
        return math.sqrt(expr)
    return ffi.makeSqrt(expr)


def exp(expr):
    '''
    Natural exponent

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.exp

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The exponent
    '''
    if _istensor(expr):
        from .. import libop
        return libop.exp(expr)
    if isinstance(expr, Number):
        return math.exp(expr)
    return ffi.makeExp(expr)


def square(expr):
    '''
    Square

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.square

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The square
    '''
    if _istensor(expr):
        from .. import libop
        return libop.square(expr)
    if isinstance(expr, Number):
        return expr * expr
    return ffi.makeSquare(expr)


def sigmoid(expr):
    '''
    Sigmoid

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.sigmoid

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    if _istensor(expr):
        from .. import libop
        return libop.sigmoid(expr)
    return ffi.makeSigmoid(expr)


def tanh(expr):
    '''
    Hyperbolic tangent

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.tanh

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    if _istensor(expr):
        from .. import libop
        return libop.tanh(expr)
    if isinstance(expr, Number):
        return math.tanh(expr)
    return ffi.makeTanh(expr)


def floor(expr):
    '''
    Round a float down to an interger (towards -inf)

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.floor

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    if _istensor(expr):
        from .. import libop
        return libop.floor(expr)
    return ffi.makeFloor(expr)


def ceil(expr):
    '''
    Round a float up to an interger (towards +inf)

    For scalar operands, it emit an expression node in AST. For non-scalar operands,
    it calls libop.ceil

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    if _istensor(expr):
        from .. import libop
        return libop.ceil(expr)
    return ffi.makeCeil(expr)


def if_then_else(cond, then_case, else_case):
    '''
    Similar to `then_case if cond else else_case`

    NOTE: there is NO guarantee that only one branch will be executed. In some cases,
    both branches will be executed and the result of one of them will be picked.
    Therefore, please do NOT use `if_then_else` to guard an out-of-bound array indexing

    Parameters
    ----------
    cond : VarRef of Number
        Condition
    lhs : VarRef or Number
        Then-case experssion
    rhs : VarRef or Number
        Else-case expression

    Returns
    -------
    VarRef or Number
        The result
    '''
    if type(cond) is bool:
        return then_case if cond else else_case
    return ffi.makeIfExpr(cond, then_case, else_case)


def cast(expr, dtype):
    '''
    Cast to another type

    Parameters
    ----------
    expr : VarRef or Number
        The operand
    dtype : DataTypr or str
        The target data type

    Returns
    -------
    VarRef or Number
        The result
    '''
    return ffi.makeCast(expr, ffi.DataType(dtype))


def intrinsic(fmt, *params, **kws):
    """
    Invoke whatever target code

    Parameters
    ----------
    fmt : str
        What to run. "%" is filled by parameters one by one. E.g. sinf(%)
    The following variadic arguments : Expr
        Parameters to `fmt`
    ret_type : DataType or str
        (Keyword argument only) The return type. Void for no return type. Defaults to Void
    has_side_effect: bool
        (Keyword argument only) True to indicate the intrinsic modifes something other than the return value. Defaults to false
    """
    ret_type = ffi.DataType("void")
    has_side_effect = False
    if "ret_type" in kws:
        ret_type = ffi.DataType(kws["ret_type"])
        del kws["ret_type"]
    if "has_side_effect" in kws:
        has_side_effect = kws["has_side_effect"]
        del kws["has_side_effect"]
    assert len(kws) == 0, "Unrecognized keyword arguments: %s" % kws
    return ffi.makeIntrinsic(fmt, params, ret_type, has_side_effect)


def any():
    '''
    Create an AnyExpr node (only for testing)

    Any nodes matches any expression nodes in `ast.match`
    '''
    return ffi.makeAnyExpr()


def ndim(var):
    ''' Get the number of dimensions of a variable '''
    if isinstance(var, VarRef):
        return var.ndim
    else:
        return 0


def shape(var, i=None):
    ''' shape(var, i): Get size of specified dimension of a variable
        shape(var): Get sizes of all dimensions of a variable '''
    if isinstance(var, VarRef):
        return var.shape(i)
    else:
        if i is None:
            return ()
        else:
            raise Exception(f'Getting size of dimension {i} of scalar {var}')


def dtype(var):
    ''' Get element data type of a variable '''
    if isinstance(var, VarRef):
        return var.dtype
    elif isinstance(var, ffi.Expr):
        return var.dtype
    else:
        # TODO: Config default type
        if isinstance(var, float):
            return ffi.DataType("float32")
        elif isinstance(var, int):
            return ffi.DataType("int32")
        else:
            raise Exception('Unknown scalar type: ' + str(type(var)))


def mtype(var):
    ''' Get memory type of a variable '''
    if isinstance(var, VarRef):
        return var.mtype
    else:
        return 'byvalue'
