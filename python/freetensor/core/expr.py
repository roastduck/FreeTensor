'''
Facility to build AST expressions

Classes and functions in this module are not only used internally for constructing AST nodes,
and also exposed to users via multi-stage programming
'''

import collections
import builtins
from typing import Sequence, Tuple, Any, Optional

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
        top.append_stmt(var.as_store(top.get_next_nid(), value))

    def select(self, idx, dim):
        assert isinstance(dim, int)
        assert dim >= 0 and dim < self.ndim
        indices = [
            slice(None, None) if d != dim else idx for d in range(self.ndim)
        ]
        return self[indices]

    def _parse_key(self, key):
        if not isinstance(key, collections.abc.Sequence):
            key = (key,)
        ffiIdx = []
        for idx, length in zip(key, self.full_shape):
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


def remainder(lhs, rhs):
    '''
    Remainder of `lhs` dividing `rhs`

    The result can be positive or negative (following C convention, instead of Python).
    End users are encouraged to use `lhs % rhs` instead, which follows Python convetion,
    and enjoys better optimization in FreeTensor

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
    return ffi.makeRemainder(lhs, rhs)


def min(lhs, rhs):
    '''
    Minimum of `lhs` and `rhs`

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
    if type(lhs) in (int, float) and type(rhs) in (int, float):
        return builtins.min(lhs, rhs)
    return ffi.makeMin(lhs, rhs)


def max(lhs, rhs):
    '''
    Maximum of `lhs` and `rhs`

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
    if type(lhs) in (int, float) and type(rhs) in (int, float):
        return builtins.max(lhs, rhs)
    return ffi.makeMax(lhs, rhs)


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


def abs(expr):
    '''
    Absolute value

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The absolute value
    '''
    if type(expr) in (int, float):
        return builtins.abs(expr)
    return ffi.makeAbs(expr)


def l_and(lhs, rhs):
    '''
    Logical and of `lhs` and `rhs`

    NOTE: Short-circuit evaluation is NOT supported

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
    if type(lhs) is bool and type(rhs) is bool:
        return lhs and rhs
    else:
        return ffi.makeLAnd(lhs, rhs)


def l_or(lhs, rhs):
    '''
    Logical or of `lhs` and `rhs`

    NOTE: Short-circuit evaluation is NOT supported

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
    if type(lhs) is bool and type(rhs) is bool:
        return lhs or rhs
    else:
        return ffi.makeLOr(lhs, rhs)


def l_not(expr):
    '''
    Logical not

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The logical not
    '''
    if type(expr) is bool:
        return not expr
    else:
        return ffi.makeLNot(expr)


def floor_div(lhs, rhs):
    '''
    Floored integer division of `lhs` dividing by `rhs`

    The result rounds towards negative infinity (following Python convention, instead of C)

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
    if type(lhs) is int and type(rhs) is int:
        return lhs // rhs
    return ffi.makeFloorDiv(lhs, rhs)


def ceil_div(lhs, rhs):
    '''
    Ceiling integer division of `lhs` dividing by `rhs`

    The result rounds towards positive infinity

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
    if type(lhs) is int and type(rhs) is int:
        return lhs // rhs + (lhs % rhs > 0)
    return ffi.makeCeilDiv(lhs, rhs)


def round_towards_0_div(lhs, rhs):
    '''
    C-style integer division of `lhs` dividing by `rhs`

    The result rounds towards 0 (following C convention, instead of Python)
    End users are encouraged to use `lhs // rhs` instead, which follows Python convetion,
    and enjoys better optimization in FreeTensor

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
    return ffi.makeRoundTowards0Div(lhs, rhs)


def sqrt(expr):
    '''
    Square root

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The square root
    '''
    return ffi.makeSqrt(expr)


def exp(expr):
    '''
    Natural exponent

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The exponent
    '''
    return ffi.makeExp(expr)


def square(expr):
    '''
    Square

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The square
    '''
    return ffi.makeSquare(expr)


def sigmoid(expr):
    '''
    Sigmoid

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    return ffi.makeSigmoid(expr)


def tanh(expr):
    '''
    Hyperbolic tangent

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    return ffi.makeTanh(expr)


def floor(expr):
    '''
    Round a float down to an interger (towards -inf)

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    return ffi.makeFloor(expr)


def ceil(expr):
    '''
    Round a float up to an interger (towards +inf)

    Parameters
    ----------
    expr : VarRef or Number
        The operand

    Returns
    -------
    VarRef or Number
        The result
    '''
    return ffi.makeCeil(expr)


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


def shape(var, i):
    ''' Get all dimensions of a variable '''
    if isinstance(var, VarRef):
        return var.shape(i)
    else:
        return []


def dtype(var):
    ''' Get element data type of a variable '''
    if isinstance(var, VarRef):
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
