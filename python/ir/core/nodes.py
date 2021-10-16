import collections
import numpy as np
from typing import Sequence, Tuple, Any, Optional

import ffi

from .utils import *


class Context:

    def __init__(self):
        self.stmt_seq = []
        self.last_if = None  # To handle else case
        self.next_nid = ""
        self.next_no_deps = False

    def append_stmt(self, stmt: ffi.Stmt):
        self.stmt_seq.append(stmt)
        self.last_if = None
        self.next_nid = ""
        self.next_no_deps = False

    def append_if_then_stmt(self, cond, body: ffi.Stmt):
        next_nid = self.next_nid
        self.append_stmt(ffi.makeIf(next_nid, cond, body))
        self.last_if = (next_nid, cond, body)

    def append_if_else_stmt(self, elseCase: ffi.Stmt):
        nid, cond, thenCase = self.last_if
        self.stmt_seq.pop()
        self.append_stmt(ffi.makeIf(nid, cond, thenCase, elseCase))

    def append_for_stmt(self,
                        iter_var,
                        begin,
                        end,
                        body,
                        nid: str = "",
                        no_deps: Optional[bool] = None):
        if nid == "":
            nid = self.next_nid
        if no_deps is None:
            no_deps = self.next_no_deps
        self.append_stmt(
            ffi.makeFor(
                nid,
                iter_var,
                begin,
                end,
                end - begin,
                no_deps,
                ffi.ForProperty(),
                body,
            ))

    def set_next_nid(self, nid: str):
        self.next_nid = nid

    def get_next_nid(self):
        return self.next_nid

    def set_next_no_deps(self, no_deps: bool = True):
        self.next_no_deps = no_deps

    def get_next_no_deps(self):
        return self.next_no_deps

    def make_stmt(self, nid: str = ""):
        if len(self.stmt_seq) == 1 and nid == "":
            return self.stmt_seq[0]
        else:
            return ffi.makeStmtSeq(nid, self.stmt_seq)


class ContextStack:

    def __init__(self):
        self.reset()

    def reset(self):
        self.stack = [Context()]

    def top(self):
        return self.stack[-1]

    def push(self):
        self.stack.append(Context())

    def pop(self):
        return self.stack.pop()

    def get_stack(self):
        return self.stack

    def set_stack(self, stack):
        self.stack = stack


ctx_stack = ContextStack()


def pop_ast():
    """ Get AST and reset context """
    ret = ctx_stack.pop().make_stmt()
    ctx_stack.reset()
    return ret


class Var(ffi.FrontendVar):

    def __init__(self,
                 name: str,
                 full_shape: Sequence,
                 dtype: ffi.DataType,
                 mtype: ffi.MemType,
                 indices: Sequence = []):
        super(Var, self).__init__(name, full_shape, dtype, mtype, indices)

    def __getitem__(self, key):
        return Var(self.name, self.full_shape, self.dtype, self.mtype,
                   self.chain_indices(self._parse_key(key)))

    def __setitem__(self, key, value):
        var = Var(self.name, self.full_shape, self.dtype, self.mtype,
                  self.chain_indices(self._parse_key(key)))
        top = ctx_stack.top()
        top.append_stmt(var.as_store(top.get_next_nid(), value))

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
            elif isinstance(idx, Var):
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
        return self.as_load() + other

    def __radd__(self, other):
        return other + self.as_load()

    def __sub__(self, other):
        return self.as_load() - other

    def __rsub__(self, other):
        return other - self.as_load()

    def __mul__(self, other):
        return self.as_load() * other

    def __rmul__(self, other):
        return other * self.as_load()

    def __truediv__(self, other):
        return self.as_load() / other

    def __rtruediv__(self, other):
        return other / self.as_load()

    def __floordiv__(self, other):
        return self.as_load() // other

    def __rfloordiv__(self, other):
        return other // self.as_load()

    def __mod__(self, other):
        return self.as_load() % other

    def __rmod__(self, other):
        return other % self.as_load()

    def __lt__(self, other):
        return self.as_load() < other

    def __le__(self, other):
        return self.as_load() <= other

    def __gt__(self, other):
        return self.as_load() > other

    def __ge__(self, other):
        return self.as_load() >= other

    def __eq__(self, other):
        return self.as_load() == other

    def __ne__(self, other):
        return self.as_load() != other

    def __neg__(self):
        return 0 - self.as_load()


class _VarDef:

    def __init__(self, name: str, shape, dtype, atype, mtype):
        '''
        A variable can be created using a literal shape, or another fixed-length
        variable as a shape. If using the latter, the shape variable can be retrived
        using a `.shape` attribute
        '''

        self.name = name
        if isinstance(shape, collections.abc.Sequence):
            self.shape = shape
        elif isinstance(shape, Var):
            assert shape.ndim == 1, "Shape of a shape should be 1-D"
            assert type(
                shape.shape(0)
            ) is ffi.IntConst, "Dynamic number of dimensions is not supported"
            ndim = shape.shape(0).val
            self.shape = [shape[i] for i in range(ndim)]
        else:
            assert False, "shape cannot be of type %s" % type(shape)
        self.dtype = parseDType(dtype)
        self.atype = parseAType(atype)
        self.mtype = parseMType(mtype)

    def __enter__(self):
        ctx_stack.push()
        return Var(self.name, self.shape, self.dtype, self.mtype)

    def __exit__(self, exc_type, exc_value, traceback):
        buf = ffi.Buffer(ffi.Tensor(self.shape, self.dtype), self.atype,
                         self.mtype)
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        top.append_stmt(
            ffi.makeVarDef(top.get_next_nid(), self.name, buf, None, body,
                           False))


class _VarsDef:

    def __init__(self, defs: Tuple[str, Any, DataType, AccessType]):
        self.defs = [VarDef(*d) for d in defs]

    def __enter__(self):
        return [d.__enter__() for d in self.defs]

    def __exit__(self, exc_type, exc_value, traceback):
        for d in reversed(self.defs):
            d.__exit__(exc_type, exc_value, traceback)


# Factory
def VarDef(*args):
    if len(args) == 1:
        return _VarsDef(args[0])
    else:
        return _VarDef(*args)


class For:

    def __init__(self,
                 iter_var: str,
                 begin,
                 end,
                 nid: str = "",
                 no_deps: Optional[bool] = None):
        self.iter_var = iter_var
        self.begin = begin
        self.end = end
        self.nid = nid
        self.no_deps = no_deps

    def __enter__(self):
        ctx_stack.push()
        return ffi.makeVar(self.iter_var)

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        top.append_for_stmt(self.iter_var,
                            self.begin,
                            self.end,
                            body,
                            nid=self.nid,
                            no_deps=self.no_deps)


class If:

    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt()
        ctx_stack.top().append_if_then_stmt(self.cond, body)


class Else:

    def __init__(self):
        pass

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt()
        ctx_stack.top().append_if_else_stmt(body)


class Assert:

    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        nid = top.get_next_nid()
        top.append_stmt(ffi.makeAssert(nid, self.cond, body))


def MarkNid(nid: str):
    """ Mark the ID of the following statement """
    ctx_stack.top().set_next_nid(nid)


class NamedScope:

    def __init__(self, nid: str):
        self.nid = nid

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt(self.nid)
        ctx_stack.top().append_stmt(body)


def Eval(expr):
    top = ctx_stack.top()
    top.append_stmt(ffi.makeEval(top.get_next_nid(), expr))


def Any():
    ctx_stack.top().append_stmt(ffi.makeAny())


def min(lhs, rhs):
    return ffi.makeMin(lhs, rhs)


def max(lhs, rhs):
    return ffi.makeMax(lhs, rhs)


def if_then_else(cond, then_case, else_case):
    return ffi.makeIfExpr(cond, then_case, else_case)


def abs(expr):
    return ffi.makeAbs(expr)


def l_and(lhs, rhs):
    if type(lhs) is bool and type(rhs) is bool:
        return lhs and rhs
    else:
        return ffi.makeLAnd(lhs, rhs)


def l_or(lhs, rhs):
    if type(lhs) is bool and type(rhs) is bool:
        return lhs or rhs
    else:
        return ffi.makeLOr(lhs, rhs)


def l_not(expr):
    if type(expr) is bool:
        return not expr
    else:
        return ffi.makeLNot(expr)


def floor_div(lhs, rhs):
    return ffi.makeFloorDiv(lhs, rhs)


def ceil_div(lhs, rhs):
    return ffi.makeCeilDiv(lhs, rhs)


def round_towards_0_div(lhs, rhs):
    return ffi.makeRoundTowards0Div(lhs, rhs)


def sqrt(expr):
    return ffi.makeSqrt(expr)


def exp(expr):
    return ffi.makeExp(expr)


def square(expr):
    return ffi.makeSquare(expr)


def floor(expr):
    return ffi.makeFloor(expr)


def ceil(expr):
    return ffi.makeCeil(expr)


def cast(expr, dtype):
    return ffi.makeCast(expr, parseDType(dtype))


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
    """
    ret_type = ffi.DataType.Void
    if "ret_type" in kws:
        ret_type = parseDType(kws["ret_type"])
        del kws["ret_type"]
    assert len(kws) == 0, "Unrecognized keyword arguments: %s" % kws
    return ffi.makeIntrinsic(fmt, params, ret_type)


def any():
    return ffi.makeAnyExpr()


def Func(name, params, body, src=None):
    return ffi.makeFunc(name, params, body, src)


class Tensor(ffi.TensorData):

    def __init__(self, data, mtype):
        super(Tensor, self).__init__(np.array(data))
        self.mtype = parseMType(mtype)
