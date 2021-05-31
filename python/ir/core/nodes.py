import collections
from typing import Sequence, Tuple, Any

import ffi

from .utils import *


class Context:

    def __init__(self):
        self.stmt_seq = []
        self.lastIf = None  # To handle else case
        self.nextNid = ""

    def append_stmt(self, stmt: ffi.Stmt):
        self.stmt_seq.append(stmt)
        self.lastIf = None
        self.nextNid = ""

    def append_if_then_stmt(self, cond, body: ffi.Stmt):
        nextNid = self.nextNid
        self.append_stmt(ffi.makeIf(nextNid, cond, body))
        self.lastIf = (nextNid, cond, body)

    def append_if_else_stmt(self, elseCase: ffi.Stmt):
        nid, cond, thenCase = self.lastIf
        self.stmt_seq.pop()
        self.append_stmt(ffi.makeIf(nid, cond, thenCase, elseCase))

    def set_next_nid(self, nid: str):
        self.nextNid = nid

    def get_next_nid(self):
        return self.nextNid

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

    def __init__(self, name: str, shape: Sequence, indices: Sequence = []):
        super(Var, self).__init__(name, shape, indices)

    def __getitem__(self, key):
        return Var(self.name, self.shape,
                   self.chain_indices(self._parse_key(key)))

    def __setitem__(self, key, value):
        var = Var(self.name, self.shape,
                  self.chain_indices(self._parse_key(key)))
        top = ctx_stack.top()
        top.append_stmt(var.as_store(top.get_next_nid(), value))

    def _parse_key(self, key):
        if not isinstance(key, collections.abc.Sequence):
            key = (key,)
        ffiIdx = []
        for idx, length in zip(key, self.shape):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else length
                assert idx.step is None or idx.step == 1
                ffiIdx.append(ffi.FrontendVarIdx(start, stop))
            elif isinstance(idx, Var):
                assert len(key) == 1, "Shape of an index should be 1-D"
                assert type(
                    idx.shape[0]
                ) is ffi.IntConst, "Dynamic number of dimensions is not supported"
                ndim = idx.shape[0].val
                ffiIdx += [
                    ffi.FrontendVarIdx(idx[i].as_load()) for i in range(ndim)
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


class _VarDef:

    def __init__(self, name: str, shape, dtype, atype, mtype):
        self.name = name
        if isinstance(shape, collections.abc.Sequence):
            self.shape = shape
        elif isinstance(shape, Var):
            assert len(shape.shape) == 1, "Shape of a shape should be 1-D"
            assert type(
                shape.shape[0]
            ) is ffi.IntConst, "Dynamic number of dimensions is not supported"
            ndim = shape.shape[0].val
            self.shape = [shape[i] for i in range(ndim)]
        else:
            assert False, "shape cannot be of type %s" % type(shape)
        self.dtype = parseDType(dtype)
        self.atype = parseAType(atype)
        self.mtype = parseMType(mtype)

    def __enter__(self):
        ctx_stack.push()
        return Var(self.name, self.shape)

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

    def __init__(self, iter_var: str, begin, end, nid: str = ""):
        self.iter_var = iter_var
        self.begin = begin
        self.end = end
        self.nid = nid

    def __enter__(self):
        ctx_stack.push()
        return ffi.makeVar(self.iter_var)

    def __exit__(self, exc_type, exc_value, traceback):
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        nid = top.get_next_nid()
        if self.nid != "":
            nid = self.nid
        top.append_stmt(
            ffi.makeFor(
                nid,
                self.iter_var,
                self.begin,
                self.end,
                self.end - self.begin,
                "",
                False,
                False,
                body,
            ))


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


def l_and(lhs, rhs):
    return ffi.makeLAnd(lhs, rhs)


def l_or(lhs, rhs):
    return ffi.makeLOr(lhs, rhs)


def l_not(expr):
    return ffi.makeLNot(expr)


def floor_div(lhs, rhs):
    return ffi.makeFloorDiv(lhs, rhs)


def ceil_div(lhs, rhs):
    return ffi.makeCeilDiv(lhs, rhs)


def round_towards_0_div(lhs, rhs):
    return ffi.makeRoundTowards0Div(lhs, rhs)


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


def Func(name, params, body):
    return ffi.makeFunc(name, params, body)
