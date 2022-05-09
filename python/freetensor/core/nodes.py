import collections
import sys
import numpy as np
from typing import Sequence, Tuple, Any, Optional

import ffi
from ffi import dump_ast, load_ast

from . import config


class Context:

    def __init__(self):
        self.stmt_seq = []
        self.last_if = None  # To handle else case
        self.next_nid = ""
        self.next_no_deps = []
        self.next_prefer_libs = False

    def append_stmt(self, stmt: ffi.Stmt):
        self.stmt_seq.append(stmt)
        self.last_if = None
        self.next_nid = ""
        self.next_no_deps = []
        self.next_prefer_libs = False

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
                        step,
                        body,
                        nid: str = "",
                        no_deps: Optional[Sequence[str]] = None,
                        prefer_libs: Optional[bool] = None):
        if nid == "":
            nid = self.next_nid
        if no_deps is None:
            no_deps = self.next_no_deps
        if prefer_libs is None:
            prefer_libs = self.next_prefer_libs
        self.append_stmt(
            ffi.makeFor(
                nid,
                iter_var,
                begin,
                end,
                step,
                (end - begin) // step,
                ffi.ForProperty().with_no_deps(no_deps).with_prefer_libs(
                    prefer_libs),
                body,
            ))

    def set_next_nid(self, nid: str):
        self.next_nid = nid

    def get_next_nid(self):
        return self.next_nid

    def add_next_no_deps(self, var):
        self.next_no_deps.append(var)

    def reset_next_no_deps(self):
        self.next_no_deps = []

    def get_next_no_deps(self):
        return self.next_no_deps

    def set_next_prefer_libs(self, prefer_libs=True):
        self.next_prefer_libs = prefer_libs

    def get_next_prefer_libs(self):
        return self.next_prefer_libs

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


def pop_ast(verbose: bool = False):
    """ Get AST and reset context """
    ret = ctx_stack.pop().make_stmt()
    ctx_stack.reset()
    if verbose:
        print("The popped AST is:", file=sys.stderr)
        print(ret, file=sys.stderr)
        print(file=sys.stderr)
    return ret


class VarRef(ffi.FrontendVar):

    def __init__(self,
                 name: str,
                 vardef,
                 full_shape: Sequence,
                 dtype: ffi.DataType,
                 mtype: ffi.MemType,
                 indices: Sequence = []):
        super(VarRef, self).__init__(name, full_shape, dtype, mtype, indices)
        self.vardef = vardef

        self.borrowed_vardefs = set()
        for idx in indices:
            for name in ffi.all_reads(idx):
                self.borrowed_vardefs.add(open_vardefs[name])
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
        if var.vardef.atype == ffi.AccessType("input"):
            raise ffi.InvalidProgram("Cannot modify an \"input\" tensor `" +
                                     self.name)
        if var.vardef.borrower_cnt > 0:
            raise ffi.InvalidProgram(
                "Cannot modify tensor `" + self.name +
                "` becuase it has been borrowed in another tensor's shape, or a tensor slice"
            )
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


open_vardefs = {}


class _VarDef:

    def __init__(self, name: str, shape, dtype, atype, mtype=None):
        '''
        Scope used for creating a VarDef AST node. A VarRef will be returned as a
        reference to the variable of the VarDef node

        Parameters
        ----------
        name : str
            Name of the variable
        shape : Sequence[Expr] or VarRef
            Shape of the variable. A variable can be created using a literal shape,
            or another fixed-length VarRef as a shape. If using the latter, the shape
            VarRef can be retrived using a `.shape` attribute
        dtype : str or DataType
            Data type of the variable
        atype : str or AccessType
            Access type of the variable. It specifies whether (and how) the variable
            is an I/O variable of the function it belongs to
        mtype : str or MemType (Optional)
            Memory type of the variable. If omitted, the main memory type of the
            default Target in config will be used
        '''

        self.name = name
        if isinstance(shape, collections.abc.Sequence):
            self.shape = shape
        elif isinstance(shape, VarRef):
            assert shape.ndim == 1, "Shape of a shape should be 1-D"
            assert type(
                shape.shape(0)
            ) is ffi.IntConst, "Dynamic number of dimensions is not supported"
            ndim = shape.shape(0).val
            self.shape = [shape[i] for i in range(ndim)]
        else:
            assert False, "shape cannot be of type %s" % type(shape)
        self.dtype = ffi.DataType(dtype)
        self.atype = ffi.AccessType(atype)
        if mtype is not None:
            self.mtype = ffi.MemType(mtype)
        else:
            self.mtype = config.default_target().main_mem_type()

        self.borrower_cnt = 0

        self.borrowed_vardefs = set()
        for dim in self.shape:
            for name in ffi.all_reads(ffi.Expr(dim)):
                self.borrowed_vardefs.add(open_vardefs[name])

    def set_atype(self, atype):
        self.atype = ffi.AccessType(atype)

    def lend_out(self):
        self.borrower_cnt += 1

    def reclaim(self):
        self.borrower_cnt -= 1

    def __enter__(self):
        for item in self.borrowed_vardefs:
            item.lend_out()

        ctx_stack.push()
        if self.name in open_vardefs:
            raise ffi.InvalidProgram("Nested VarDefs with the same name `" +
                                     self.name + "` is not allowed")
        open_vardefs[self.name] = self
        return VarRef(self.name, self, self.shape, self.dtype, self.mtype)

    def __exit__(self, exc_type, exc_value, traceback):
        del open_vardefs[self.name]
        for item in self.borrowed_vardefs:
            item.reclaim()

        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        buf = ffi.Buffer(ffi.Tensor(self.shape, self.dtype), self.atype,
                         self.mtype)
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        top.append_stmt(
            ffi.makeVarDef(top.get_next_nid(), self.name, buf, None, body,
                           False))


class _VarsDef:
    '''
    Helper class to create a series of nested VarDef nodes
    '''

    def __init__(self, defs: Tuple[str, Any, ffi.DataType, ffi.AccessType]):
        self.defs = [VarDef(*d) for d in defs]

    def __enter__(self):
        return [d.__enter__() for d in self.defs]

    def __exit__(self, exc_type, exc_value, traceback):
        for d in reversed(self.defs):
            d.__exit__(exc_type, exc_value, traceback)


def VarDef(*args):
    '''
    A factory function that creates a VarDef or a series of nested `VarDef`s
    '''

    if len(args) == 1:
        return _VarsDef(args[0])
    else:
        return _VarDef(*args)


class For:

    def __init__(self,
                 iter_var: str,
                 begin,
                 end,
                 step=1,
                 nid: str = "",
                 no_deps: Optional[Sequence[str]] = None,
                 prefer_libs: Optional[bool] = None):
        self.iter_var = iter_var
        self.begin = begin
        self.end = end
        self.step = step
        self.nid = nid
        self.no_deps = no_deps
        self.prefer_libs = prefer_libs

    def __enter__(self):
        ctx_stack.push()
        return ffi.makeVar(self.iter_var)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        body = ctx_stack.pop().make_stmt()
        top = ctx_stack.top()
        top.append_for_stmt(self.iter_var,
                            self.begin,
                            self.end,
                            self.step,
                            body,
                            nid=self.nid,
                            no_deps=self.no_deps,
                            prefer_libs=self.prefer_libs)


class If:

    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        body = ctx_stack.pop().make_stmt()
        ctx_stack.top().append_if_then_stmt(self.cond, body)


class Else:

    def __init__(self):
        pass

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        body = ctx_stack.pop().make_stmt()
        ctx_stack.top().append_if_else_stmt(body)


class Assert:

    def __init__(self, cond):
        self.cond = cond

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
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
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        body = ctx_stack.pop().make_stmt(self.nid)
        ctx_stack.top().append_stmt(body)


def Invoke(func, *args, **kvs):
    top = ctx_stack.top()
    top.append_stmt(ffi.inlined_invoke(top.get_next_nid(), func, args, kvs))


def Eval(expr):
    top = ctx_stack.top()
    top.append_stmt(ffi.makeEval(top.get_next_nid(), expr))


def Any():
    ctx_stack.top().append_stmt(ffi.makeAny())


def remainder(lhs, rhs):
    return ffi.makeRemainder(lhs, rhs)


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


def sigmoid(expr):
    return ffi.makeSigmoid(expr)


def tanh(expr):
    return ffi.makeTanh(expr)


def floor(expr):
    return ffi.makeFloor(expr)


def ceil(expr):
    return ffi.makeCeil(expr)


def cast(expr, dtype):
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
    return ffi.makeAnyExpr()


def Func(name, params, returns, body, closure={}):
    return ffi.makeFunc(name, params, returns, body, closure)


def ndim(var):
    if isinstance(var, VarRef):
        return var.ndim
    else:
        return 0


def shape(var, i):
    if isinstance(var, VarRef):
        return var.shape(i)
    else:
        raise Exception('Scalar object has no shape')


def dtype(var):
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
    if isinstance(var, VarRef):
        return var.mtype
    else:
        return 'byvalue'
