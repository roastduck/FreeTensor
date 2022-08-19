'''
Facility to build AST statements

Classes and functions in this module are internally used by `transformer` to construct ASTs.
They are also used by some internal tests. API of these classes and functions are subject to changes.
End users are encouraged to use `transformer`, instead of this module.

Classes and functions in this module are all in BigCamel naming style, to distinguish from
expressions in `expr.py`
'''

import collections
from typing import Sequence, Tuple, Any, Optional

import freetensor_ffi as ffi

from . import config
from .context import ctx_stack
from .expr import VarRef

open_vardefs = {}


def find_borrowed_vardefs(exprs: Sequence):
    borrowed_vardefs = set()
    for expr in exprs:
        for name in ffi.all_reads(
                expr if type(expr) is ffi.FrontendVarIdx else ffi.Expr(expr)):
            borrowed_vardefs.add(open_vardefs[name])
    return borrowed_vardefs


class _VarDef:

    def __init__(self, name: str, shape, dtype, atype, mtype=None):
        '''
        Scope used for creating a VarDef AST node. A VarRef will be returned as a
        reference to the variable of the VarDef node

        This scope is internally used by `transformer` and tests

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
        self.borrowed_vardefs = find_borrowed_vardefs(self.shape)
        self.metadata = ctx_stack.top().get_metadata()

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
            ffi.makeVarDef(self.name, buf, None, body, False, self.metadata))


class _VarsDef:
    '''
    Helper class to create a series of nested VarDef nodes

    This scope is internally used by `transformer` and tests
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

    This scope is internally used by `transformer` and tests
    '''

    if len(args) == 1:
        return _VarsDef(args[0])
    else:
        return _VarDef(*args)


class For:
    '''
    Scope used to create a For node

    This scope is internally used by `transformer` and tests

    E.g.:

    ```
    with For('i', 0, n) as i:
        ... # Loop body
    ```
    '''

    def __init__(self,
                 iter_var: str,
                 begin,
                 end,
                 step=1,
                 label: Optional[str] = None,
                 no_deps: Optional[Sequence[str]] = None,
                 prefer_libs: Optional[bool] = None):
        self.iter_var = iter_var
        self.begin = begin
        self.end = end
        self.step = step
        self.label = label
        self.no_deps = no_deps
        self.prefer_libs = prefer_libs

        self.borrowed_vardefs = set()
        for x in [begin, end, step]:
            for name in ffi.all_reads(ffi.Expr(x)):
                self.borrowed_vardefs.add(open_vardefs[name])

    def __enter__(self):
        for item in self.borrowed_vardefs:
            item.lend_out()
        ctx_stack.push()
        return ffi.makeVar(self.iter_var)

    def __exit__(self, exc_type, exc_value, traceback):
        for item in self.borrowed_vardefs:
            item.reclaim()
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
                            metadata=ffi.SourceMetadata([self.label])
                            if self.label is not None else None,
                            no_deps=self.no_deps,
                            prefer_libs=self.prefer_libs)


class If:
    '''
    Scope used to create an If node

    This scope is internally used by `transformer` and tests

    E.g.:

    ```
    with If(i > 0):
        ... # Branch body
    ```
    '''

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
    '''
    Scope used to create an else branch of an If node

    This scope is internally used by `transformer` and tests

    E.g.:

    ```
    with If(i > 0):
        ... # True branch
    with Else():
        ... # Else branch
    ```
    '''

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
    '''
    Scope used to create an Assert node

    This scope is internally used by `transformer` and tests

    E.g.:

    ```
    with Assert(i > 0):
        ... # Assertion body
    ```
    '''

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
        top.append_stmt(ffi.makeAssert(self.cond, body, top.get_metadata()))


def MarkLabel(label: str):
    """
    Mark the ID of the following statement

    This scope is internally used by `transformer` and tests
    """
    ctx_stack.top().add_label(label)


class NamedScope:
    '''
    Scope used to create an StmtSeq node with an explicit ID

    E.g.:

    ```
    with NamedScope():
        ... # body
    ```

    This scope is used for testing only. StmtSeq nodes can be deleted in many lowering passes
    '''

    def __init__(self, *labels: str):
        self.labels = labels

    def __enter__(self):
        ctx_stack.push()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        finished_scope = ctx_stack.pop()
        metadata = ctx_stack.top().get_metadata(self.labels)
        body = finished_scope.make_stmt(metadata)
        ctx_stack.top().append_stmt(body)


def Invoke(func, *args, **kvs):
    '''
    Inlined invocation of another AST

    This scope is internally used by `transformer` and tests

    `Invoke` can be used for invoking a gradient function, which has already been lowered as an AST.
    Please note that once a user function has been lowered as an AST, the dimensionalities of its
    tensors get fixed. Therefore, to invoke ordinary user functions, please use `inline` in `transformer`
    instead, which supports generic types
    '''
    top = ctx_stack.top()
    top.append_stmt(ffi.inlined_invoke(top.get_metadata(), func, args, kvs))


def Alloc(var: VarRef):
    top = ctx_stack.top()
    top.append_stmt(ffi.makeAlloc(var.name, top.get_metadata()))


def Free(var: VarRef):
    top = ctx_stack.top()
    top.append_stmt(ffi.makeFree(var.name, top.get_metadata()))


def Eval(expr):
    '''
    Create an Eval node

    This scope is internally used by `transformer` and tests
    '''
    top = ctx_stack.top()
    top.append_stmt(ffi.makeEval(expr, top.get_metadata()))


def Any():
    '''
    Create an Any node (only for testing)

    Any nodes matches any statement nodes in `ast.match`
    '''
    ctx_stack.top().append_stmt(ffi.makeAny())


def Func(name, params, returns, body, closure={}):
    return ffi.makeFunc(name, params, returns, body, closure)


def lookup_id(ast, pattern):
    from .schedule import Schedule
    return Schedule(ast).find(pattern).id
