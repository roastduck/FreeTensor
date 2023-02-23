'''
Facility to build AST statements

Classes and functions in this module are internally used by `transformer` to construct ASTs.
They are also used by some internal tests. API of these classes and functions are subject to changes.
End users are encouraged to use `transformer`, instead of this module.

Classes and functions in this module are all in BigCamel naming style, to distinguish from
expressions in `expr.py`
'''

import collections
from typing import Sequence, Mapping, Tuple, Any, Optional

import freetensor_ffi as ffi

from . import config
from .context import ctx_stack, StmtRange
from .expr import VarRef, VarRefFromVarDef

open_vardefs = {}


def find_borrowed_vardefs(exprs: Sequence):
    borrowed_vardefs = set()
    for expr in exprs:
        for name in ffi.all_reads(
                expr if type(expr) is ffi.FrontendVarIdx else ffi.Expr(expr)):
            borrowed_vardefs.add(open_vardefs[name])
    return borrowed_vardefs


class _VarDef:

    def __init__(self,
                 name: str,
                 shape,
                 dtype,
                 atype,
                 mtype=None,
                 view_of=None):
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
        view_of : str (Optional)
            (Internal use only) Set the VarDef node of another VarDef node
        '''

        self.name = name
        if isinstance(shape, collections.abc.Sequence):
            self.shape = shape
        elif isinstance(shape, VarRef):
            assert shape.ndim == 1, "Shape of a shape should be 1-D"
            assert type(shape.shape(
                0)) is int, "Dynamic number of dimensions is not supported"
            ndim = shape.shape(0)
            self.shape = [shape[i] for i in range(ndim)]
        else:
            assert False, "shape cannot be of type %s" % type(shape)
        self.dtype = ffi.DataType(dtype)
        self.atype = ffi.AccessType(atype)
        if mtype is not None:
            self.mtype = ffi.MemType(mtype)
        else:
            self.mtype = config.default_target().main_mem_type()
        self.view_of = view_of

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
        return VarRefFromVarDef(self.name, self, self.shape, self.dtype,
                                self.mtype)

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
            ffi.makeVarDef(self.name, buf, self.view_of, body, False,
                           self.metadata))


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


def VarDef(*args, **kvs):
    '''
    A factory function that creates a VarDef or a series of nested `VarDef`s

    This scope is internally used by `transformer` and tests
    '''

    if len(args) == 1:
        return _VarsDef(args[0])
    else:
        return _VarDef(*args, **kvs)


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


class Invoke:
    '''
    Inlined invocation of another AST

    `Invoke` is used as a scope (`with Invoke(...) as returned_vars`), so that variables returned by
    the callee can be used in the socpe

    `Invoke` can be used for invoking a gradient function, which has already been lowered as an AST.
    Please note that once a user function has been lowered as an AST, the dimensionalities of its
    tensors get fixed. Therefore, to invoke ordinary user functions, please use `inline` in `transformer`
    instead, which supports generic types
    '''

    def __init__(self,
                 ret_names: Sequence[str],
                 func: ffi.Func,
                 args: Sequence = [],
                 kvs: Mapping = {}):
        self.args = args
        self.kvs = kvs
        self.func, returns = ffi.strip_returns(func)
        self.vardefs = []  # Outer to inner
        assert len(ret_names) == len(returns)
        for name, ret in zip(ret_names, returns):
            self.vardefs.append(
                _VarDef(name, ret.tensor.shape, ret.tensor.dtype, "cache",
                        ret.mtype))

    def __enter__(self):
        varrefs = []
        ret_names = []
        for vardef in self.vardefs:
            varref = vardef.__enter__()
            varrefs.append(varref)
            ret_names.append(varref.name)
        ctx_stack.top().append_stmt(
            ffi.inlined_invoke(ctx_stack.top().get_metadata(), self.func,
                               self.args, self.kvs, ret_names))
        return varrefs[0] if len(varrefs) == 1 else tuple(varrefs)

    def __exit__(self, exc_type, exc_value, traceback):
        for vardef in reversed(self.vardefs):
            vardef.__exit__(exc_type, exc_value, traceback)


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


def MarkVersion(tape_name: str, var: VarRef):
    '''
    Create an MarkVersion node (only for custom gradient)

    This node is only used for custom gradient. See `UserGradForPrevStmt`.
    '''
    top = ctx_stack.top()
    top.append_stmt(ffi.makeMarkVersion(tape_name, var.name,
                                        top.get_metadata()))


class UserGrad:
    '''
    Define a custom gradient

    Follow the following steps to define custom gradient:

    1. Add some `mark_version` statements in the program. `mark_version('y0', y)` marks the specific
    versions of variable `y` **at the program position of the statement** and **at all iterations**
    as `'y0'`.
    2. Add a `UserGrad` scope.
    2.1. `UserGrad` optionally receives parameter `stmt_range`, recorded by the `StmtRange` helper class,
    which means the gradient is for the code specified in the range. Ignoring the parameter means setting
    gradient for the previous statement of the scope.
    2.2. Other parameters of `UserGrad` sets the mapping from original variables to gradient variables.
    `with UserGradForPrevStmt(x, y) as (dx, dy)` provides `VarRef` `dx` and `dy` as gradient variables
    to be used inside the scope.
    3. In order to use the value from the forward pass in the backward pass, do not access the forward
    variables directly in the scope. Instead, use `load_at_version` expressions. `load_at_version(y0, i, j)`
    loads from `y[i, j]` **at the specific version marked by `y0 = mark_version(y)`**, saved from **the same
    iteration in the forward pass**. (If directly writing staged code, it is `MarkVersion('y0', y)`). In
    other words, after AD, the position of `mark_version` and the dynamic loop iterator together makes up
    the actual version number for the tape.
    4. Build the AST with `pop_ast_and_user_grads` instead of `pop_ast`. An extra list will be returned
    together with the AST, which you need to pass as `grad`'s `user_bwds` argument. This list records
    the forward-to-backward relation of the nodes.

    Parameters
    ----------
    stmt_range: Optional[StmtRange]
        The range in the original program that we are setting custom gradient for
    args: Sequence[VarRef]
        (Positional variadic) Mapping from original variables to gradient variables
    '''

    def __init__(self, *args: Sequence[VarRef], **kvs):
        self.ori_vars = args
        self.body = None
        self.grad_defs = []
        if 'stmt_range' in kvs:
            stmt_range = kvs['stmt_range']
            if isinstance(stmt_range, StmtRange):
                self.begin_id, self.end_id = stmt_range.make()
            else:
                raise TypeError(
                    "`stmt_range` should be a `StmtRange` for `UserGrad`")
            del kvs['stmt_range']
        else:
            self.begin_id = self.end_id = ctx_stack.get_last_stmt_id()
        for key in kvs:
            raise TypeError(f"Unrecognized parameter `{key}` of `UserGrad`")

    def __enter__(self):
        # Make `VarDef` scopes for the gradients
        grad_vars = []
        for ori_var in self.ori_vars:
            grad_def = VarDef(ori_var.name + ".grad", ori_var.full_shape,
                              ori_var.dtype, "cache")
            grad_vars.append(grad_def.__enter__())
            self.grad_defs.append(grad_def)

        # Make a context, which is used for popping out the body we need
        ctx_stack.push()

        return grad_vars

    def __exit__(self, exc_type, exc_value, traceback):
        # Pop out the body we need
        self.body = ctx_stack.pop().make_stmt()

        # Although we are discarding the gradient `VarDef` scopes, we still need to close
        # them, to restore ctx_stack. After that, we pop out the `VarDef` statement
        for grad_def in reversed(self.grad_defs):
            grad_def.__exit__(exc_type, exc_value, traceback)
        if exc_value is not None:
            # Do not generate an AST node
            return False  # Do not suppress the exception
        ctx_stack.top().stmt_seq.pop()

        # Record the body to context
        ctx_stack.user_grads.append(
            ffi.UserBwd(self.begin_id, self.end_id, self.body))


class Func(ffi.Func):

    def __init__(self,
                 name,
                 params,
                 returns,
                 body,
                 closure={},
                 custom_callback=None,
                 user_grads=[]):
        super().__init__(
            name, params,
            list(map(lambda x: (x[0], ffi.DataType(x[1])), returns)), body,
            closure)
        self.custom_callback = custom_callback
        self.user_grads = user_grads

        # Mimic a Python function
        self.__name__ = name

    def __call__(self, *args, **kvs):
        return self.custom_callback(*args, **kvs)
