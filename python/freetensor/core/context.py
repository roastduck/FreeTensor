'''
Facility to pick statements to build an AST

Classes and functions in this module are internally used by `transformer` to construct ASTs.
They are also used by some internal tests. API of these classes and functions are subject to changes.
End users are encouraged to use `transformer`, instead of this module.
'''

import sys
from typing import List, Sequence, Optional, Callable

import freetensor_ffi as ffi


class Context:

    def __init__(self,
                 parent_context_stack,
                 caller_metadata: Optional[ffi.Metadata] = None):
        self.parent_context_stack = parent_context_stack
        self.stmt_seq = []

        self.next_labels = []
        self.next_location = None
        self.caller_metadata = caller_metadata

        self.last_if = None  # To handle else case
        self.next_no_deps = []
        self.next_prefer_libs = False

    def get_metadata(self, labels=None):
        if labels is None:
            labels = self.next_labels
        if len(labels) > 0 or self.caller_metadata is not None:
            return ffi.SourceMetadata(list(labels), self.next_location,
                                      self.caller_metadata)
        else:
            return None

    def clear_metadata(self):
        self.next_labels = []
        self.next_location = None

    def append_stmt(self, stmt: ffi.Stmt):
        self.stmt_seq.append(stmt)

        self.clear_metadata()

        self.last_if = None
        self.next_no_deps = []
        self.next_prefer_libs = False

        for callback in self.parent_context_stack.next_append_stmt_callbacks:
            callback(stmt)
        self.parent_context_stack.next_append_stmt_callbacks = []

    def append_if_then_stmt(self, cond, body: ffi.Stmt):
        next_metadata = self.get_metadata()
        last_if = (next_metadata, cond, body,
                   list(self.parent_context_stack.next_append_stmt_callbacks))
        self.append_stmt(ffi.makeIf(cond, body, next_metadata))
        self.last_if = last_if

    def append_if_else_stmt(self, elseCase: ffi.Stmt):
        metadata, cond, thenCase, self.parent_context_stack.next_append_stmt_callbacks = self.last_if
        self.stmt_seq.pop()
        self.append_stmt(ffi.makeIf(cond, thenCase, elseCase, metadata))

    def append_for_stmt(self,
                        iter_var,
                        begin,
                        end,
                        step,
                        body,
                        metadata: ffi.Metadata = None,
                        no_deps: Optional[Sequence[str]] = None,
                        prefer_libs: Optional[bool] = None):
        if metadata is None:
            metadata = self.get_metadata()
        if no_deps is None:
            no_deps = self.next_no_deps
        if prefer_libs is None:
            prefer_libs = self.next_prefer_libs
        for_property = ffi.ForProperty(). \
            with_no_deps(no_deps).with_prefer_libs(prefer_libs)

        from .expr import ceildiv
        self.append_stmt(
            ffi.makeFor(iter_var, begin, end, step, ceildiv(end - begin, step),
                        for_property, body, metadata))

    def add_label(self, label):
        self.next_labels.append(label)

    def set_next_location(self, file, line):
        self.next_location = (file, line)

    def set_caller_metadata(self, metadata):
        self.caller_metadata = metadata

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

    def make_stmt(self, metadata: ffi.Metadata = None):
        if len(self.stmt_seq) == 1 and metadata is None:
            return self.stmt_seq[0]
        else:
            return ffi.makeStmtSeq(self.stmt_seq, metadata or None)


class ContextStack:

    def __init__(self):
        self.reset()

    def reset(self):
        self.stack = [Context(self)]
        self.user_grads = []

        # fn(ffi.Stmt) invoked for the next `append_stmt`
        self.next_append_stmt_callbacks = []

    def top(self) -> Context:
        return self.stack[-1]

    def push(self):
        self.stack.append(Context(self, self.top().caller_metadata))

    def pop(self):
        return self.stack.pop()

    def get_stack(self) -> List[Context]:
        return self.stack

    def set_stack(self, stack: List[Context]):
        self.stack = stack

    def get_last_stmt_id(self):
        '''
        Can be used inside the staged code, to get the ID of the immediately preceding statement
        '''
        for ctx in reversed(self.stack):
            if len(ctx.stmt_seq) > 0:
                return ctx.stmt_seq[-1].id
        raise ft.InvalidProgram("There is no statement yet")

    def add_next_append_stmt_callback(self, callback: Callable[[ffi.Stmt],
                                                               None]):
        '''
        Add a callback to be called with the next statement to be appended. It will be called with
        only the next statement. But for `If` statement, it can be called twice, one without "else"
        branch, and then maybe one more with "else" branch
        '''
        self.next_append_stmt_callbacks.append(callback)


ctx_stack = ContextStack()


def pop_ast(verbose: bool = False):
    """
    Get AST and reset context

    Internally used by `transformer` and tests
    """
    ret = ctx_stack.pop().make_stmt()
    ctx_stack.reset()
    if verbose:
        print("The popped AST is:", file=sys.stderr)
        print(ret, file=sys.stderr)
        print(file=sys.stderr)
    return ret


def pop_ast_and_user_grads(verbose: bool = False):
    """
    Get AST and reset context. Return an extra list for custom gradients

    Set `UserGrad` for details
    """
    ast = ctx_stack.pop().make_stmt()
    user_grads = ctx_stack.user_grads
    ctx_stack.reset()
    if verbose:
        print("The popped AST is:", file=sys.stderr)
        print(ret, file=sys.stderr)
        print(file=sys.stderr)
    return ast, user_grads


class StmtRange:
    '''
    Record a range of statement in a program, can be used for custom gradient

    Usage:

    ```
    with StmtRange() as rng:
        # Some statements
    ```

    `StmtRange` can be used interleaved with AST scopes. In these cases, you can directly call
    `__enter__` and `__exit__`. E.g.,

    ```
    rng = StmtRange()
    rng.__enter__()
    # Some statements
    with VarDef(...)  # Some scopes
        # Some other statements
        rng.__exit__(None, None, None)

    ```
    '''

    def __init__(self):
        self.begin_id = None
        self.end_id = None

    def __enter__(self):

        def callback(stmt):
            self.begin_id = stmt.id

        ctx_stack.add_next_append_stmt_callback(callback)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_id = ctx_stack.get_last_stmt_id()

    def make(self):
        if self.begin_id is None:
            raise ffi.InvalidProgram("StmtRange is not properly entered")
        if self.end_id is None:
            raise ffi.InvalidProgram("StmtRange is not properly exited")
        return self.begin_id, self.end_id
