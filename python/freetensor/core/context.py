'''
Facility to pick statements to build an AST

Classes and functions in this module are internally used by `transformer` to construct ASTs.
They are also used by some internal tests. API of these classes and functions are subject to changes.
End users are encouraged to use `transformer`, instead of this module.
'''

import sys
from typing import Sequence, Tuple, Any, Optional

import freetensor_ffi as ffi


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
        from .expr import ceildiv
        self.append_stmt(
            ffi.makeFor(
                nid,
                iter_var,
                begin,
                end,
                step,
                ceildiv(end - begin, step),
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
