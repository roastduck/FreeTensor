'''
Facility to pick statements to build an AST

Classes and functions in this module are internally used by `transformer` to construct ASTs.
They are also used by some internal tests. API of these classes and functions are subject to changes.
End users are encouraged to use `transformer`, instead of this module.
'''

import sys
from typing import List, Sequence, Optional

import freetensor_ffi as ffi


class Context:

    def __init__(self, caller_metadata: Optional[ffi.Metadata] = None):
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
        if len(labels) > 0:
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

    def append_if_then_stmt(self, cond, body: ffi.Stmt):
        next_metadata = self.get_metadata()
        self.append_stmt(ffi.makeIf(cond, body, next_metadata))
        self.last_if = (next_metadata, cond, body)

    def append_if_else_stmt(self, elseCase: ffi.Stmt):
        metadata, cond, thenCase = self.last_if
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
        if metadata == None:
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
        if len(self.stmt_seq) == 1 and metadata == None:
            return self.stmt_seq[0]
        else:
            return ffi.makeStmtSeq(self.stmt_seq, metadata or None)


class ContextStack:

    def __init__(self):
        self.reset()

    def reset(self):
        self.stack = [Context()]

    def top(self) -> Context:
        return self.stack[-1]

    def push(self):
        self.stack.append(Context(self.top().caller_metadata))

    def pop(self):
        return self.stack.pop()

    def get_stack(self) -> List[Context]:
        return self.stack

    def set_stack(self, stack: List[Context]):
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
