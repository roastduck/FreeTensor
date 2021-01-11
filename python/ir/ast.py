from typing import Sequence, Tuple

import ffi
from ffi import AccessType, DataType

class Context:
	def __init__(self):
		self.stmt_seq = []
		self.lastIf = None # To handle else case
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

	def setNextNid(self, nid: str):
		self.nextNid = nid

	def getNextNid(self):
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

ctx_stack = ContextStack()

''' Get AST and reset context '''
def pop_ast():
	ret = ctx_stack.pop().make_stmt()
	ctx_stack.reset()
	return ret

class Var:
	def __init__(self, name: str):
		self.var = name

	def __getitem__(self, key):
		if type(key) is not tuple and type(key) is not list:
			key = (key,)
		return ffi.makeLoad(self.var, key)

	def __setitem__(self, key, value):
		if type(key) is not tuple and type(key) is not list:
			key = (key,)
		top = ctx_stack.top()
		top.append_stmt(ffi.makeStore(top.getNextNid(), self.var, key, value))

class _VarDef:
	def __init__(self, name: str, shape: Sequence, dtype: DataType, atype: AccessType):
		self.name = name
		self.shape = shape
		self.dtype = dtype
		self.atype = atype

	def __enter__(self):
		ctx_stack.push()
		return Var(self.name)

	def __exit__(self, exc_type, exc_value, traceback):
		buf = ffi.Buffer(ffi.Tensor(self.shape, self.dtype), self.atype)
		body = ctx_stack.pop().make_stmt()
		top = ctx_stack.top()
		top.append_stmt(ffi.makeVarDef(top.getNextNid(), self.name, buf, body))

class _VarsDef:
	def __init__(self, defs: Tuple[str, Sequence, DataType, AccessType]):
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
		nid = top.getNextNid()
		if self.nid != "":
			nid = self.nid
		top.append_stmt(ffi.makeFor(nid, self.iter_var, self.begin, self.end, body))

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

''' Mark the ID of the following statement '''
def MarkNid(nid: str):
	ctx_stack.top().setNextNid(nid)

class NamedScope:
	def __init__(self, nid: str):
		self.nid = nid

	def __enter__(self):
		ctx_stack.push()

	def __exit__(self, exc_type, exc_value, traceback):
		body = ctx_stack.pop().make_stmt(self.nid)
		ctx_stack.top().append_stmt(body)

def Any():
	ctx_stack.top().append_stmt(ffi.makeAny())

def min(lhs, rhs):
	return ffi.makeMin(lhs, rhs)

def max(lhs, rhs):
	return ffi.makeMax(lhs, rhs)

