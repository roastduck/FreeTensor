from typing import Sequence

import ffi
from ffi import AccessType, DataType

class Context:
	def __init__(self):
		self.stmt_seq = []

	def append_stmt(self, stmt: ffi.Stmt):
		self.stmt_seq.append(stmt)

	def make_stmt(self):
		if len(self.stmt_seq) == 1:
			return self.stmt_seq[0]
		else:
			return ffi.makeStmtSeq(self.stmt_seq)

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
		self.name = name

	def __getitem__(self, key):
		if type(key) is not tuple and type(key) is not list:
			key = (key,)
		return ffi.makeLoad(ffi.makeVar(self.name), key)

	def __setitem__(self, key, value):
		if type(key) is not tuple and type(key) is not list:
			key = (key,)
		ctx_stack.top().append_stmt(ffi.makeStore(ffi.makeVar(self.name), key, value))

class VarDef:
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
		ctx_stack.top().append_stmt(ffi.makeVarDef(self.name, buf, body))

