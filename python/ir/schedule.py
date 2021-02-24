import ffi

from .utils import *

class Schedule(ffi.Schedule):
	def __init__(self, ast):
		super(Schedule, self).__init__(ast)

	def cache(self, stmt, var, mtype):
		return super(Schedule, self).cache(stmt, var, parseMType(mtype))

