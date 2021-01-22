import ffi

from .utils import *

class Schedule(ffi.Schedule):
	def __init__(self, ast):
		super(Schedule, self).__init__(ast)

	def cache_read(self, stmt, var, mtype):
		return super(Schedule, self).cache_read(stmt, var, parseMType(mtype))

	def cache_write(self, stmt, var, mtype):
		return super(Schedule, self).cache_write(stmt, var, parseMType(mtype))

