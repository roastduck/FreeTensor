import ffi
from ffi import AccessType, MemType, DataType

def toId(node):
	if type(node) is str:
		return node
	if isinstance(node, ffi.Cursor):
		return node.nid()
	if isinstance(node, ffi.Stmt):
		return node.nid
	assert False, "%s is not a valid statement" % node

def parseDType(dtype):
	if type(dtype) is DataType:
		return dtype
	elif type(dtype) is str:
		if dtype.lower() == "float32":
			return DataType.Float32
		elif dtype.lower() == "int32":
			return DataType.Int32
	assert False, "Unrecognized data type %s" % dtype

def parseAType(atype):
	if type(atype) is AccessType:
		return atype
	elif type(atype) is str:
		if atype.lower() == "input":
			return AccessType.Input
		elif atype.lower() == "output":
			return AccessType.Output
		elif atype.lower() == "inout":
			return AccessType.InOut
		elif atype.lower() == "cache":
			return AccessType.Cache
	assert False, "Unrecognized access type %s" % atype

def parseMType(mtype):
	if type(mtype) is MemType:
		return mtype
	elif type(mtype) is str:
		if mtype.lower() == "byvalue":
			return MemType.ByValue
		elif mtype.lower() == "cpu":
			return MemType.CPU
		elif mtype.lower() == "gpu/global":
			return MemType.GPUGlobal
		elif mtype.lower() == "gpu/shared":
			return MemType.GPUShared
		elif mtype.lower() == "gpu/local":
			return MemType.GPULocal
	assert False, "Unrecognized memory type %s" % mtype

