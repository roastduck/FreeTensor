from ffi import AccessType, MemType, DataType

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
		if mtype.lower() == "cpu":
			return MemType.CPU
		elif mtype.lower() == "gpuglobal":
			return MemType.GPUGlobal
		elif mtype.lower() == "gpushared":
			return MemType.GPUShared
		elif mtype.lower() == "gpulocal":
			return MemType.GPULocal
	assert False, "Unrecognized memory type %s" % mtype

