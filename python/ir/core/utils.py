import ffi
from ffi import (ID, AccessType, MemType, DataType, OpenMPScope,
                 CUDAStreamScope, CUDAScope, CUDAScopeLevel, CUDAScopeDim)


def toId(node):
    if type(node) is ID:
        return node
    if type(node) is str:
        return node
    if isinstance(node, ffi.Stmt):
        return node.nid
    assert False, "%s is not a valid statement" % node


def parseDType(dtype):
    if type(dtype) is DataType:
        return dtype
    elif type(dtype) is str:
        if dtype.lower() == "float64":
            return DataType.Float64
        elif dtype.lower() == "float32":
            return DataType.Float32
        elif dtype.lower() == "int64":
            return DataType.Int64
        elif dtype.lower() == "int32":
            return DataType.Int32
        elif dtype.lower() == "bool":
            return DataType.Bool
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
        elif mtype.lower() == "gpu/warp":
            return MemType.GPUWarp
    assert False, "Unrecognized memory type %s" % mtype


def parseParallelScope(parallel):
    if type(parallel) is OpenMPScope:
        return parallel
    elif type(parallel) is CUDAScope:
        return parallel
    elif type(parallel) is str:
        if parallel.lower() == "openmp":
            return OpenMPScope()
        elif parallel.lower() == "cudastream":
            return CUDAStreamScope()
        elif parallel.lower() == "blockidx.x":
            return CUDAScope(CUDAScopeLevel.Block, CUDAScopeDim.X)
        elif parallel.lower() == "blockidx.y":
            return CUDAScope(CUDAScopeLevel.Block, CUDAScopeDim.Y)
        elif parallel.lower() == "blockidx.z":
            return CUDAScope(CUDAScopeLevel.Block, CUDAScopeDim.Z)
        elif parallel.lower() == "threadidx.x":
            return CUDAScope(CUDAScopeLevel.Thread, CUDAScopeDim.X)
        elif parallel.lower() == "threadidx.y":
            return CUDAScope(CUDAScopeLevel.Thread, CUDAScopeDim.Y)
        elif parallel.lower() == "threadidx.z":
            return CUDAScope(CUDAScopeLevel.Thread, CUDAScopeDim.Z)
    assert False, "Unrecognized parallel scope %s" % parallel
