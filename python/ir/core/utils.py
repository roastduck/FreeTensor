import ffi
from ffi import (ID, AccessType, MemType, DataType, OpenMPScope,
                 CUDAStreamScope, CUDAScope, CUDAScopeLevel, CUDAScopeDim)


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
