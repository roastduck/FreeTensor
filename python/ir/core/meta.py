import ffi
from ffi import up_cast
from .utils import *


def min_value(dtype):
    dtype = parseDType(dtype)
    if dtype == DataType.Float32:
        return -float("inf")
    elif dtype == DataType.Int32:
        return 0x80000000
    else:
        assert False, "Unrecognized data type %s" % dtype


def max_value(dtype):
    dtype = parseDType(dtype)
    if dtype == DataType.Float32:
        return float("inf")
    elif dtype == DataType.Int32:
        return 0x7fffffff
    else:
        assert False, "Unrecognized data type %s" % dtype
