import ffi
from ffi import up_cast, neutral_val, is_float, DataType, MemType
from .utils import *


def min_value(dtype):
    dtype = DataType(dtype)
    if is_float(dtype):
        return -float("inf")
    elif dtype == DataType("int32"):
        return 0x80000000
    elif dtype == DataType("int64"):
        return 0x8000000000000000
    else:
        assert False, "Unrecognized data type %s" % dtype


def max_value(dtype):
    dtype = DataType(dtype)
    if is_float(dtype):
        return float("inf")
    elif dtype == DataType("int32"):
        return 0x7fffffff
    elif dtype == DataType("int64"):
        return 0x7fffffffffffffff
    else:
        assert False, "Unrecognized data type %s" % dtype


def same_mtype(lhs, rhs):
    lhs = MemType(lhs)
    rhs = MemType(rhs)
    assert lhs == rhs or lhs == MemType("byvalue") or rhs == MemType(
        "byvalue"), "Variables must be on the same memory"
    return lhs
