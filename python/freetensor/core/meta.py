__all__ = [
    'DataType', 'up_cast', 'neutral_val', 'is_float', 'is_int', 'is_bool',
    'MemType', 'is_writable', 'is_inputting', 'is_outputting', 'add_outputting',
    'remove_outputting', 'zero_value', 'min_value', 'max_value', 'same_mtype',
    'AccessType'
]

from freetensor_ffi import (DataType, up_cast, neutral_val, is_float, is_int,
                            is_bool, MemType, is_writable, is_inputting,
                            is_outputting, add_outputting, remove_outputting,
                            AccessType)


def zero_value(dtype):
    dtype = DataType(dtype)
    if is_float(dtype):
        return 0.
    elif is_int(dtype):
        return 0
    elif is_bool(dtype):
        return False
    else:
        assert False, "Unrecognized data type %s" % dtype


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
    if lhs == MemType("byvalue"):
        return rhs
    if rhs == MemType("byvalue"):
        return lhs
    assert lhs == rhs, "Variables must be on the same memory"
    return lhs
