__all__ = [
    'DataType', 'up_cast', 'neutral_val', 'is_float', 'is_int', 'is_bool',
    'to_numpy_dtype', 'to_torch_dtype', 'MemType', 'is_writable',
    'is_inputting', 'is_outputting', 'add_outputting', 'remove_outputting',
    'zero_value', 'one_value', 'min_value', 'max_value', 'same_mtype',
    'AccessType'
]

from ..ffi import (DataType, up_cast, neutral_val, is_float, is_int, is_bool,
                   MemType, is_writable, is_inputting, is_outputting,
                   add_outputting, remove_outputting, AccessType)
import numpy as np


def to_numpy_dtype(dtype):
    dtype = DataType(dtype)
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int32':
        return np.int32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'bool':
        return np.bool_
    else:
        assert False, "Cannot convert data type %s to a NumPy type" % dtype


def to_torch_dtype(dtype):
    import torch  # torch is optional dependency
    dtype = DataType(dtype)
    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float64':
        return torch.float64
    elif dtype == 'int32':
        return torch.int32
    elif dtype == 'int64':
        return torch.int64
    elif dtype == 'bool':
        return torch.bool
    else:
        assert False, "Cannot convert data type %s to a PyTorch type" % dtype


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


def one_value(dtype):
    dtype = DataType(dtype)
    if is_float(dtype):
        return 1.
    elif is_int(dtype):
        return 1
    elif is_bool(dtype):
        return True
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
