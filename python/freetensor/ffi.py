# Import freetensor_ffi.*.so from either PYTHONPATH (run in tree) or the current directory (when installed)

try:
    from freetensor_ffi import *
except ImportError:
    from .freetensor_ffi import *
