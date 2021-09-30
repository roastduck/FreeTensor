from typing import Optional

import ffi

from ffi import grad
from ffi import simplify_pass
from ffi import sink_var
from ffi import shrink_var
from ffi import shrink_for
from ffi import merge_and_hoist_if
from ffi import make_reduction
from ffi import make_atomic
from ffi import remove_writes
from ffi import remove_dead_var
from ffi import make_const_shape
from ffi import make_1d_var
from ffi import use_builtin_div
from ffi import gpu_make_sync
from ffi import gpu_correct_shared_and_global
from ffi import gpu_normalize_threads
from ffi import gpu_lower_vector


def lower(ast, target: Optional[ffi.Target] = None):
    return ffi.lower(ast, target)
