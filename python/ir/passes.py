from typing import Optional

import ffi

from ffi import simplify_pass
from ffi import sink_var
from ffi import shrink_var
from ffi import shrink_for
from ffi import merge_if
from ffi import hoist_if
from ffi import seperate_tail
from ffi import make_reduction
from ffi import make_atomic
from ffi import remove_writes
from ffi import make_const_shape
from ffi import make_1d_var
from ffi import use_builtin_div
from ffi import gpu_make_sync
from ffi import gpu_correct_shared
from ffi import gpu_normalize_threads

def lower(ast, target: Optional[ffi.Target]=None):
    ast = simplify_pass(ast)
    ast = sink_var(ast)
    ast = shrink_var(ast)
    ast = hoist_if(ast)
    ast = merge_if(ast)
    ast = seperate_tail(ast)
    ast = remove_writes(ast) # After seperate_tail
    ast = shrink_for(ast) # After seperate_tail and remove_writes
    ast = make_atomic(ast)

    # After all general analysis
    ast = use_builtin_div(ast)

    if target is None:
        return ast

    if target.type() == ffi.TargetType.GPU:
        ast = gpu_make_sync(ast)

        # TODO: Support dynamic shared memory size, but the size should be determined
        # outside of kernels
        ast = make_const_shape(ast, [ffi.MemType.GPUShared, ffi.MemType.GPULocal])
        ast = gpu_correct_shared(ast)

        ast = make_1d_var(ast)

        # After gpu_make_sync and gpu_correct_shared. Otherwise, these 2 passes
        # cannot get the right thread info
        ast = gpu_normalize_threads(ast)

    return ast

