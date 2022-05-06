from typing import Optional, Sequence

import ffi

from ffi import grad
from ffi import GradTapeMode
from ffi import output_intermediates
from ffi import scalar_prop_const
from ffi import tensor_prop_const
from ffi import prop_one_time_use
from ffi import simplify
from ffi import z3_simplify
from ffi import sink_var
from ffi import shrink_var
from ffi import shrink_for
from ffi import merge_and_hoist_if
from ffi import make_reduction
from ffi import make_parallel_reduction
from ffi import remove_writes
from ffi import remove_dead_var
from ffi import make_const_shape
from ffi import make_1d_var
from ffi import use_builtin_div
from ffi import hoist_var_over_stmt_seq
from ffi import cpu_lower_parallel_reduction
from ffi import gpu_lower_parallel_reduction
from ffi import gpu_make_sync
from ffi import gpu_multiplex_buffers
from ffi import gpu_simplex_buffers
from ffi import gpu_normalize_threads
from ffi import gpu_lower_vector
from ffi import lower


def lower(ast,
          target: Optional[ffi.Target] = None,
          skip_passes: Sequence[str] = []):
    '''
    Lower an AST using a series of passes

    Parameters
    ----------
    ast : AST
        The AST to be lowered. Can be a `Func` or a `Stmt`
    target : Target (Optional)
        Lower the AST to a target with target-specific passes, then the AST can
        be used for codegen. If not set, use the default Target in Config
    skip_passes : Sequence[str]
        Skip some pass for testing or debugging. Names in `skipPasses` are in
        underscore_style, as in Python. Please note that some passes will not be
        skipped even specified in these parameter, because they are indirectly
        called in some other passes
    '''

    return ffi.lower(ast, target, set(skip_passes))
