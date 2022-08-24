from typing import Optional, Sequence
import functools

from . import config

import freetensor_ffi as ffi

from freetensor_ffi import lower
from freetensor_ffi import scalar_prop_const
from freetensor_ffi import tensor_prop_const
from freetensor_ffi import prop_one_time_use
from freetensor_ffi import simplify
from freetensor_ffi import z3_simplify
from freetensor_ffi import sink_var
from freetensor_ffi import shrink_var
from freetensor_ffi import shrink_for
from freetensor_ffi import merge_and_hoist_if
from freetensor_ffi import make_reduction
from freetensor_ffi import make_parallel_reduction
from freetensor_ffi import remove_writes
from freetensor_ffi import remove_dead_var
from freetensor_ffi import make_heap_alloc
from freetensor_ffi import make_const_shape
from freetensor_ffi import use_builtin_div
from freetensor_ffi import hoist_var_over_stmt_seq
from freetensor_ffi import cpu_lower_parallel_reduction

if config.with_cuda():
    from freetensor_ffi import gpu_set_bsp_scope
    from freetensor_ffi import gpu_lower_parallel_reduction
    from freetensor_ffi import gpu_make_sync
    from freetensor_ffi import gpu_multiplex_buffers
    from freetensor_ffi import gpu_simplex_buffers
    from freetensor_ffi import gpu_normalize_threads
    from freetensor_ffi import gpu_lower_vector


def lower(ast=None,
          target: Optional[ffi.Target] = None,
          skip_passes: Optional[Sequence[str]] = None,
          verbose: Optional[int] = None):
    '''
    Lower an AST using a series of passes

    Parameters
    ----------
    ast : AST
        The AST to be lowered. Can be a `Func` or a `Stmt`. If not specified, a
        partial function of `lower` will be returned, which can be used as a
        decorator
    target : Target (Optional)
        Lower the AST to a target with target-specific passes, then the AST can
        be used for codegen. If not set, use the default Target in Config
    skip_passes : Sequence[str] (Optional)
        Skip some pass for testing or debugging. Names in `skipPasses` are in
        underscore_style, as in Python. Please note that some passes will not be
        skipped even specified in these parameter, because they are indirectly
        called in some other passes
    verbose : int (Optional)
        0 = print nothing. 1 = print the lowered AST. 2 = print AST after every
        single passes
        '''

    if ast is not None:
        return ffi.lower(ast, target,
                         set() if skip_passes is None else set(skip_passes),
                         0 if verbose is None else verbose)
    else:
        _lower = lower
        if target is not None:
            _lower = functools.partial(_lower, target=target)
        if skip_passes is not None:
            _lower = functools.partial(_lower, skip_passes=skip_passes)
        if verbose is not None:
            _lower = functools.partial(_lower, verbose=verbose)
        return _lower
