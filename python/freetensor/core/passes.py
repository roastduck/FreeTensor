__all__ = [
    'lower', 'scalar_prop_const', 'tensor_prop_const', 'prop_one_time_use',
    'simplify', 'pb_simplify', 'z3_simplify', 'sink_var', 'shrink_var',
    'shrink_for', 'merge_and_hoist_if', 'make_reduction',
    'make_parallel_reduction', 'remove_writes', 'remove_cyclic_assign',
    'remove_dead_var', 'make_heap_alloc', 'use_builtin_div',
    'hoist_var_over_stmt_seq', 'flatten_stmt_seq',
    'cpu_lower_parallel_reduction', 'gpu_lower_parallel_reduction',
    'gpu_make_sync', 'gpu_multiplex_buffers', 'gpu_simplex_buffers',
    'gpu_normalize_threads', 'gpu_normalize_var_in_kernel', 'gpu_lower_vector'
]

from typing import Optional, Sequence, Callable
import functools

import freetensor_ffi as ffi

from freetensor_ffi import lower
from freetensor_ffi import scalar_prop_const
from freetensor_ffi import tensor_prop_const
from freetensor_ffi import prop_one_time_use
from freetensor_ffi import simplify
from freetensor_ffi import pb_simplify
from freetensor_ffi import z3_simplify
from freetensor_ffi import sink_var
from freetensor_ffi import shrink_var
from freetensor_ffi import shrink_for
from freetensor_ffi import merge_and_hoist_if
from freetensor_ffi import make_reduction
from freetensor_ffi import make_parallel_reduction
from freetensor_ffi import remove_writes
from freetensor_ffi import remove_cyclic_assign
from freetensor_ffi import remove_dead_var
from freetensor_ffi import make_heap_alloc
from freetensor_ffi import use_builtin_div
from freetensor_ffi import hoist_var_over_stmt_seq
from freetensor_ffi import flatten_stmt_seq
from freetensor_ffi import cpu_lower_parallel_reduction
from freetensor_ffi import gpu_lower_parallel_reduction
from freetensor_ffi import gpu_make_sync
from freetensor_ffi import gpu_multiplex_buffers
from freetensor_ffi import gpu_simplex_buffers
from freetensor_ffi import gpu_normalize_threads
from freetensor_ffi import gpu_normalize_var_in_kernel
from freetensor_ffi import gpu_lower_vector

from . import config
from .func import Func
from .jit import JITTemplate
from .utils import as_decorator


@as_decorator
def lower(ast=None,
          target: Optional[ffi.Target] = None,
          skip_passes: Sequence[str] = [],
          jit_cache: Callable[Callable, Callable] = functools.cache,
          verbose: int = 0):
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
        Skip some pass for testing or debugging. Names in `skip_passes` are in
        underscore_style, as in Python. Please note that some passes will not be
        skipped even specified in these parameter, because they are indirectly
        called in some other passes
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    verbose : int (Optional)
        0 = print nothing. 1 = print the lowered AST. 2 = print AST after every
        single passes

    Returns
    -------
    Func or JITTemplate
        Return a Func for an AST if there is no JIT parameters. Return a JITTemplate
        that generates a Func if there is at least one
    '''

    # Note for JIT: `lower` should respect the default target when it is called
    if target is None:
        target = config.default_target()

    if isinstance(ast, JITTemplate):

        class LowerTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return lower(ast.instantiate_by_only_jit_args(*jit_args),
                             target=target,
                             skip_passes=skip_passes,
                             verbose=verbose)

        return LowerTemplate(ast.params, ast.jit_param_names)

    ret = ffi.lower(ast, target, skip_passes=set(skip_passes), verbose=verbose)
    if isinstance(ast, Func):
        ret = Func(ret.name, ret.params, ret.returns, ret.body)
        if ast.has_backward():
            ret.attach_backward(
                lower(ast.backward,
                      target=target,
                      skip_passes=skip_passes,
                      jit_cache=jit_cache,
                      verbose=verbose), ast.input_name_to_gradient_name,
                ast.output_name_to_gradient_name)
    return ret
