from typing import Optional, Callable
import sys
import functools

import freetensor_ffi as ffi

from . import config
from .. import debug
from .func import Func
from .enable_attach_backward import EnableAttachBackward
from .jit import JITTemplate
from .utils import as_decorator


class NativeCode(EnableAttachBackward):

    def __init__(self, func, code, target):
        super().__init__()
        self.func = func
        self.code = code
        self.target = target

    def __str__(self):
        return self.code


@as_decorator
def codegen(ast: Func = None,
            target: Optional[ffi.Target] = None,
            jit_cache: Callable[Callable, Callable] = functools.cache,
            verbose: bool = False) -> NativeCode:
    '''
    Generate native code

    Parameters
    ----------
    ast : Func
        The AST to be lowered. It must includes function signature to determine
        parameters and return values. If not specified, a partial function is
        returned, which can be used as a decorator
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    target : Target (Optional)
        The target architecture. If omitted, use the default one in config

    Returns
    -------
    NativeCode or JITTemplate
        Return a NativeCode for the generated code if there is no JIT parameters.
        Return a JITTemplate that generates a NativeCode if there is at least one
    '''

    # Note for JIT: `codegen` should respect the default target when it is called
    if target is None:
        target = config.default_target()

    if isinstance(ast, JITTemplate):

        class CodeGenTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return codegen(ast.instantiate_by_only_jit_args(*jit_args),
                               target=target,
                               verbose=verbose)

        return CodeGenTemplate(ast.params, ast.jit_param_names)

    raw_code = ffi.code_gen(ast, target)
    if verbose:
        print(debug.with_line_no(raw_code), file=sys.stderr)

    ret = NativeCode(ast, raw_code, target)
    if isinstance(ast, Func) and ast.has_backward():
        ret.attach_backward(
            codegen(ast.backward,
                    target=target,
                    jit_cache=jit_cache,
                    verbose=verbose), ast.input_name_to_gradient_name,
            ast.output_name_to_gradient_name)
    return ret
