from typing import Optional, Callable
import sys
import functools
import astor
from pygments import highlight
from pygments.lexers import CppLexer, CudaLexer
from pygments.formatters import TerminalFormatter

import freetensor_ffi as ffi

from . import config
from .func import Func
from .enable_attach_backward import EnableAttachBackward
from .jit import JITTemplate
from .utils import as_decorator


class NativeCode(EnableAttachBackward, ffi.NativeCode):
    '''
    Generated native code with metadata

    NOTE: This class does not support serialization yet. If you need serialization,
    serialize the Func, and re-run codegen.
    '''

    def __init__(self, *args, **kvs):
        super().__init__(*args, **kvs)

    def __str__(self):
        params = "[" + ", ".join(map(str, self.params)) + "]"
        returns = "[" + ", ".join(map(str, self.returns)) + "]"
        code = self.code
        if config.pretty_print():
            if self.target.type == ffi.TargetType.GPU:
                lexer = CudaLexer()
            else:
                lexer = CppLexer()
            code = highlight(code, lexer,
                             TerminalFormatter(bg='dark', linenos=True))
        return (f'NativeCode(params={params}, returns={returns}):\n'
                f'---- BEGIN NATIVE CODE ----\n'
                f'{code}'
                f'---- END NATIVE CODE ----\n')

    def __contains__(self, item):
        ''' Legacy interface for testing if a string is in the code '''
        return item in self.code


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

    ret = ffi.code_gen(ast, target)
    ret = NativeCode(ret.name, ret.params, ret.returns, ret.code,
                     ret.target)  # ffi.NativeCode -> ft.NativeCode
    if verbose:
        print(ret, file=sys.stderr)

    if isinstance(ast, Func) and ast.has_backward():
        ret.attach_backward(
            codegen(ast.backward,
                    target=target,
                    jit_cache=jit_cache,
                    verbose=verbose), ast.input_name_to_gradient_name,
            ast.output_name_to_gradient_name)
    return ret
