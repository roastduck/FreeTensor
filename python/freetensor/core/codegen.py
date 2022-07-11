import freetensor_ffi as ffi
import sys
import functools

from . import config
from .. import debug

from typing import Optional


class NativeCode:

    def __init__(self, func, code, target):
        self.func = func
        self.code = code
        self.target = target

    def __str__(self):
        return self.code


def codegen(ast=None,
            target: Optional[ffi.Target] = None,
            verbose: Optional[bool] = None) -> NativeCode:
    '''
    Generate native code

    Parameters
    ----------
    ast : AST
        The AST to be lowered. It must includes function signature to determine
        parameters and return values. If not specified, a partial function is
        returned, which can be used as a decorator
    target : Target (Optional)
        The target architecture. If omitted, use the default one in config
    '''

    if ast is not None:

        if target is None:
            target = config.default_target()
        raw_code = ffi.code_gen(ast, target)
        if verbose:
            print(debug.with_line_no(raw_code), file=sys.stderr)

        return NativeCode(ast, raw_code, target)

    else:

        f = codegen
        if target is not None:
            f = functools.partial(f, target=target)
        if verbose is not None:
            f = functools.partial(f, verbose=verbose)
        return f
