from typing import Optional
import sys

import freetensor_ffi as ffi

from . import config
from .. import debug
from .utils import as_decorator


class NativeCode:

    def __init__(self, func, code, target):
        self.func = func
        self.code = code
        self.target = target

    def __str__(self):
        return self.code


@as_decorator
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

    if target is None:
        target = config.default_target()
    raw_code = ffi.code_gen(ast, target)
    if verbose:
        print(debug.with_line_no(raw_code), file=sys.stderr)

    return NativeCode(ast, raw_code, target)
