import ffi
import sys
import functools

from . import config
from .. import debug

from typing import Optional


class NativeCode:

    def __init__(self, func, code):
        self.func = func
        self.code = code

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

        if target.type() == ffi.TargetType.CPU:
            raw_code = ffi.code_gen_cpu(ast)
        elif target.type() == ffi.TargetType.GPU:
            raw_code = ffi.code_gen_cuda(ast)
        else:
            assert False, "Unrecognized target %s" % target

        if verbose:
            print(debug.with_line_no(raw_code), file=sys.stderr)

        return NativeCode(ast, raw_code)

    else:

        f = codegen
        if target is not None:
            f = functools.partial(f, target=target)
        if verbose is not None:
            f = functools.partial(f, verbose=verbose)
        return f
