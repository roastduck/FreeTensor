import ffi

from . import config

from typing import Optional


def codegen(ast, target: Optional[ffi.Target] = None):
    '''
    Generate native code

    Parameters
    ----------
    ast : AST
        The AST to be lowered. It must includes function signature to determine
        parameters and return values
    target : Target (Optional)
        The target architecture. If omitted, use the default one in config
    '''

    if target is None:
        target = config.default_target()

    if target.type() == ffi.TargetType.CPU:
        return ffi.code_gen_cpu(ast)
    elif target.type() == ffi.TargetType.GPU:
        return ffi.code_gen_cuda(ast)
    else:
        assert False, "Unrecognized target %s" % target
