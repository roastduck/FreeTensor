import functools
from typing import Optional, Callable

import ffi

from .transformer import transform
from .schedule import Schedule, schedule
from .passes import lower
from .codegen import codegen
from .driver import Target, Device, build_binary


def optimize(func=None,
             schedule_callback: Optional[Callable[[Schedule], None]] = None,
             target: Optional[Target] = None,
             device: Optional[Device] = None,
             verbose: Optional[int] = None):
    '''
    An one-click optimization from Python function to binary executable

    Usage:

    ```
    @optimize
    def f(...):
        ...
    ```

    It is equivalent to:

    ```
    @build_binary
    @codegen
    @lower
    @transform
    def f(...):
        ...
    ```

    Parameters
    ----------
    func : Python function or AST
        The user function to optimize. If not specified, a partial function will
        be returend, which can be used as a decorator
    schedule_callback : Callable (Optional)
        Schedule(s) to apply
    target : Target (Optional)
        The target architecture. You don't have to set target if you set device
    device : Device (Optional)
        Where to run the program
    verbose : int (Optional)
        Verbosity level. Can be 0, 1 or 2
    '''
    if func is not None:
        if target is None and device is not None:
            target = device.target()

        if not issubclass(type(func), ffi.AST):
            ast = transform(func, verbose=verbose, depth=2)
        else:
            ast = func
        ast = schedule(ast, schedule_callback, verbose=verbose)
        ast = lower(ast, target, verbose=verbose)
        code = codegen(ast, target, verbose=verbose)
        exe = build_binary(code, device)
        return exe

    else:
        f = optimize
        if schedule_callback is not None:
            f = functools.partial(f, schedule_callback=schedule_callback)
        if target is not None:
            f = functools.partial(f, target=target)
        if device is not None:
            f = functools.partial(f, device=device)
        if verbose is not None:
            f = functools.partial(f, verbose=verbose)
        return f
