__all__ = ['transform', 'inline']

import sys
import inspect
import functools

import freetensor_ffi as ffi
from .stmt import VarRef
from .func import Func
from .frontend import lang_overload, staged_callable, LifetimeScope, dynamic_range
from .context import pop_ast_and_user_grads
from .staging import StagingError, TransformError
from .meta import add_outputting


def _prepare_extra_locals(default_dynamic_range):
    extra_locals = {'__ft__': sys.modules['freetensor']}
    if default_dynamic_range:
        extra_locals['range'] = dynamic_range
    return extra_locals


def transform(func=None, default_dynamic_range=True, verbose: int = 0):
    '''
    Transform a user function to an AST

    Parameters
    ----------
    func : Python function
        The user function to transform. If not specified, a partial function will
        be returend, which can be used as a decorator
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : int
        0 = print nothing. 1 = print the resulting AST. 2 = 1 + print the generated
        Python code that is used for transforming
    '''

    if func is None:
        return functools.partial(transform,
                                 default_dynamic_range=default_dynamic_range,
                                 verbose=verbose)

    if verbose is None:
        verbose = 0

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    params = list(inspect.signature(func).parameters)

    staging_func = lang_overload.into_staging(func,
                                              extra_locals,
                                              verbose=verbose >= 2)

    try:
        # Initialize lang_overload to prepare for staging.
        lang_overload.__init__()
        # Create a new scope for the function
        with LifetimeScope():
            # Run staging function with the tensor program arguments' names as parameters
            returns = staging_func(*params,
                                   __freetensor_transform_outermost__=True)
            # Check returned vardefs (if any)
            if isinstance(returns, VarRef):
                returns = [returns]
            elif isinstance(returns, tuple):
                for ret in returns:
                    if not isinstance(ret, VarRef):
                        raise lang_overload.error(
                            f'Illegal return at top level, need to be a `VarRef` or a tuple of'
                            f' `VarRef`s, got {ret}')
                returns = list(returns)
            elif returns is None:
                returns = []
            else:
                raise lang_overload.error(
                    f'Illegal return at top level, need to be a `VarRef` or a tuple of `VarRef`s,'
                    f' got {returns}')
            # Set returned vardefs' access type to inout/output according to whether it was an input
            for ret in returns:
                ret.vardef.set_atype(add_outputting(ret.vardef.atype))
            returns = [
                ffi.FuncRet(ret.vardef.name, ret.vardef.dtype)
                for ret in returns
            ]

            # Set closure; they are from captured Arrays.
            closure = lang_overload.closure
    except StagingError:
        raise
    except TransformError:
        raise
    except Exception as e:
        raise lang_overload.error('Exception occurred in staging') from e
    finally:
        # Despite whether the exception is raised, we need to clean up the ctx_stack
        staged_ast, user_grads = pop_ast_and_user_grads()

    staged = Func(func.__name__,
                  params + list(closure.keys()),
                  returns,
                  staged_ast,
                  closure,
                  user_grads=user_grads)

    if verbose >= 1:
        print("The transformed AST is:", file=sys.stderr)
        print(staged, file=sys.stderr)
        print(file=sys.stderr)

    return staged


def inline(func=None,
           src=None,
           fallback=None,
           default_dynamic_range=True,
           verbose=False):
    '''
    Enable a user function to be called by a transformed function at run time

    Parameters
    ----------
    func : Python function
        The user function
    src : str (Optional)
        The source code of `func`. This parameter is only required if the source
        code cannot be get automatically, e.g., if `func` is generated from an
        `exec`
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : bool
        True to print the generated Python code that is used for transforming
    '''

    if func is None:
        return functools.partial(inline,
                                 src=src,
                                 fallback=fallback,
                                 default_dynamic_range=default_dynamic_range,
                                 verbose=verbose)

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    # Do not initialize lang_overload here, since `into_staging` does not use the context.
    # Keep the context as-is to support adding new inline functions during transforming.
    # Such a case occurs when a transformed function dynamically imports a new inline.
    transformed = lang_overload.into_staging(func,
                                             extra_locals,
                                             src,
                                             verbose=verbose)

    return functools.wraps(func)(staged_callable(transformed, fallback or func))
