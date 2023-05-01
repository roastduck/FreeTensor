__all__ = ['transform', 'inline']

from typing import Mapping, Callable, Any
import sys
import inspect
import functools

import freetensor_ffi as ffi
from . import config
from .expr import UndeclaredParam
from .stmt import VarRef
from .func import Func
from .frontend import lang_overload, staged_callable, LifetimeScope, dynamic_range
from .context import pop_ast_and_user_grads, ctx_stack
from .staging import StagingError, TransformError
from .jit import JIT, JITTemplate
from .meta import add_outputting
from .utils import as_decorator


def _prepare_extra_locals(default_dynamic_range):
    extra_locals = {'__ft__': sys.modules['freetensor']}
    if default_dynamic_range:
        extra_locals['range'] = dynamic_range
    return extra_locals


@as_decorator
def transform(func=None,
              default_dynamic_range=True,
              bind: Mapping[str, Any] = {},
              jit_cache: Callable[Callable, Callable] = functools.cache,
              target: ffi.Target = None,
              verbose: int = 0):
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
    bind : Mapping[str, Any]
        Bind some parameters to specific values before transformations. Accpeting a
        parameter-name-to-value dict.
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    target : Target
        If not None, set `config.default_target` when transforming. This affects the
        default memory type use to create variables from `Var`, `empty` and etc.
    verbose : int
        0 = print nothing. 1 = print the resulting AST. 2 = 1 + print the generated
        Python code that is used for transforming

    Returns
    -------
    Func or JITTemplate
        Return a Func for an AST if there is no JIT parameters. Return a JITTemplate
        that generates a Func if there is at least one
    '''

    # Note for JIT: `transform` should respect the default target when it is called
    if target is None:
        target = config.default_target()

    params = inspect.signature(func).parameters
    param_names = []
    jit_param_names = []
    for name, param in params.items():
        if name not in bind:
            param_names.append(name)
            is_jit = False
            try:
                is_jit = issubclass(param.annotation, JIT)
            except TypeError:
                pass  # issubclass will raise if the first argument is not a class
            if is_jit:
                jit_param_names.append(name)

    if len(jit_param_names) > 0:

        class TransformTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                new_bind = {
                    name: value
                    for name, value in zip(self.jit_param_names, jit_args)
                }
                return transform(func,
                                 default_dynamic_range=default_dynamic_range,
                                 bind={
                                     **bind,
                                     **new_bind
                                 },
                                 target=target,
                                 verbose=verbose)

            def __call__(self, *args, **kvs):
                '''
                Enable invoking an AST template in another function being transformed, via
                `inlined_invoke`
                '''

                self.args = args
                self.kvs = kvs
                return self.instantiate_and_call(*args, **kvs)

            @property
            def backward(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .backward")
                # TODO: Here we re-instantiate. Change to another implementation if the overhead
                # is too much
                return self.instantiate(*self.args, **self.kvs).backward

            @property
            def input_name_to_gradient_name(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .input_name_to_gradient_name")
                return self.instantiate(*self.args,
                                        **self.kvs).input_name_to_gradient_name

            @property
            def output_name_to_gradient_name(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .output_name_to_gradient_name")
                return self.instantiate(*self.args,
                                        **self.kvs).output_name_to_gradient_name

        template = TransformTemplate(params, jit_param_names)
        template.__name__ = func.__name__
        if func.__doc__ is not None:
            template.__doc__ = func.__doc__
        return template

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    with target, lang_overload, ctx_stack:
        staging_func = lang_overload.into_staging(func,
                                                  extra_locals,
                                                  verbose=verbose >= 2)
        staging_func = functools.partial(staging_func, **bind)

        try:
            # Create a new scope for the function
            with LifetimeScope():
                # Each argument is passed by default an `UndeclaredParam`, until it is declared
                returns = staging_func(**{
                    name: UndeclaredParam(name) for name in param_names
                },
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
                  param_names + list(closure.keys()),
                  returns,
                  staged_ast,
                  closure,
                  user_grads=user_grads)

    if verbose >= 1:
        print("The transformed AST is:", file=sys.stderr)
        print(staged, file=sys.stderr)
        print(file=sys.stderr)

    return staged


@as_decorator
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

    extra_locals = _prepare_extra_locals(default_dynamic_range)

    # No need to push lang_overload here, because `into_staging` is a function on
    # `StagedOverloadStack`, instead of `StagedOverload`. Since the staged code is run in
    # `transform` rather than in `inline`, this design is to ensure the overload used here
    # in `inline` should be the same object with that is pushed from `transform`.
    transformed = lang_overload.into_staging(func,
                                             extra_locals,
                                             src,
                                             verbose=verbose)

    return functools.wraps(func)(staged_callable(transformed, fallback or func))
