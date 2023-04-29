from typing import Optional, Callable, Union, Sequence
import functools

import freetensor_ffi as ffi
from freetensor_ffi import GradTapeMode

from .frontend import staged_callable
from .transform import transform
from .autograd import grad
from .schedule import Schedule, schedule
from .passes import lower
from .codegen import codegen
from .driver import Target, Device, build_binary
from .jit import JITTemplate
from .utils import as_decorator


@as_decorator
def optimize(func=None,
             schedule_callback: Optional[Callable[[Schedule], None]] = None,
             backward_schedule_callback: Callable[[Schedule], None] = None,
             target: Optional[Target] = None,
             device: Optional[Device] = None,
             default_dynamic_range: bool = True,
             jit_cache: Callable[Callable, Callable] = functools.cache,
             verbose: int = 0):
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
    backward_schedule_callback : Callable
        Specify what schedule(s) to do for the backward function, if `ast` is returned
        from AD with `attach_backward=True`. Defaults to be the same with `callback`
    target : Target (Optional)
        The target architecture. You don't have to set target if you set device
    device : Device (Optional)
        Where to run the program
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : int (Optional)
        Verbosity level. Can be 0, 1 or 2
    '''

    if target is None and device is not None:
        target = device.target()

    if not isinstance(func, ffi.AST) and not isinstance(func, JITTemplate):
        ast = transform(func,
                        default_dynamic_range=default_dynamic_range,
                        jit_cache=jit_cache,
                        verbose=verbose)
    else:
        ast = func
    ast = schedule(ast,
                   schedule_callback,
                   backward_schedule_callback,
                   jit_cache=jit_cache,
                   verbose=verbose)
    ast = lower(ast, target, jit_cache=jit_cache, verbose=verbose)
    code = codegen(ast, target, jit_cache=jit_cache, verbose=verbose)
    exe = build_binary(code, device, jit_cache=jit_cache, verbose=verbose)
    return exe


@as_decorator
def optimize_to_pytorch(
        func=None,
        tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
        forward_schedule_callback: Optional[Callable[[Schedule], None]] = None,
        backward_schedule_callback: Optional[Callable[[Schedule], None]] = None,
        target: Optional[Target] = None,
        device: Optional[Device] = None,
        default_dynamic_range: bool = True,
        verbose: int = 0):
    '''
    Compile a FreeTensor function to a PyTorch call, whose gradient can be
    recognized by PyTorch

    The compiled function will be a typical PyTorch's "function" (rather than
    a PyTorch's "module"). Technically, this means it is a wrapper function
    around a PyTorch's `Function`'s `apply` method

    Schedules (if any) must be applied to the forward function and the backward
    function separated. For this reason, currently only first-order gradient
    is supported

    Parameters
    ----------
    func : Python function or AST
        The user function to optimize. If not specified, a partial function will
        be returend, which can be used as a decorator
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    forward_schedule_callback : Callable (Optional)
        Schedule(s) to apply to the forward function
    backward_schedule_callback : Callable (Optional)
        Schedule(s) to apply to the backward function
    target : Target (Optional)
        The target architecture. You don't have to set target if you set device
    device : Device (Optional)
        Where to run the program
    default_dynamic_range : bool
        If True, the built-in range is replaced with freetensor.dynamic_range.
        Defaults to True
    verbose : int (Optional)
        Verbosity level. Can be 0, 1 or 2
    '''

    import torch

    # Transform from Python source to AST
    if not isinstance(func, ffi.AST) and not isinstance(func, JITTemplate):
        ast = transform(func,
                        default_dynamic_range=default_dynamic_range,
                        verbose=verbose)
    else:
        ast = func

    # Compile lazily because we know `requires` and `provides` only when executing. Re-compile
    # when gradient requirements changes
    saved_requires = set()
    saved_provides = set()
    cur_requires = None
    cur_provides = None

    # JIT instance for the current input
    ast_inst = None
    exe_inst = None

    def lazy_compile():
        nonlocal saved_requires, saved_provides, cur_requires, cur_provides, ast_inst, exe_inst

        if saved_requires == cur_requires and saved_provides == cur_provides:
            return
        saved_requires = cur_requires
        saved_provides = cur_provides
        if len(cur_requires) != 0:
            fwd_ast = grad(
                ast_inst,
                attach_backward=True,
                requires=saved_requires,
                provides=saved_provides,
                tapes=tapes,
                # PyTorch requires explicitly marking saved states via `save_for_backward()`
                tape_in_closure=False,
                verbose=verbose)
        else:
            fwd_ast = ast_inst
        exe_inst = optimize(
            fwd_ast,
            schedule_callback=forward_schedule_callback,
            backward_schedule_callback=backward_schedule_callback,
            target=target,
            device=device,
            default_dynamic_range=default_dynamic_range,
            verbose=verbose)

    # Generate a PyTorch Function
    class GeneratedPyTorchFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *args, **kvs):
            nonlocal cur_requires, cur_provides, ast_inst, exe_inst

            # We only get to know provided gradients of output tensors when we run `backward`,
            # but we need to run autograd and compile the program here in `forward`. We can
            # only assume gradients are provided for every output tensors, even if they are
            # unrelated to the inputs. Setting this option to True makes PyTorch generate zero
            # gradient for such outputs. (TODO: better solution?)
            ctx.set_materialize_grads(True)

            if isinstance(ast, JITTemplate):
                non_jit_args, jit_args = ast.separate_args(*args, **kvs)
                non_jit_kvs = {}
            else:
                non_jit_args = args
                non_jit_kvs = kvs

            # Gather required gradients of the inputs
            cur_requires = set()
            for param, arg in zip(ast_inst.params, non_jit_args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    cur_requires.add(param.name)
            for key, value in non_jit_kvs.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    cur_requires.add(key)

            # For the reason above, we assume gradients are provided for every
            # output tensors
            cur_provides = set()
            for ret in ast_inst.returns:
                cur_provides.add(ret.name)

            lazy_compile()

            exe_inst.set_args(*non_jit_args, **non_jit_kvs)
            exe_inst.run()
            returns = exe_inst.collect_returns(always_return_pack=True)
            returns = tuple(item.torch() for item in returns)

            # Save states for 1) all inputs and 2) all taped tensors (taped outputs are also
            # taped tensors). For taped tensors, we need to make them output tensors, so PyTorch
            # can recognize them. This is an officially recommanded trick at
            # https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html#saving-intermediate-results
            # So, please be aware that only the first part in `returns` are real return tensors
            saved_tensors = []
            for arg in non_jit_args:  # 1)
                saved_tensors.append(arg)
            for key, value in non_jit_kvs.items():  # 1)
                saved_tensors.append(value)
            for ret in returns:  # 2) and maybe other junks
                saved_tensors.append(ret)
            ctx.save_for_backward(*saved_tensors)

            return returns[0] if len(returns) == 1 else returns

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *args, **kvs):
            nonlocal ast_inst, exe_inst

            input_grad_map = exe_inst.input_name_to_gradient_name
            output_grad_map = exe_inst.output_name_to_gradient_name
            tape_rets = exe_inst.func.returns[len(ast_inst.returns):]
            saved_tensors = ctx.saved_tensors
            internal_kvs = {}
            for ret, arg in zip(ast_inst.returns, args):
                internal_kvs[output_grad_map[ret.name]] = arg
            for key, value in kvs:
                internal_kvs[output_grad_map[key]] = value
            for param, saved in zip(ast_inst.params, saved_tensors):
                # NOTE: Now we only support "input" parameters for PyTorch interface (no "inout"
                # or "output"), so we can forward all parameters. If we support "inout" or
                # "output" in the future, we need to filter only "input" parameters here
                internal_kvs[param.name] = saved
            for tape_ret, saved in zip(
                    tape_rets, saved_tensors[len(ast_inst.params) +
                                             len(ast_inst.returns):]):
                internal_kvs[tape_ret.name] = saved

            exe_inst.backward.set_args(**internal_kvs)
            exe_inst.backward.run()
            input_grads = exe_inst.backward.collect_returns(
                always_return_pack=True)

            # PyTorch requires returning gradient of inputs in their original order. If no
            # gradient is required for an input, set it to None
            if isinstance(ast, JITTemplate):
                param_names = list(ast.params)
            else:
                param_names = [param.name for param in ast.params]
            returns = tuple(
                input_grads[input_grad_map[param_name]].torch() if param_name in
                input_grad_map else None for param_name in param_names)

            return returns[0] if len(returns) == 1 else returns

    # Wrap around the PyTorch `Function`, to be a real Python "function", and remove our extra
    # tape outputs
    def generatedPyTorchFunction(*args, **kvs):
        nonlocal ast_inst

        if isinstance(ast, JITTemplate):
            ast_inst = ast.instantiate(*args, **kvs)
        else:
            ast_inst = ast

        returns = GeneratedPyTorchFunction.apply(*args, **kvs)
        returns_tuple = returns if isinstance(returns, Sequence) else (returns,)
        returns_tuple = returns_tuple[:len(ast_inst.returns)]
        return returns_tuple[0] if len(returns_tuple) == 1 else returns_tuple

    # If called inside a FreeTensor funcion, don't care about PyTorch, just inline the
    # transformed AST
    return staged_callable(ast, generatedPyTorchFunction)
