__all__ = [
    'GradTapeMode', 'grad_body', 'grad', 'grad_', 'jacrev',
    'output_all_intermediates'
]

from typing import Optional, Set, Union, Sequence, Callable
import sys
import functools

import freetensor_ffi as ffi

from freetensor_ffi import GradTapeMode
from freetensor_ffi import output_all_intermediates

from .analyze import find_stmt, find_all_stmt
from .frontend import LifetimeScope, ndim
from .transform import transform, inline
from .func import Func, FuncRet
from .stmt import VarDef
from .context import pop_ast, ctx_stack
from .frontend import lang_overload
from .param_ret_dict import Parameter, Return, ParamRetDict
from .jit import JITTemplate
from .utils import as_decorator
from .passes import sink_var


def grad_body(stmt: ffi.Stmt,
              requires: Sequence[str],
              provides: Sequence[str],
              tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
              reset_provided_grad: bool = True,
              invert: bool = True,
              user_grads: Sequence[ffi.StmtSetToUserGrad] = []):
    ''' `grad` or `grad_` on a function body (for internal tests only) '''

    req = set(requires)
    prov = set(provides)
    if type(tapes) is not GradTapeMode:
        tapes = {find_stmt(stmt, t).id for t in tapes}
    return ffi.grad_body(stmt, req, prov, tapes, reset_provided_grad, invert,
                         user_grads)


@as_decorator
def _grad_func(func,
               impl,
               requires: Sequence[Union[str, Parameter]],
               provides: Sequence[Union[str, Parameter, Return]],
               tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
               tape_in_closure: bool = True,
               reset_provided_grad: bool = True,
               invert: bool = True,
               user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
               attach_backward: bool = False,
               jit_cache: Callable[Callable, Callable] = functools.cache,
               verbose: int = 0,
               _override_func_params: Sequence[str] = None):

    if requires is None:
        raise TypeError("Parameter `requires` is required")
    if provides is None:
        raise TypeError("Parameter `provides` is required")

    if not isinstance(func, ffi.AST) and not isinstance(func, JITTemplate):
        func = transform(func, verbose=verbose)

    if isinstance(func, JITTemplate):

        if not attach_backward:
            raise KeyError(
                "JIT is only supported when attach_backward=True in `grad`"
                " or `grad_`")

        class GradTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return _grad_func(
                    func.instantiate_by_only_jit_args(*jit_args),
                    impl=impl,
                    requires=requires,
                    provides=provides,
                    tapes=tapes,
                    tape_in_closure=tape_in_closure,
                    reset_provided_grad=reset_provided_grad,
                    invert=invert,
                    user_grads=user_grads,
                    attach_backward=attach_backward,
                    verbose=verbose,
                    # Override func params to leave please for JIT parameters
                    _override_func_params=list(self.params))

        return GradTemplate(func.params, func.jit_param_names)

    if user_grads is None:
        if func.user_grads is not None:
            user_grads = func.user_grads
        else:
            user_grads = []
    req = set([])
    prov = set([])
    func_param_names = [p.name for p in func.params]
    func_return_names = [r.name for r in func.returns]
    if _override_func_params is not None:
        func_param_names = _override_func_params
    for r in requires:
        if type(r) is Parameter:
            req.add(r.get_name(func.name, func_param_names))
        else:
            req.add(r)
    for p in provides:
        if type(p) is Parameter:
            prov.add(p.get_name(func.name, func_param_names))
        elif type(p) is Return:
            prov.add(p.get_name(func.name, func_return_names))
        else:
            prov.add(p)
    if type(tapes) is not GradTapeMode:
        tapes = {find_stmt(func, t).id for t in tapes}
    fwd, bwd, req_map, prov_map = impl(func, req, prov, tapes, tape_in_closure,
                                       reset_provided_grad, invert, user_grads)

    # Wrap fwd and bwd (originally ft.ffi.Func with ft.Func)
    fwd = Func(fwd.name, fwd.params, fwd.returns, fwd.body)
    bwd = Func(bwd.name, bwd.params, bwd.returns, bwd.body)

    # Enable lookup by positions of parameters or return values
    req_map = ParamRetDict(req_map, func=func)
    prov_map = ParamRetDict(prov_map, func=func)
    if _override_func_params is not None:
        req_map = ParamRetDict(req_map.d,
                               func_name=req_map.func_name,
                               param_names=_override_func_params,
                               return_names=req_map.return_names)

    if verbose is not None and verbose >= 1:
        print("Forward pass from AD:", file=sys.stderr)
        print(fwd, file=sys.stderr)
        print("Backward pass from AD:", file=sys.stderr)
        print(bwd, file=sys.stderr)

    if attach_backward:
        fwd.attach_backward(bwd, req_map, prov_map)
        return fwd
    else:
        return fwd, bwd, req_map, prov_map


def grad_(func=None,
          requires: Sequence[Union[str, Parameter]] = None,
          provides: Sequence[Union[str, Parameter, Return]] = None,
          tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
          tape_in_closure: bool = True,
          reset_provided_grad: bool = True,
          invert: bool = True,
          user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
          attach_backward: bool = False,
          jit_cache: Callable[Callable, Callable] = functools.cache,
          verbose: int = 0,
          _override_func_params: Sequence[str] = None):
    '''
    Reverse mode automatic differentiation (in-place version)

    It returns a forward function, a backward function, and two maps on names. The
    forward function computes the original results. The backward function computes
    the gradients. The maps map from the names of the original arguments and return
    values, to the names of their gradients.

    If `tape_in_closure` is True (by default), the forward function has the same
    interface of the original function, but it will store some intermediate tensors
    (the tape) in some hidden global states. The backward functions receives the same
    inputs as the original function plus the graidents of the outputs, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `grad_` is an in-place AD interface, the backward
    function passes the resulting gradients by additional mutable arguments. Names of
    the additional arguments can be looked up in the maps returned by `grad_`.

    If `tape_in_closure` is False, global states described above will be passed by
    explicit arguments and return values, so you can store or manipluate these states
    between the forward run and the backward run.

    Parameters
    ----------
    func : AST
        The original function
    requires : Sequence[Union[str, Parameter]]
        Name of input variables that need gradients. A parameter of a function can also
        be specified with a `Parameter` object by position
    provides : Sequence[Union[str, Parameter, Return]]
        Name of output variables whose gradients are known. A mutable parameter of a
        function can also be specified with a `Parameter` object by position. A return
        value of a function can also be specified with a `Return` object by position
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    tape_in_closure : bool
        True to pass taped tensors from the forward function to the backward function in
        implicit I/O parameters, i.e. in closure. False to pass these tensors as
        explicit I/O parameters. Default to True
    reset_provided_grad : bool
        If true, reset gradients for all variables in `provides` to 0 after use. This ensures
        the final result is correct when computing gradients of a program part by part with
        multiple calls to this function. If false, do not touch the provided gradient, which
        makes it convenient to run for multiple rounds for timing.
    invert : bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
        to true.
    user_grads : List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    attach_backward : bool
        If True, the forward function will be the only return value, with backward function
        and other metadata attached to it
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    verbose: int
        Verbosity level

    Returns
    -------
    Func if attach_backward else Tuple
        If `attach_backward` is False, return a tuple.

        - Return[0]: forward AST.
        - Return[1]: backward AST.
        - Return[2]: Mapping from names in requries to its gradient name.
        - Return[3]: Mapping from names in provides to its gradient name.

        If `attach_backward` is True, only the forward AST is returned, and others can be get by
        `.backward`, `.input_name_to_gradient_name` and `.output_name_to_gradient_name` on the
        return value, respectively.
    '''

    if func is None and not attach_backward:
        raise TypeError("`_grad` can be used as a decorator only when"
                        " `attach_backward=True`")
    return _grad_func(func,
                      impl=ffi.grad_,
                      requires=requires,
                      provides=provides,
                      tapes=tapes,
                      tape_in_closure=tape_in_closure,
                      reset_provided_grad=reset_provided_grad,
                      invert=invert,
                      user_grads=user_grads,
                      attach_backward=attach_backward,
                      jit_cache=jit_cache,
                      verbose=verbose,
                      _override_func_params=_override_func_params)


def grad(func=None,
         requires: Sequence[Union[str, Parameter]] = None,
         provides: Sequence[Union[str, Parameter, Return]] = None,
         tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
         tape_in_closure: bool = True,
         reset_provided_grad: bool = True,
         invert: bool = True,
         user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
         attach_backward: bool = False,
         jit_cache: Callable[Callable, Callable] = functools.cache,
         verbose: int = 0,
         _override_func_params: Sequence[str] = None):
    '''
    Reverse mode automatic differentiation (out-of-place version)

    It returns a forward function, a backward function, and two maps on names. The
    forward function computes the original results. The backward function computes
    the gradients. The maps map from the names of the original arguments and return
    values, to the names of their gradients.

    If `tape_in_closure` is True (by default), the forward function has the same
    interface of the original function, but it will store some intermediate tensors
    (the tape) in some hidden global states. The backward functions receives the same
    inputs as the original function plus the graidents of the outputs, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `grad` is an out-of-place AD interface, the backward
    function returns the resulting gradients as additional return values. Names of
    the additional arguments and return values can be looked up in the maps returned
    by `grad`.

    If `tape_in_closure` is False, global states described above will be passed by
    explicit arguments and return values, so you can store or manipluate these states
    between the forward run and the backward run.

    Parameters
    ----------
    func : AST
        The original function
    requires : Sequence[Union[str, Parameter]]
        Name of input variables that need gradients. A parameter of a function can also
        be specified with a `Parameter` object by position
    provides : Sequence[Union[str, Parameter, Return]]
        Name of output variables whose gradients are known. A mutable parameter of a
        function can also be specified with a `Parameter` object by position. A return
        value of a function can also be specified with a `Return` object by position
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    tape_in_closure : bool
        True to pass taped tensors from the forward function to the backward function in
        implicit I/O parameters, i.e. in closure. False to pass these tensors as
        explicit I/O parameters. Default to True
    reset_provided_grad : bool
        If true, reset gradients for all variables in `provides` to 0 after use. This ensures
        the final result is correct when computing gradients of a program part by part with
        multiple calls to this function. If false, do not touch the provided gradient, which
        makes it convenient to run for multiple rounds for timing.
    invert : bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
    user_grads : List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    attach_backward : bool
        If True, the forward function will be the only return value, with backward function
        and other metadata attached to it
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    verbose: int
        Verbosity level

    Returns
    -------
    Func if attach_backward else Tuple
        If `attach_backward` is False, return a tuple.

        - Return[0]: forward AST.
        - Return[1]: backward AST.
        - Return[2]: Mapping from names in requries to its gradient name.
        - Return[3]: Mapping from names in provides to its gradient name.

        If `attach_backward` is True, only the forward AST is returned, and others can be get by
        `.backward`, `.input_name_to_gradient_name` and `.output_name_to_gradient_name` on the
        return value, respectively.
    '''

    if func is None and not attach_backward:
        raise TypeError("`grad` can be used as a decorator only when"
                        " `attach_backward=True`")
    return _grad_func(func,
                      impl=ffi.grad,
                      requires=requires,
                      provides=provides,
                      tapes=tapes,
                      tape_in_closure=tape_in_closure,
                      reset_provided_grad=reset_provided_grad,
                      invert=invert,
                      user_grads=user_grads,
                      attach_backward=attach_backward,
                      jit_cache=jit_cache,
                      verbose=verbose,
                      _override_func_params=_override_func_params)


@as_decorator
def jacrev_(func=None,
            inputs: Sequence[Union[str, Parameter]] = None,
            output: Union[str, Parameter, Return] = None,
            flatten: bool = False,
            tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
            tape_in_closure: bool = True,
            invert: bool = True,
            user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
            attach_backward: bool = False,
            jit_cache: Callable[Callable, Callable] = functools.cache,
            verbose: int = 0,
            _override_func_params: Sequence[str] = None):
    '''
    Compute Jacobian tensors using Reverse mode automatic differentiation (in-place)

    `jacrev` computes one Jacobian tensor for one output and one or more inputs. Each
    Jacobian tensor consists of derivatives of all elements in the output tensor w.r.t.
    all elements in each inputs tensor.

    It returns a forward function, a backward function, and a map on names. The
    forward function computes the original results. The backward function computes
    the Jacobian tensors. The map maps from the names of the original arguments to the
    names of their Jacobian tensors.

    By default, the forward function has the same interface of the original function, but
    it will store some intermediate tensors (the tape) in some hidden global states. The
    backward functions receives the same inputs as the original function, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `jacrev_` is an in-place interface, the backward
    function passes the resulting gradients by additional mutable arguments. Names of
    the additional arguments can be looked up in the map returned by `jacrev_`.

    Suppose the output's shape is `(d1, d2, ...)`, and there are two inputs, whose shapes
    are `(e1, e2, ...)` and `(f1, f2, ...)`, respectively. If `flatten` is False (by
    default), the Jacobian tensors' shape will be `(d1, d2, ..., e1, e2, ...)` and
    `(d1, d2, ..., f1, f2, ...)`, respectively. If `flatten` is True, there will be only
    one Jacbian tensor, whose shape will be `(d1 * d2 * ..., e1 * e2 * ... + f1 * f2 * ...)`.

    If `tape_in_closure` is False, global states described above will be passed by
    explicit arguments and return values, so you can store or manipluate these states
    between the forward run and the backward run.

    Parameters
    ----------
    func : AST
        The original function
    inputs : Sequence[Union[str, Parameter]]
        Name of input variables that the Jacobian tensors are for. A parameter of a function
        can also be specified with a `Parameter` object by position
    output : Union[str, Parameter, Return]
        Name of one output variables that the Jacobian tensors are for. A mutable parameter
        of a function can also be specified with a `Parameter` object by position. A return
        value of a function can also be specified with a `Return` object by position
    flatten : bool
        If True, concatenate all Jacobian tensors together, to form an `(n, m)`-shaped output,
        where `n` is the total number of elements in the specified output, and `m` is the
        total number of elements in the specified inputs. This requires all involved inputs
        having the same data type and memory type. In this case, the name of the Jacobian
        tensor will be `"jacrev.flatten"`, and the returned name map will be empty
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    tape_in_closure : bool
        True to pass taped tensors from the forward function to the backward function in
        implicit I/O parameters, i.e. in closure. False to pass these tensors as
        explicit I/O parameters. Default to True
    invert: bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
    user_grads: List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    attach_backward : bool
        If True, the forward function will be the only return value, with backward function
        and other metadata attached to it
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    verbose: int
        Verbosity level

    Returns
    -------
    Func if attach_backward else Tuple
        If `attach_backward` is False, return a tuple.

        - Return[0]: forward AST.
        - Return[1]: backward AST.
        - Return[2]: Mapping from names in inputs to its Jacobian tensor name.

        If `attach_backward` is True, only the forward AST is returned, and others can be get by
        `.backward` and `.input_name_to_gradient_name` on the return value, respectively.
    '''

    if not isinstance(func, ffi.AST) and not isinstance(func, JITTemplate):
        func = transform(func, verbose=verbose)

    if isinstance(func, JITTemplate):

        if not attach_backward:
            raise KeyError(
                "JIT is only supported when attach_backward=True in `jacrev_`")

        class JacRevTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return jacrev_(
                    func.instantiate_by_only_jit_args(*jit_args),
                    inputs=inputs,
                    output=output,
                    flatten=flatten,
                    tapes=tapes,
                    tape_in_closure=tape_in_closure,
                    invert=invert,
                    user_grads=user_grads,
                    attach_backward=attach_backward,
                    verbose=verbose,
                    # Override func params to leave please for JIT parameters
                    _override_func_params=list(self.params))

        return JacRevTemplate(func.params, func.jit_param_names)

    # 1. Do a general AD
    fwd, bwd, req_map, prov_map = grad_(
        func,
        inputs, [output],
        tapes=tapes,
        tape_in_closure=tape_in_closure,
        invert=invert,
        user_grads=user_grads,
        verbose=verbose,
        _override_func_params=_override_func_params)

    # 2. Build a new backawrd, which runs the general backward multiple times. Each time we
    # set one of the elements of the output gradient to 1, and others to 0.
    #
    # TODO: We can remove redundant computation on zero elements by first performing
    # pass/shrink_var on the forward pass and then do AD. But this requires to somehow merge
    # the forward passes for every items, to share the forward pass among multiple backward
    # passes
    with lang_overload, ctx_stack:
        # 2a. Find I/O VarDefs, modify them, and build them in correct order

        def is_io_vardef(x):
            if x.type() == ffi.ASTNodeType.VarDef:
                for param in bwd.params:
                    if param.name == x.name:
                        return True
                for ret in bwd.returns:
                    if ret.name == x.name:
                        return True
            return False

        def is_out_grad(name):
            return name == prov_map[output]

        def is_in_grad(name):
            for item in inputs:
                if name == req_map[item]:
                    return True
            return False

        old_io_vardefs = find_all_stmt(bwd.body,
                                       is_io_vardef)  # In DFS pre order
        old_out_grad_vardef = None
        for old_io_vardef in old_io_vardefs:
            if is_out_grad(old_io_vardef.name):
                old_out_grad_vardef = old_io_vardef
        assert old_out_grad_vardef is not None, "Output is not found"
        open_new_vardefs = []
        in_grad_to_new_ref = {}
        other_to_new_ref = {}
        new_out_ref = None
        for old_io_vardef in old_io_vardefs:
            new_vardef = None
            if is_in_grad(old_io_vardef.name):
                # Prepend the output dimension to the input dimension. FIXME: What if a variable
                # used in the output dimension is undefined here?
                new_shape = list(old_io_vardef.buffer.tensor.shape)
                new_shape = list(
                    old_out_grad_vardef.buffer.tensor.shape) + new_shape
                new_vardef = VarDef(
                    old_io_vardef.name, new_shape,
                    old_io_vardef.buffer.tensor.dtype,
                    'cache' if flatten else old_io_vardef.buffer.atype,
                    old_io_vardef.buffer.mtype)
                in_grad_to_new_ref[old_io_vardef.name] = new_vardef.__enter__()
            elif old_io_vardef is old_out_grad_vardef:
                new_vardef = VarDef(old_io_vardef.name,
                                    old_io_vardef.buffer.tensor.shape,
                                    old_io_vardef.buffer.tensor.dtype, 'cache',
                                    old_io_vardef.buffer.mtype)
                new_out_ref = new_vardef.__enter__()
            else:
                new_vardef = VarDef(old_io_vardef.name,
                                    old_io_vardef.buffer.tensor.shape,
                                    old_io_vardef.buffer.tensor.dtype,
                                    old_io_vardef.buffer.atype,
                                    old_io_vardef.buffer.mtype)
                other_to_new_ref[old_io_vardef.name] = new_vardef.__enter__()
            open_new_vardefs.append(new_vardef)
        # There may be unused parameters, just pass None to them
        for p in bwd.params:
            if (p.name != prov_map[output] and
                    p.name not in in_grad_to_new_ref and
                    p.name not in other_to_new_ref):
                other_to_new_ref[p.name] = None

        # 2b. Iterate through all elements in the output, and computes its gradient

        @inline
        def body(new_out_ref_slice, in_grad_to_new_ref_slice):
            if ndim(new_out_ref_slice) == 0:
                new_out_ref[...] = 0
                new_out_ref_slice[...] = 1
                params = {
                    prov_map[output]: new_out_ref,
                    **in_grad_to_new_ref_slice,
                    **other_to_new_ref
                }
                #! label: general_bwd
                bwd(**params, _freetensor_force_allow_closures=True)
            else:
                for _freetensor_jacrev_i in range(new_out_ref_slice.shape(0)):
                    body(
                        new_out_ref_slice[_freetensor_jacrev_i], {
                            key:
                                in_grad_to_new_ref_slice[key]
                                [_freetensor_jacrev_i]
                            for key in in_grad_to_new_ref_slice
                        })

        def prod(iterable):
            return functools.reduce(lambda x, y: x * y, iterable, 1)

        @inline
        def body_wrapper(*args, **kvs):
            body(*args, **kvs)
            if flatten:
                from .. import libop
                dtype = None
                mtype = None
                tot_in_size = 0
                for p in bwd.params:
                    if p.name in in_grad_to_new_ref:
                        in_grad_new_ref = in_grad_to_new_ref[p.name]
                        tot_in_size += prod(
                            in_grad_new_ref.shape()[new_out_ref.ndim:])
                        if dtype is not None and dtype != in_grad_new_ref.dtype:
                            raise ffi.InvalidAutoGrad(
                                "jacrev with flatten=True requires all involved inputs having the"
                                " same data type")
                        if mtype is not None and mtype != in_grad_new_ref.mtype:
                            raise ffi.InvalidAutoGrad(
                                "jacrev with flatten=True requires all involved inputs having the"
                                " same memory type")
                        dtype = in_grad_new_ref.dtype
                        mtype = in_grad_new_ref.mtype
                tot_out_size = prod(new_out_ref.shape())
                with VarDef("jacrev.flatten", (tot_out_size, tot_in_size),
                            dtype, 'output', mtype) as flattened_ref:
                    flattened_off = 0
                    for p in bwd.params:
                        if p.name in in_grad_to_new_ref:
                            in_grad_new_ref = in_grad_to_new_ref[p.name]
                            this_in_size = prod(
                                in_grad_new_ref.shape()[new_out_ref.ndim:])
                            libop.flatten_onnx_(
                                in_grad_new_ref,
                                flattened_ref[:, flattened_off:flattened_off +
                                              this_in_size],
                                axis=new_out_ref.ndim)
                            flattened_off += this_in_size

        with LifetimeScope():
            body_wrapper(new_out_ref, in_grad_to_new_ref)

        # 2c. Close scopes and build a new AST
        for new_vardef in reversed(open_new_vardefs):
            new_vardef.__exit__(None, None, None)
        new_bwd_body = pop_ast()

        # 2d. Post optimization
        #
        # The outputs' gradient are out of the `_freetensor_jacrev_i` loops that iterates elements
        # of the output tensor, because they are inputs to `old_bwd`. However, they are actually
        # local variables now, so we can sink them to enable more schedules.
        new_bwd_body = sink_var(new_bwd_body)

        # 2e. Make Func signature
        new_params = list(filter(lambda p: not is_out_grad(p.name), bwd.params))
        if flatten:
            new_params = list(
                filter(lambda p: not is_in_grad(p.name),
                       new_params)) + ['jacrev.flatten']
        new_bwd = ffi.Func(bwd.name, new_params, bwd.returns, new_bwd_body)

    if verbose:
        print("Backward pass from jacrev:", file=sys.stderr)
        print(new_bwd, file=sys.stderr)

    if flatten:
        req_map = {}
    if attach_backward:
        fwd.attach_backward(new_bwd, req_map, None)
        return fwd
    else:
        return fwd, new_bwd, req_map


@as_decorator
def jacrev(func=None,
           inputs: Sequence[Union[str, Parameter]] = None,
           output: Union[str, Parameter, Return] = None,
           flatten: bool = False,
           tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
           tape_in_closure: bool = True,
           invert: bool = True,
           user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
           attach_backward: bool = False,
           jit_cache: Callable[Callable, Callable] = functools.cache,
           verbose: int = 0,
           _override_func_params: Sequence[str] = None):
    '''
    Compute Jacobian tensors using Reverse mode automatic differentiation (out-of-place)

    `jacrev` computes one Jacobian tensor for one output and one or more inputs. Each
    Jacobian tensor consists of derivatives of all elements in the output tensor w.r.t.
    all elements in each inputs tensor.

    It returns a forward function, a backward function, and a map on names. The
    forward function computes the original results. The backward function computes
    the Jacobian tensors. The map maps from the names of the original arguments to the
    names of their Jacobian tensors.

    By default, the forward function has the same interface of the original function, but
    it will store some intermediate tensors (the tape) in some hidden global states. The
    backward functions receives the same inputs as the original function, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `jacrev` is an out-of-place interface, the backward
    function returns the resulting Jacobian as additional return values. Names of the
    additional return values can be looked up in the map returned by `jacrev`.

    Suppose the output's shape is `(d1, d2, ...)`, and there are two inputs, whose shapes
    are `(e1, e2, ...)` and `(f1, f2, ...)`, respectively. If `flatten` is False (by
    default), the Jacobian tensors' shape will be `(d1, d2, ..., e1, e2, ...)` and
    `(d1, d2, ..., f1, f2, ...)`, respectively. If `flatten` is True, there will be only
    one Jacbian tensor, whose shape will be `(d1 * d2 * ..., e1 * e2 * ... + f1 * f2 * ...)`.

    If `tape_in_closure` is False, global states described above will be passed by
    explicit arguments and return values, so you can store or manipluate these states
    between the forward run and the backward run.

    Parameters
    ----------
    func : AST
        The original function
    inputs : Sequence[Union[str, Parameter]]
        Name of input variables that the Jacobian tensors are for.
    output : Union[str, Parameter, Return]
        Name of one output variables that the Jacobian tensors are for. A return value of a
        function can be specified with a `Return` object
    flatten : bool
        If True, concatenate all Jacobian tensors together, to form an `(n, m)`-shaped output,
        where `n` is the total number of elements in the specified output, and `m` is the
        total number of elements in the specified inputs. This requires all involved inputs
        having the same data type and memory type. In this case, the name of the Jacobian
        tensor will be `"jacrev.flatten"`, and the returned name map will be empty
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    tape_in_closure : bool
        True to pass taped tensors from the forward function to the backward function in
        implicit I/O parameters, i.e. in closure. False to pass these tensors as
        explicit I/O parameters. Default to True
    invert: bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
    user_grads: List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    attach_backward : bool
        If True, the forward function will be the only return value, with backward function
        and other metadata attached to it
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    verbose: int
        Verbosity level

    Returns
    -------
    Func if attach_backward else Tuple
        If `attach_backward` is False, return a tuple.

        - Return[0]: forward AST.
        - Return[1]: backward AST.
        - Return[2]: Mapping from names in inputs to its Jacobian tensor name.

        If `attach_backward` is True, only the forward AST is returned, and others can be get by
        `.backward` and `.input_name_to_gradient_name` on the return value, respectively.
    '''

    if not isinstance(func, ffi.AST) and not isinstance(func, JITTemplate):
        func = transform(func, verbose=verbose)

    if isinstance(func, JITTemplate):

        if not attach_backward:
            raise KeyError(
                "JIT is only supported when attach_backward=True in `jacrev`")

        class JacRevTemplate(JITTemplate):

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return jacrev(
                    func.instantiate_by_only_jit_args(*jit_args),
                    inputs=inputs,
                    output=output,
                    flatten=flatten,
                    tapes=tapes,
                    tape_in_closure=tape_in_closure,
                    invert=invert,
                    user_grads=user_grads,
                    attach_backward=True,
                    verbose=verbose,
                    # Override func params to leave please for JIT parameters
                    _override_func_params=list(self.params))

        return JacRevTemplate(func.params, func.jit_param_names)

    fwd, bwd, input_map = jacrev_(func,
                                  inputs,
                                  output,
                                  flatten=flatten,
                                  tapes=tapes,
                                  tape_in_closure=tape_in_closure,
                                  invert=invert,
                                  user_grads=user_grads,
                                  attach_backward=False,
                                  verbose=verbose,
                                  _override_func_params=_override_func_params)

    in_grads = set()
    if flatten:
        in_grads = {"jacrev.flatten"}
    else:
        for key in input_map:
            in_grads.add(input_map[key])

    new_params = []
    new_returns = bwd.returns
    for p in bwd.params:
        if p.name in in_grads:
            assert not p.is_in_closure
            dtype = find_stmt(
                bwd, lambda x: x.type() == ffi.ASTNodeType.VarDef and x.name ==
                p.name).buffer.tensor.dtype
            new_returns.append(FuncRet(p.name, dtype))
        else:
            new_params.append(p)

    new_bwd = Func(bwd.name, new_params, new_returns, bwd.body)
    if attach_backward:
        fwd.attach_backward(new_bwd, input_map, None)
        return fwd
    else:
        return fwd, new_bwd, input_map


def output_all_intermediates(stmt: ffi.Stmt, intermediates: Set[Union[str,
                                                                      ffi.ID]]):
    return ffi.output_all_intermediates(
        stmt, {find_stmt(stmt, i).id for i in intermediates})
