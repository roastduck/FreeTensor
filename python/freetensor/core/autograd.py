__all__ = [
    'GradTapeMode', 'Return', 'ArgRetDict', 'grad_body', 'grad', 'grad_',
    'jacrev', 'output_all_intermediates'
]

from typing import Optional, Set, Union, Sequence
import sys

import freetensor_ffi as ffi

from freetensor_ffi import GradTapeMode
from freetensor_ffi import output_all_intermediates

from .analyze import find_stmt, find_all_stmt
from .frontend import LifetimeScope, ndim
from .transform import transform, inline
from .func import Func, FuncRet
from .stmt import VarDef
from .context import pop_ast


class Return:
    '''
    Alias of a return value of a function

    `Return(n)` represents the n-th return value (counted from 0)

    `Return()` can be used if there is only one return value
    '''

    def __init__(self, n: Optional[int] = None):
        self.n = n

    def get_name(self, func):
        assert len(func.returns) > 0, f"{func.name} has no return value"
        if self.n is not None:
            return func.returns[self.n].name
        else:
            assert len(
                func.returns
            ) == 1, f"{func.name} has more than one return value, and you need to specify the number of a return value"
            return func.returns[0].name

    def __str__(self):
        return f"Return({self.n})"


class ArgRetDict:
    ''' Look an object using either a function argument or return value's name or its position '''

    def __init__(self, func, d):
        self.func = func
        self.d = d

    def __getitem__(self, key):
        if type(key) is Return:
            key = key.get_name(self.func)
        try:
            return self.d[key]
        except KeyError as e:
            raise KeyError(
                f"There is no {key} in arguments or return values") from e

    def __contains__(self, key):
        # Python's auto fallback from __getitem__ to __contains__ only works for
        # integer index
        if type(key) is Return:
            key = key.get_name(self.func)
        return key in self.d

    def __str__(self):
        return str(self.d)

    def __iter__(self):
        return iter(self.d)


def grad_body(stmt: ffi.Stmt,
              requires: Sequence[Union[str, Return]],
              provides: Sequence[Union[str, Return]],
              tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
              invert: bool = True,
              user_grads: Sequence[ffi.StmtSetToUserGrad] = []):
    ''' `grad` or `grad_` on a function body (for internal tests only) '''

    req = set(requires)
    prov = set(provides)
    if type(tapes) is not GradTapeMode:
        tapes = {find_stmt(stmt, t).id for t in tapes}
    return ffi.grad_body(stmt, req, prov, tapes, invert, user_grads)


def _grad_func(impl,
               func: ffi.Func,
               requires: Sequence[Union[str, Return]],
               provides: Sequence[Union[str, Return]],
               tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
               tape_in_closure: bool = True,
               invert: bool = True,
               user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
               verbose: Optional[int] = None):

    if not issubclass(type(func), ffi.AST):
        func = transform(func, verbose=verbose)
    if user_grads is None:
        if func.user_grads is not None:
            user_grads = func.user_grads
        else:
            user_grads = []
    req = set(requires)
    prov = set([])
    for p in provides:
        if type(p) is Return:
            prov.add(p.get_name(func))
        else:
            prov.add(p)
    if type(tapes) is not GradTapeMode:
        tapes = {find_stmt(func, t).id for t in tapes}
    fwd, bwd, req_map, prov_map = impl(func, req, prov, tapes, tape_in_closure,
                                       invert, user_grads)

    # Wrap fwd and bwd (originally ft.ffi.Func with ft.Func)
    fwd = Func(fwd.name, fwd.params, fwd.returns, fwd.body)
    bwd = Func(bwd.name, bwd.params, bwd.returns, bwd.body)

    if verbose is not None and verbose >= 1:
        print("Forward pass from AD:", file=sys.stderr)
        print(fwd, file=sys.stderr)
        print("Backward pass from AD:", file=sys.stderr)
        print(bwd, file=sys.stderr)
    return fwd, bwd, ArgRetDict(func, req_map), ArgRetDict(func, prov_map)


def grad_(func: ffi.Func,
          requires: Sequence[str],
          provides: Sequence[Union[str, Return]],
          tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
          tape_in_closure: bool = True,
          invert: bool = True,
          user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
          verbose: Optional[int] = None):
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
    requires : Sequence[str]
        Name of input variables that need gradients
    provides : Sequence[Union[str, Return]]
        Name of output variables whose gradients are known. A return value of a
        function can be specified with a `Return` object
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
        to true.
    user_grads: List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    verbose: int
        Verbosity level

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in requries to its gradient name.
         3. Mapping from names in provides to its gradient name.
        )
    '''

    return _grad_func(ffi.grad_,
                      func,
                      requires,
                      provides,
                      tapes,
                      tape_in_closure,
                      invert,
                      user_grads,
                      verbose=verbose)


def grad(func: ffi.Func,
         requires: Sequence[str],
         provides: Sequence[Union[str, Return]],
         tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
         tape_in_closure: bool = True,
         invert: bool = True,
         user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
         verbose: Optional[int] = None):
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
    requires : Sequence[str]
        Name of input variables that need gradients
    provides : Sequence[Union[str, Return]]
        Name of output variables whose gradients are known. A return value of a
        function can be specified with a `Return` object
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
    verbose: int
        Verbosity level

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in requries to its gradient name.
         3. Mapping from names in provides to its gradient name.
        )
    '''

    return _grad_func(ffi.grad,
                      func,
                      requires,
                      provides,
                      tapes,
                      tape_in_closure,
                      invert,
                      user_grads,
                      verbose=verbose)


def jacrev_(func: ffi.Func,
            inputs: Sequence[str],
            output: Union[str, Return],
            tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
            invert: bool = True,
            user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
            verbose: Optional[int] = None):
    '''
    Compute Jacobian tensors using Reverse mode automatic differentiation (in-place)

    `jacrev` computes one Jacobian tensor for one output and one or more inputs. Each
    Jacobian tensor consists of derivatives of all elements in the output tensor w.r.t.
    all elements in each inputs tensor.

    It returns a forward function, a backward function, and a map on names. The
    forward function computes the original results. The backward function computes
    the Jacobian tensors. The map maps from the names of the original arguments to the
    names of their Jacobian tensors.

    The forward function has the same interface of the original function, but it will
    store some intermediate tensors (the tape) in some hidden global states. The
    backward functions receives the same inputs as the original function, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `jacrev_` is an in-place interface, the backward
    function passes the resulting gradients by additional mutable arguments. Names of
    the additional arguments can be looked up in the map returned by `jacrev_`.

    Suppose the output

    Parameters
    ----------
    func : AST
        The original function
    inputs : Sequence[str]
        Name of input variables that the Jacobian tensors are for.
    output : Union[str, Return]
        Name of one output variables that the Jacobian tensors are for. A return value of a
        function can be specified with a `Return` object
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    invert: bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
    user_grads: List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    verbose: int
        Verbosity level

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in inputs to its gradient name.
        )
    '''

    # 1. Do a general AD
    fwd, bwd, req_map, prov_map = grad_(func,
                                        inputs, [output],
                                        tapes=tapes,
                                        invert=invert,
                                        user_grads=user_grads,
                                        verbose=verbose)

    # 2. Build a new backawrd, which runs the general backward multiple times. Each time we
    # set one of the elements of the output gradient to 1, and others to 0.
    #
    # TODO: We can remove redundant computation on zero elements by first performing
    # pass/shrink_var on the forward pass and then do AD. But this requires to somehow merge
    # the forward passes for every items, to share the forward pass among multiple backward
    # passes

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

    def is_out_grad(old_io_vardef):
        return old_io_vardef.name == prov_map[output]

    def is_in_grad(old_io_vardef):
        for item in inputs:
            if old_io_vardef.name == req_map[item]:
                return True
        return False

    old_io_vardefs = find_all_stmt(bwd.body, is_io_vardef)  # In DFS pre order
    old_out_grad_vardef = None
    for old_io_vardef in old_io_vardefs:
        if is_out_grad(old_io_vardef):
            old_out_grad_vardef = old_io_vardef
    assert old_out_grad_vardef is not None, "Output is not found"
    open_new_vardefs = []
    in_grad_to_new_ref = {}
    other_to_new_ref = {}
    new_out_ref = None
    for old_io_vardef in old_io_vardefs:
        new_vardef = None
        if is_in_grad(old_io_vardef):
            # Prepend the output dimension to the input dimension. FIXME: What if a variable
            # used in the output dimension is undefined here?
            new_shape = list(old_io_vardef.buffer.tensor.shape)
            new_shape = list(
                old_out_grad_vardef.buffer.tensor.shape) + new_shape
            new_vardef = VarDef(old_io_vardef.name, new_shape,
                                old_io_vardef.buffer.tensor.dtype,
                                old_io_vardef.buffer.atype,
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
        if (p.name != prov_map[output] and p.name not in in_grad_to_new_ref and
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
            bwd(**params)
        else:
            for i in range(new_out_ref_slice.shape(0)):
                body(
                    new_out_ref_slice[i], {
                        key: in_grad_to_new_ref_slice[key][i]
                        for key in in_grad_to_new_ref_slice
                    })

    with LifetimeScope():
        body(new_out_ref, in_grad_to_new_ref)

    # 2c. Close scopes and build a new AST
    for new_vardef in reversed(open_new_vardefs):
        new_vardef.__exit__(None, None, None)
    new_bwd_body = pop_ast()

    # 2d. Make Func signature
    new_bwd = ffi.Func(
        bwd.name, list(filter(lambda x: x.name != prov_map[output],
                              bwd.params)), bwd.returns, new_bwd_body)

    if verbose:
        print("Backward pass from jacrev:", file=sys.stderr)
        print(new_bwd, file=sys.stderr)

    return (fwd, new_bwd, req_map)


def jacrev(func: ffi.Func,
           inputs: Sequence[str],
           output: Union[str, Return],
           tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
           invert: bool = True,
           user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
           verbose: Optional[int] = None):
    '''
    Compute Jacobian tensors using Reverse mode automatic differentiation (out-of-place)

    `jacrev` computes one Jacobian tensor for one output and one or more inputs. Each
    Jacobian tensor consists of derivatives of all elements in the output tensor w.r.t.
    all elements in each inputs tensor.

    It returns a forward function, a backward function, and a map on names. The
    forward function computes the original results. The backward function computes
    the Jacobian tensors. The map maps from the names of the original arguments to the
    names of their Jacobian tensors.

    The forward function has the same interface of the original function, but it will
    store some intermediate tensors (the tape) in some hidden global states. The
    backward functions receives the same inputs as the original function, and also reads
    from the hidden states. The outputs of the original function are no longer exist
    in the backward function, and the input-outputs of the original function are
    converted to pure inputs. As `jacrev` is an out-of-place interface, the backward
    function returns the resulting Jacobian as additional return values. Names of the
    additional return values can be looked up in the map returned by `jacrev`.

    Suppose the output

    Parameters
    ----------
    func : AST
        The original function
    inputs : Sequence[str]
        Name of input variables that the Jacobian tensors are for.
    output : Union[str, Return]
        Name of one output variables that the Jacobian tensors are for. A return value of a
        function can be specified with a `Return` object
    tapes : Union[Sequence, GradTapeMode]
        Intermediate variables that need to be stored from the forward pass and
        reused in the backward pass. This parameter can be a sequence, which contains
        VarDef selectors of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history
    invert: bool
        If set to true, it can reduce the amount of recomputation or taping required.
        However, this may result in a loss of precision for floating-point numbers. Defaults
    user_grads: List[ffi.StmtSetToUserGrad]
        For custom gradient. You do not have to explicitly set this parameter unless you
        are manipulating `func` by yourself (not getting it from the Python frontend). See
        `UserGrad` for details
    verbose: int
        Verbosity level

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in inputs to its gradient name.
        )
    '''

    fwd, bwd, input_map = jacrev_(func,
                                  inputs,
                                  output,
                                  tapes=tapes,
                                  invert=invert,
                                  user_grads=user_grads,
                                  verbose=verbose)

    in_grads = set()
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
    return fwd, new_bwd, input_map


def output_all_intermediates(stmt: ffi.Stmt, intermediates: Set[Union[str,
                                                                      ffi.ID]]):
    return ffi.output_all_intermediates(
        stmt, {find_stmt(stmt, i).id for i in intermediates})
