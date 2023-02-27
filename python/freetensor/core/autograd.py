from typing import Optional, Set, Union, Sequence
import sys

import freetensor_ffi as ffi

from freetensor_ffi import GradTapeMode
from freetensor_ffi import output_intermediates

from .analyze import find_stmt
from .frontend import transform


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
        return self.d[key]

    def __contains__(self, key):
        # Python's auto fallback from __getitem__ to __contains__ only works for
        # integer index
        if type(key) is Return:
            key = key.get_name(self.func)
        return key in self.d

    def __str__(self):
        return str(self.d)


def grad_body(stmt: ffi.Stmt,
              requires: Sequence[Union[str, Return]],
              provides: Sequence[Union[str, Return]],
              tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
              user_grads: Sequence[ffi.StmtSetToUserGrad] = []):
    ''' `grad` or `grad_` on a function body (for internal tests only) '''

    req = set(requires)
    prov = set(provides)
    if type(tapes) is not GradTapeMode:
        tapes = {find_stmt(stmt, t).id for t in tapes}
    return ffi.grad_body(stmt, req, prov, tapes, user_grads)


def _grad_func(impl,
               func: ffi.Func,
               requires: Sequence[Union[str, Return]],
               provides: Sequence[Union[str, Return]],
               tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
               tape_in_closure: bool = True,
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
                                       user_grads)
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
          user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
          verbose: Optional[int] = None):
    '''
    Reverse mode automatic differentiation

    It returns a forward function and a backward function. The forward has the same
    interface of the original function, but it will store some intermediate tensors
    (the tape) to be reused by the backward function in some global states. The
    backward function computes the gradients.

    `grad_` is an inplace version. The resulting gradient are mutable arguments of
    the backward function.

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
                      user_grads,
                      verbose=verbose)


def grad(func: ffi.Func,
         requires: Sequence[str],
         provides: Sequence[Union[str, Return]],
         tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly,
         tape_in_closure: bool = True,
         user_grads: Optional[Sequence[ffi.StmtSetToUserGrad]] = None,
         verbose: Optional[int] = None):
    '''
    Reverse mode automatic differentiation

    It returns a forward function and a backward function. The forward has the same
    interface of the original function, but it will store some intermediate tensors
    (the tape) to be reused by the backward function in some global states. The
    backward function computes the gradients.

    `grad` is an out-of-place version. The resulting gradient are returned from the
    backward function.

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
                      user_grads,
                      verbose=verbose)


def output_intermediates(stmt: ffi.Stmt, intermediates: Set[Union[str,
                                                                  ffi.ID]]):
    return ffi.output_intermediates(
        stmt, {find_stmt(stmt, i).id for i in intermediates})
