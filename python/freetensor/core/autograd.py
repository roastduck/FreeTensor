from typing import Optional, Union, Sequence

import freetensor_ffi as ffi

from freetensor_ffi import GradTapeMode
from freetensor_ffi import output_intermediates


class Return:
    '''
    Alias of a return value of a function

    `Return(n)` represents the n-th return value (counted from 0)

    `Return()` can be used if there is only one return value
    '''

    def __init__(self, n: Optional[int] = None):
        self.n = n


def grad_body(stmt: ffi.Stmt,
              requires: Sequence[Union[str, Return]],
              provides: Sequence[Union[str, Return]],
              tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly):
    ''' `grad` or `grad_` on a function body (for internal tests only) '''

    req = set(requires)
    prov = set(provides)
    if type(tapes) is not GradTapeMode:
        tapes = set(tapes)
    return ffi.grad_body(stmt, req, prov, tapes)


def _grad_func(impl,
               func: ffi.Func,
               requires: Sequence[Union[str, Return]],
               provides: Sequence[Union[str, Return]],
               tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly):

    req = set(requires)
    prov = set([])
    for p in provides:
        if type(p) is Return:
            assert len(func.returns) > 0, f"{func.name} has no return value"
            if p.n is not None:
                prov.add(func.returns[p.n].name)
            else:
                assert len(
                    func.returns
                ) == 1, f"{func.name} has more than one return value, and you need to specify the number of a return value"
                prov.add(func.returns[0].name)
        else:
            prov.add(p)
    if type(tapes) is not GradTapeMode:
        tapes = set(tapes)
    return impl(func, req, prov, tapes)


def grad_(func: ffi.Func,
          requires: Sequence[Union[str, Return]],
          provides: Sequence[Union[str, Return]],
          tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly):
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
        VarDef IDs of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in requries to its gradient name.
         3. Mapping from names in provides to its gradient name.
         4. Mapping from VarDef IDs of intermediate variables being stored to its
        corresponding output names
        )
    '''

    return _grad_func(ffi.grad_, func, requires, provides, tapes)


def grad(func: ffi.Func,
         requires: Sequence[Union[str, Return]],
         provides: Sequence[Union[str, Return]],
         tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly):
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
        VarDef IDs of them. It can also be a `GradTapeMode`, then it will determine
        which intermediate variables to be stored by heuristics. Avail `GradTapeMode`s
        are: All: store all variables including local scalars; None: store nothing;
        NoReuseOnly: store variables that only hold one version of data, which means
        we do not have to store each version of them in their history

    Returns
    -------
    tuple
        (
         0. Forward AST.
         1. Backward AST.
         2. Mapping from names in requries to its gradient name.
         3. Mapping from names in provides to its gradient name.
         4. Mapping from VarDef IDs of intermediate variables being stored to its
        corresponding output names
        )
    '''

    return _grad_func(ffi.grad, func, requires, provides, tapes)
