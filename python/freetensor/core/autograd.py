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


def grad(ast,
         requires: Sequence[Union[str, Return]],
         provides: Sequence[Union[str, Return]],
         tapes: Union[Sequence, GradTapeMode] = GradTapeMode.NoReuseOnly):
    '''
    Reverse mode Auto differentiation

    Parameters
    ----------
    ast : AST (Func or Stmt)
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

    req = set(requires)
    prov = set([])
    for p in provides:
        if type(p) is Return:
            assert type(ast) is ffi.Func
            assert len(ast.returns) > 0, f"{ast.name} has no return value"
            if p.n is not None:
                prov.add(ast.returns[p.n].name)
            else:
                assert len(
                    ast.returns
                ) == 1, f"{ast.name} has more than one return value, and you need to specify the number of a return value"
                prov.add(ast.returns[0].name)
        else:
            prov.add(p)
    if type(tapes) is not GradTapeMode:
        tapes = set(tapes)
    return ffi.grad(ast, req, prov, tapes)
