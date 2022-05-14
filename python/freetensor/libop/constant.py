from .. import core
from .assign import assign


@core.inline
def zeros_(y):
    '''
    Fill zeros to a tensor

    Parameters
    ----------
    y : VarRef
        The tensor to fill
    '''
    if core.ndim(y) == 0:
        y[()] = 0
    else:
        #! nid: L_elem
        for i in range(core.shape(y, 0)):
            #! nid: recur
            zeros_(y[i])


@core.inline
def zeros(shape, dtype, mtype=None):
    '''
    Create a zero tensor

    Parameters
    ----------
    shape : Sequence[Expr] or Var
        Shape of the variable. A variable can be created using a literal shape,
        or another fixed-length VarRef as a shape
    dtype : str or DataType
        Data type of the variable
    mtype : str or MemType (Optional)
        Memory type of the variable. If omitted, the main memory type of the
        default Target in config will be used

    Returns
    -------
    VarRef :
        The zero tensor
    '''
    y = core.empty(shape, dtype, mtype)
    #! nid: recur
    zeros_(y)
    return y
