import ffi
from ffi import MoveToSide

from .utils import *

class Schedule(ffi.Schedule):
    def __init__(self, ast):
        super(Schedule, self).__init__(ast)

    '''
    Split a loop into two nested loops

    To fission a loop into two consecutive loops, use `fission` instead

    Parameters
    ----------
    node : str, Stmt or Cursor
        The loop to be split
    factor : int
        Length of the inner loop. Set to -1 if using `nparts`
    nparts : int
        Length of the outer loop. Set to -1 if using `factor`

    Raises
    ------
    InvalidSchedule
        if the loop is not found

    Returns
    -------
    (str, str)
        (outer loop ID, inner loop ID)
    '''
    def split(self, node, factor=-1, nparts=-1):
        return super(Schedule, self).split(toId(node), factor, nparts)

    '''
    Reorder directly nested loops

    To swap consecutive loops, use `swap` instead

    Parameters
    ----------
    order : array like of str, Stmt or Cursor
        Vector of loops. The requested order of the loops

    Raises
    ------
    InvalidSchedule
        if the input is invalid or there are breaking dependencies
    '''
    def reorder(self, order):
        super(Schedule, self).reorder(list(map(toId, order)))

    '''
    Merge two directly nested loops into one

    To fuse consecutive loops, use `fuse` instead

    Parameters
    ----------
    loop1, loop2 : str, Stmt or Cursor
        loops to be merged, can be in any order

    Raises
    ------
    InvalidSchedule
        if the loops are not directly nested

    Returns
    -------
    str
        ID of the merged loop
    '''
    def merge(self, loop1, loop2):
        return super(Schedule, self).merge(toId(loop1), toId(loop2))

    '''
    Fission a loop into two loops each containing part of the statements, one
    followed by another

    To split loop into two nested loops, use `split` instead

    Parameters
    ----------
    loop : str, Stmt or Cursor
        The loop to be fissioned
    after : str, Stmt or Cursor
        The last statement of the first loop
    suffix0 : str
        ID suffix of the statements in the first loop, default to ".a", can be
        "" for convenience, but cannot be the same with suffix1
    suffix1 : str
        ID suffix of the statements in the second loop, default to ".b", can be
        "" for convenience, but cannot be the same with suffix0

    Raises
    ------
    InvalidSchedule
        if any dependency cannot be resolved

    Returns
    -------
    (map, map)
        ({old ID -> new ID in 1st loop}, {old ID -> new ID in 2nd loop})
    '''
    def fission(self, loop, after, suffix0=".a", suffix1=".b"):
        return super(Schedule, self).fission(
                toId(loop), toId(after), suffix0, suffix1)

    '''
    Fuse two directly following loops with the same length into one

    To merge nested loops into one, use `merge` instead

    Parameters
    ----------
    loop0 : str, Stmt or Cursor
        The leading loop
    loop1 : str, Stmt or Cursor
        The following loop

    Raises
    ------
    InvalidSchedule
        if the two loops are not directly following, the two loops are not with
        the same length, or there is any dependency cannot be resolved

    Returns
    -------
    str
        ID of the result loop
    '''
    def fuse(self, loop0, loop1):
        return super(Schedule, self).fuse(toId(loop0), toId(loop1))

    '''
    Swap statements in the same block

    To reorder nested loops, use `reorder` instead

    Parameters
    ----------
    order : array like of str, Stmt or Cursor
        The statements

    Raises
    ------
    InvalidSchedule
        if the statements are not found or the dependencies cannot be solved
    '''
    def swap(self, order):
        super(Schedule, self).swap(list(map(toId, order)))

    '''
    Cache a variable into a new local variable

    All needed data will be filled into the cache first, then all reads and
    writes will be directed to the cache, and finally all needed data will be
    flushed from the cache

    Note for reduction: This transformation preserves the computation order.
    It will transform

    ```
    a += x
    a += y
    ```

    to

    ```
    a.cache = a + x + y
    a = a.cache
    ```

    If you need a "real" cache for reduction, which reorders the computation,
    use `cache_reduction` instead

    Parameters
    ----------
    stmt : str, Stmt or Cursor
        The statement or block (e.g. an If or a For) to be modified
    var : str
        Name of the variable to be cached
    mtype : MemType
        Where to cache

    Raises
    ------
    InvalidSchedule
        if the ID or name is not found

    Returns
    -------
    (str, str, str)
        (ID of the statement that fills the cache, ID of the statement that
        flushes from the cache, name of the cache variable)
    '''
    def cache(self, stmt, var, mtype):
        return super(Schedule, self).cache(toId(stmt), var, parseMType(mtype))

    '''
    Perform local reductions (e.g. sum) in a local variable first, and then
    reduce the local result to the global variable

    E.g.

    ```
    a += x
    a += y
    ```

    will be transformed to be

    ```
    a.cache = x + y
    a += a.cache
    ```

    Parameters
    ----------
    stmt : str, Stmt or Cursor
        The statement or block (e.g. an If or a For) to be modified
    var : str
        Name of the variable to be cached. Only reductions are allowed on
        `var` in `stmt`. Plain reads or writes are not allowed
    mtype : MemType
        Where to cache

    Raises
    ------
    InvalidSchedule
        if the ID or name is not found, or there are unsupported reads or
        writes

    Returns
    -------
    (str, str, str)
        (ID of the statement that initialize the cache, ID of the statement
        that reduces the local result to the global result, name of the
        cache variable)
    '''
    def cache_reduction(self, stmt, var, mtype):
        return super(Schedule, self).cache_reduction(toId(stmt), var, parseMType(mtype))

    '''
    Move a statement to a new position

    This is a composite schedule command, which is implemented with other
    commands

    Parameters
    ----------
    stmt : str, Stmt or Cursor
        The statement to be moved
    side : MoveToSide
        Whether `stmt` will be BEFORE or AFTER `dst
    dst : str, Stmt or Cursor
        Insert `stmt` to be directly after this statement

    Raises
    ------
    InvalidSchedule
        if there is no feasible path to move

    Returns
    -------
    str
        The new ID of stmt
    '''
    def move_to(self, stmt, side, dst):
        return super(Schedule, self).move_to(toId(stmt), side, toId(dst))

    '''
    Mark a loop with a parallel implementation

    Parameters
    ----------
    loop : str, Stmt or Cursor
        The loop
    parallel : str
        Parallel implementation. Supported values are "openmp",
        "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x",
        "threadIdx.y", "threadIdx.z"
    '''
    def parallelize(self, loop, parallel):
        super(Schedule, self).parallelize(toId(loop), parallel)

    '''
    Mark a loop with a parallel implementation

    Parameters
    ----------
    loop : str, Stmt or Cursor
        The loop which is needing to unroll
    '''

    def unroll(self, loop):
        super(Schedule, self).unroll(toId(loop))