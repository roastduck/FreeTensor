import ffi
from ffi import FissionSide, MoveToSide, VarSplitMode

from .utils import *


class Schedule(ffi.Schedule):

    def __init__(self, ast):
        super(Schedule, self).__init__(ast)

    def split(self, node, factor=-1, nparts=-1):
        """
        Split a loop into two nested loops

        To fission a loop into two consecutive loops, use `fission` instead

        Two modes are provided:

        1. Specify `factor` and leave `nparts` to -1. It will result in an outer
        loop with length `ceil(n / factor)`, and an inner loop with length
        `factor`, where `n` is the original loop length. The original iterator
        `i` will be transformed to `i0 * factor + i1`, where `i0` and `i1` are
        the iterators of the new outer and inner loops, respectively
        2. Specify `nparts` and leave `factor` to -1. It will result in an
        outer loop with length `nparts`, and an inner loop with length `ceil(n /
        nparts)`, where `n` is the original loop length. The original iterator
        `i` will be transformed to `i0 * ceil(n / nparts) + i1`, where `i0` and
        `i1` are the iterators of the new outer and inner loops, respectively

        Please note that the second mode will introduce an `i0 * ceil(n /
        nparts)` factor into the program, which cannot be recognized by
        polyhedral analysis, which may hinder some following schedules. If
        possible, plese use the first mode, and then reorder the inner and outer
        loops

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
        """
        return super(Schedule, self).split(toId(node), factor, nparts)

    def reorder(self, order):
        """
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
        """
        super(Schedule, self).reorder(list(map(toId, order)))

    def merge(self, loop1, loop2):
        """
        Merge two directly nested loops into one

        To fuse consecutive loops, use `fuse` instead

        `parallelize`, `unroll` and `vectorize` properties will be reset on the
        merged loop

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
        """
        return super(Schedule, self).merge(toId(loop1), toId(loop2))

    def fission(self, loop, side, splitter, suffix0=".a", suffix1=".b"):
        """
        Fission a loop into two loops each containing part of the statements, one
        followed by another

        To split loop into two nested loops, use `split` instead

        Parameters
        ----------
        loop : str, Stmt or Cursor
            The loop to be fissioned
        side : FissionSide
            If `After`, `splitter` is the last statement of the first loop. If `Before`,
            `splitter` is the first statement of the second loop
        splitter : str, Stmt or Cursor
            Where to fission the loop
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
        """
        return super(Schedule, self).fission(toId(loop), side, toId(splitter),
                                             suffix0, suffix1)

    def fuse(self, loop0, loop1):
        """
        Fuse two directly following loops with the same length into one

        To merge nested loops into one, use `merge` instead

        `parallelize`, `unroll` and `vectorize` properties will be reset on the
        fused loop

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
        """
        return super(Schedule, self).fuse(toId(loop0), toId(loop1))

    def swap(self, order):
        """
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
        """
        super(Schedule, self).swap(list(map(toId, order)))

    def blend(self, loop):
        """
        Unroll a loop and interleave statements from each iteration

        E.g.

        ```
        for i = 0 to 2 {
        f(i);
        g(i);
        }
        ```

        will be transformed to be

        ```
        f(0);
        f(1);
        g(0);
        g(1);
        ```

        Virtual threads in TVM can be implemented via blend

        Parameters
        ----------
        loop : str, Stmt or Cursor
            The loop being transformed

        Raises
        ------
        InvalidSchedule
            if the loop is not found, the loop length is not a constant, or
            the dependencies cannot be solved
        """
        super(Schedule, self).blend(toId(loop))

    def cache(self, stmt, var, mtype):
        """
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
        (str, str, str, str)
            (ID of the statement that fills the cache, ID of the statement that
            flushes from the cache, name of the cache variable, ID of the VarDef
            node of the cache variable)
        """
        return super(Schedule, self).cache(toId(stmt), var, parseMType(mtype))

    def cache_reduction(self, stmt, var, mtype):
        """
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
        (str, str, str, str)
            (ID of the statement that initialize the cache, ID of the statement
            that reduces the local result to the global result, name of the
            cache variable, ID of the VarDef node of the cache variable)
        """
        return super(Schedule, self).cache_reduction(toId(stmt), var,
                                                     parseMType(mtype))

    def set_mem_type(self, vardef, mtype):
        """
        Change where a variable is stored

        Parameters
        ----------
        vardef : str, Stmt or Cursor
            ID of the VarDef statement of the specific variable
        mtype : MemType
            Where the variable should be stored

        Raises
        ------
        InvalidSchedule
            if the variable is not found
        """
        super(Schedule, self).set_mem_type(toId(vardef), parseMType(mtype))

    def var_split(self, vardef, dim, mode, factor=-1, nparts=-1):
        """
        Split a dimension of a variable into two

        Parameters
        ----------
        vardef : str, Stmt or Cursor
            ID of the VarDef statement of the specific variable
        dim : int
            which dimension to be split
        mode : VarSplitMode
            When the dimension to split is not divisible by `factor` or `nparts`,
            the resulting shape may become larger. In `FixedSize` mode, the actual
            buffer size will not be changed, and gurads will be added to prevent
            out-of-bound accesses. In `RelaxedSize` mode, the buffer size may
            increase. The `RelaxedSize` mode cannot be applied to I/O variables
        factor : int
            Length of the inner (higher no.) dimension. Set to -1 if using `nparts`
        nparts : int
            Length of the outer (lower no.) loop. Set to -1 if using `factor`

        Raises
        ------
        InvalidSchedule
            if the variable or the dimension is not found
        """
        return super(Schedule, self).var_split(toId(vardef), dim, mode, factor,
                                               nparts)

    def var_reorder(self, vardef, order):
        """
        Reorder the dimensions of a variable

        Parameters
        ----------
        vardef : str, Stmt or Cursor
            ID of the VarDef statement of the specific variable
        order : array like of str, Stmt or Cursor
            Vector of integers. The new order of the dimensions

        Raises
        ------
        InvalidSchedule
            if the variable or the order is illegal
        """
        return super(Schedule, self).var_reorder(toId(vardef), order)

    def move_to(self, stmt, side, dst):
        """
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
        """
        return super(Schedule, self).move_to(toId(stmt), side, toId(dst))

    def inline(self, vardef):
        """
        Remove a variable. When the variable is used, recompute its value

        Parameters
        ----------
        vardef : str, Stmt or Cursor
            The VarDef statement of the specific variable. It can not be an
            I/O varible

        Raises
        ------
        InvalidSchedule
            if the variable cannot be completely removed
        """
        return super(Schedule, self).inline(toId(vardef))

    def parallelize(self, loop, parallel):
        """
        Mark a loop with a parallel implementation

        Parameters
        ----------
        loop : str, Stmt or Cursor
            The loop
        parallel : str
            Parallel implementation. Supported values are "openmp",
            "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x",
            "threadIdx.y", "threadIdx.z"
        """
        super(Schedule, self).parallelize(toId(loop), parallel)

    def unroll(self, loop, immediate=False):
        """
        Unroll a loop

        Parameters
        ----------
        loop : str, Stmt or Cursor
            ID of the loop
        immediate : bool
            If false (by default), postpone the unroll procedure to the backend
            compiler, which saves scheduling time. If true, unroll the loop
            immediately, which may help further simplifications based on the
            unrolled result. If your purpose is just to fill the instruction cache,
            set it to false. If you are unrolling a loop that computes array indices,
            set it to true

        Raises
        ------
        InvalidSchedule
            if the loop is not found or length of the loop is not a constant
        """
        super(Schedule, self).unroll(toId(loop), immediate)

    def vectorize(self, loop):
        """
        Vectorize a loop

        Please note that, as vectorization is different from architecture to
        achitecture, the scheduler may or may not postpone it to the backend
        compiler. The vectorization is a best-effort schedule

        Parameters
        ----------
        loop : str, Stmt or Cursor
            ID of the loop

        Raises
        ------
        InvalidSchedule
            if the ID or name is not found, or the dependency requirement is
            not met
        """
        super(Schedule, self).vectorize(toId(loop))

    def seperate_tail(self):
        """
        Seperate main iterations and tail iterations of a loop

        E.g.

        ```
        for i = 0 -> 3 {
          for j = 0 -> 4 {
             if (i * 4 + j < 10) {
               ...
             }
          }
        }
        ```

        Each loop will be seperated into 2 parts: the body and the tail. After
        simplification, the program will finally be transformed to

        ```
        for i = 0 -> 2 {
          for j = 0 -> 4 {
            ...
          }
        }
        for j = 0 -> 2 {
          ...
        }
        ```

        If there is two VarDef nodes in two branches, it may result in doubled
        memory use, since different thread may go to different branch. Therefore,
        this pass will not duplicate VarDef nodes. (TODO: This restriction may be
        limited to non-local buffers)

        Ideally, all programs can benefit from this schedule. However, this
        schedule may greatly increase the program size and make the compiling
        time way too long. Therefore, this transformation is implemented as a
        schedule, which can be applied optionally. (TODO: Optionally apply this
        schedule to part of the program)
        """
        super(Schedule, self).seperate_tail()

    def as_matmul(self, loop):
        """
        Transform nested loops to be a external call to a matrix multiplication

        Parameters
        ----------
        loop : str, Stmt or Cursor
            ID of the loop

        Raises
        ------
        InvalidSchedule
            if the loop cannot be transformed to be a matrix multiplication
        """
        super(Schedule, self).as_matmul(toId(loop))

    def auto_schedule(self, target):
        """
        Automatic scheduling using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_schedule(target)

    def auto_use_lib(self, target):
        """
        Automatically use external libs using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_use_lib(target)

    def auto_fuse(self, target):
        """
        Automatically fuse consecutive loops using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_fuse(target)

    def auto_parallelize(self, target):
        """
        Automatically parallelize some loops using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_parallelize(target)

    def auto_set_mem_type(self, target):
        """
        Automatically set memory types using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_set_mem_type(target)

    def auto_unroll(self, target):
        """
        Automatically unroll loops using some heuristics

        Parameters
        ----------
        target : Target
            Target architecture
        """
        super(Schedule, self).auto_unroll(target)
