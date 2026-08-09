__all__ = [
    'structural_feature', 'find_stmt', 'find_all_stmt', 'find_deps',
    'find_deps_exists', 'DepDirection', 'FindDepsMode', 'FindDepsAxis',
    'AccessPointSnapshot', 'AxisDirection', 'DependenceSnapshot', 'DEP_WAW',
    'DEP_WAR', 'DEP_RAW', 'DEP_ALL'
]

from .. import ffi
from ..ffi import (AccessPointSnapshot, AxisDirection, DEP_ALL, DEP_RAW,
                   DEP_WAR, DEP_WAW, DepDirection, DependenceSnapshot,
                   FindDepsAxis, FindDepsMode, find_all_stmt, find_stmt,
                   structural_feature)


def _as_find_deps_axis(axis):
    if isinstance(axis, FindDepsAxis):
        return axis
    if isinstance(axis, str):
        return FindDepsAxis(ffi.ParallelScope(axis))
    if hasattr(axis, 'id'):
        return FindDepsAxis(axis.id)
    return FindDepsAxis(axis)


def _normalize_direction(direction):
    if direction is None:
        return None
    return [[(_as_find_deps_axis(axis), dep_dir)
             for axis, dep_dir in conj]
            for conj in direction]


def _normalize_filter_sub_ast(filter_sub_ast):
    if filter_sub_ast is None:
        return None
    if hasattr(filter_sub_ast, 'id'):
        return filter_sub_ast.id
    return filter_sub_ast


def find_deps(ast,
              *,
              dep_type=DEP_ALL,
              mode=FindDepsMode.Dep,
              direction=None,
              filter_sub_ast=None,
              ignore_reduction_waw=True,
              erase_outside_vardef=True,
              no_project_out_private_axis=False):
    '''Find RAW, WAR, and WAW memory dependences in a statement AST.

    Parameters
    ----------
    ast : Stmt
        AST root to analyze.
    dep_type : int
        Bitmask selecting dependence kinds. Use ``DEP_WAW``, ``DEP_WAR``,
        ``DEP_RAW``, ``DEP_ALL``, or a bitwise-or of them. Defaults to all
        dependence kinds.
    mode : FindDepsMode
        Controls whether each reported dependence only has to be possible, or
        whether it must cover all instances of one side:

        - ``FindDepsMode.Dep``: no coverage restriction.
        - ``FindDepsMode.KillEarlier``: every instance of the earlier access is
          dependent by the later access.
        - ``FindDepsMode.KillLater``: every instance of the later access depends
          on the earlier access.
        - ``FindDepsMode.KillBoth``: both ``KillEarlier`` and ``KillLater``.

        Killing tests are insensitive to loop-invariant expressions and may have
        false negatives.
    direction : Sequence[Sequence[Tuple[axis, DepDirection]]] | None
        Direction constraints on loops or parallel scopes, in OR-of-AND form.
        For example, ``[[(L1, Same), (L2, Normal)]]`` means dependences inside
        one iteration of ``L1`` and along ``L2``; ``[[(L1, Same)], [(L2,
        Normal)]]`` means either of these constraints is sufficient. An ``axis``
        can be an ``ID``, a ``Stmt``, or a parallel-scope string. ``None`` means
        no restriction. An empty outer sequence matches no dependences.
    filter_sub_ast : ID | Stmt | None
        Restrict analysis to accesses whose containing statements are inside the
        specified sub-AST. ``None`` means the whole input AST is analyzed.
    ignore_reduction_waw : bool
        Ignore WAW dependences between two ``ReduceTo`` nodes. These dependences
        are false dependences for serial execution. Defaults to ``True``.
    erase_outside_vardef : bool
        Ignore dependences outside a variable's ``VarDef`` scope. Defaults to
        ``True``.
    no_project_out_private_axis : bool
        Disable the private-axis projection optimization. Set this to ``True``
        when inspecting the Presburger map strings stored in returned
        ``DependenceSnapshot`` objects. When this is ``False``, these fields are
        ``None`` because projected maps are not meaningful to users.

    Returns
    -------
    list[DependenceSnapshot]
        Copied dependence snapshots that remain valid after this function
        returns.
    '''
    return ffi.find_deps(ast, dep_type, mode, _normalize_direction(direction),
                         _normalize_filter_sub_ast(filter_sub_ast),
                         ignore_reduction_waw, erase_outside_vardef,
                         no_project_out_private_axis)


def find_deps_exists(ast,
                     *,
                     dep_type=DEP_ALL,
                     mode=FindDepsMode.Dep,
                     direction=None,
                     filter_sub_ast=None,
                     ignore_reduction_waw=True,
                     erase_outside_vardef=True,
                     no_project_out_private_axis=False):
    '''Return whether memory dependences matching the options exist.

    The keyword arguments have the same meanings as in :func:`find_deps`.
    '''
    return ffi.find_deps_exists(ast, dep_type, mode,
                                _normalize_direction(direction),
                                _normalize_filter_sub_ast(filter_sub_ast),
                                ignore_reduction_waw, erase_outside_vardef,
                                no_project_out_private_axis)
