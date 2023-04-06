__all__ = ['logger', 'check_conflict_id', 'with_line_no']

import itertools

from freetensor_ffi import logger
from freetensor_ffi import check_conflict_id


def with_line_no(s):
    s = str(s)
    lines = list(s.splitlines())
    maxNuLen = len(str(len(lines)))
    fmt = "{:%dd}" % maxNuLen
    return "\n".join(
        map(
            lambda arg: "\033[33m" + fmt.format(arg[1] + 1) + "\033[0m" + " " +
            arg[0],
            zip(lines, itertools.count()),
        ))
