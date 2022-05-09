import itertools

from ffi import logger


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
