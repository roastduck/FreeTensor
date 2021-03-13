import itertools

def with_line_no(s: str):
    return "\n".join(map(
        lambda arg: '\033[33m' + str(arg[1] + 1) + '\033[0m' + ' ' + arg[0],
        zip(s.splitlines(), itertools.count())))

