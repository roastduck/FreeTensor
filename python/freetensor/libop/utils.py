def named_partial(name: str, f, *args, **kvs):
    ''' Similar to functools.partial, but it sets the returned function's __name__ '''

    def g(*_args, **_kvs):
        return f(*args, *_args, **kvs, **_kvs)

    g.__name__ = name
    return g


def begin_with_0(lst):
    return len(lst) > 0 and lst[0] == 0


def all_minus_one(lst):
    return list(map(lambda x: x - 1, lst))
