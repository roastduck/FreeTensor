import functools


def as_decorator(f):
    '''
    Enable a multi-parameter function `f` to be used as a decorator

    Suppose `g = as_decorator(f)`, enable the following usages:

    ```
    @g
    def h(...):
        ...

    @g(a=a, b=b, c=c)
    def h(...):
        ...
    ```

    Formally, `g` will have the same parameters as `f`. `f`'s first parameter should
    be the function it decorate, say `h`, and may have other parameters with default
    values. If `h` is set when called, `g` will return the decorated function, just
    as `f` does. If `h` is not set, `g` will return an `f`'s partial function with
    all other parameters set, and the partial function can then be decorate another
    `h` again.
    '''

    def g(h=None, *args, **kvs):
        if h is not None:
            # g(h, a=a, b=b, c=c)
            return f(h, *args, **kvs)
        elif len(args) == 0:
            # g(a=a, b=b, c=c)(h)
            return functools.partial(f, **kvs)
        else:
            # g(None, x, y, a=a, b=b)
            raise TypeError(
                f"Please do not explicitly pass None as a missing first argument to "
                f"{f.__name__}")

    return functools.wraps(f)(g)
