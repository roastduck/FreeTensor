''' Global configurations '''

import freetensor_ffi as ffi


def _import_func(f):
    ''' Make our doc builder know we have this function '''

    def g(*args, **kvs):
        return f(*args, **kvs)

    g.__name__ = f.__name__
    g.__qualname__ = f.__qualname__
    if hasattr(f, '__annotations__'):
        g.__annotations__ = f.__annotations__
    g.__doc__ = f.__doc__
    return g


with_mkl = _import_func(ffi.with_mkl)

with_cuda = _import_func(ffi.with_cuda)

set_pretty_print = _import_func(ffi.set_pretty_print)
pretty_print = _import_func(ffi.pretty_print)

set_print_all_id = _import_func(ffi.set_pretty_print)
print_all_id = _import_func(ffi.pretty_print)

set_werror = _import_func(ffi.set_werror)
werror = _import_func(ffi.werror)

set_default_target = _import_func(ffi.set_default_target)
default_target = _import_func(ffi.default_target)

set_default_device = _import_func(ffi.set_default_device)
default_device = _import_func(ffi.default_device)
