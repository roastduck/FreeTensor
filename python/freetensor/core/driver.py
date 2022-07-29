import freetensor_ffi as ffi
import functools
import numpy as np

from typing import Optional, Sequence
from freetensor_ffi import CPU, GPU, Device, Array

from . import config
from .codegen import NativeCode


def array(data):
    '''
    Factory function for Array

    It converts more data format to Array
    '''

    if type(data) is Array:
        return data

    # For NumPy, Although Pybind11's `array_t` type provides a flag `forcecast` to
    # cast from a strided array to a contiguous one. But it always casts to a specific
    # type, e.g. float64. I have no idea how to support multiple types. Therfore,
    # we have to call NumPy's `.copy(order='C')` to make a new NumPy array. This
    # function can only be called from Python side (not from PyBind11's `py::array`
    # type).
    if type(data) is np.ndarray:
        if not data.flags['C_CONTIGUOUS']:
            data = data.copy(order='C')
        return Array(data)

    if data.__class__.__module__ == 'torch':
        import torch
        if type(data) is torch.Tensor:
            if not config.with_pytorch():
                raise ffi.DriverError(
                    "FreeTensor should be built with WITH_PYTORCH to accept a PyTorch tensor"
                )
            if not data.is_contiguous():
                data = data.contiguous()
            return Array(data)

    raise ffi.DriverError(f"Unsupported data type {type(data)} for Array")


class Target(ffi.Target):
    '''
    A target architecture

    A Target can be used as a "with" scope, then all the `lower`s and `codegen`s
    will use it by default. In this style, it also sets the default Device as the
    0-th device of the given Target. E.g:

    ```
    with Target(...):
        ast = lower(ast)  # Use the Target above by default
        a = Array(...)  # Use the 0-th device of the Target above by default
    ```
    '''

    def __init__(self, use_native_arch: bool = True):
        super(Target, self).__init__(use_native_arch)

    def __enter__(self):
        self.old_target = config.default_target()
        self.old_device = config.default_device()
        config.set_default_target(self)
        config.set_default_device(Device(self, 0))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        config.set_default_target(self.old_target)
        config.set_default_device(self.old_device)


class Device(ffi.Device):
    '''
    A computing device of a Target

    E.g. suppose GPU() is a Target (architecture), then Device(GPU(), 0) means
    the 0-th GPU (device)

    A Device can be used as a "with" scope, then all the `Array`s and `Driver`s
    will use it by default. In this style, it also sets the default Target. E.g:

    ```
    with Device(...):
        ast = lower(ast)  # Use the Target of the Device above by default
        a = Array(...)  # Use the Device above by default
    ```
    '''

    def __init__(self, target: Target, num: int = 0):
        super(Device, self).__init__(target, num)

    def __enter__(self):
        self.old_target = config.default_target()
        self.old_device = config.default_device()
        config.set_default_target(self.target())
        config.set_default_device(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        config.set_default_target(self.old_target)
        config.set_default_device(self.old_device)


class ReturnValuesPack:
    '''
    Hold return values from a Driver invocation

    Return values can be retrieved in an anonymous manner: `x, y, z = pack`,
    or in a named manner: `pack['x']`

    Please note that a ReturnValuesPack is different from a OrderedDict, as
    OrderedDict unpacks to keys rather than values
    '''

    def __init__(self, keys: Sequence[str], values: Sequence[Array]):
        keys = list(keys)
        values = list(values)
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values

    def __iter__(self):
        ''' Get all return values in the order declared in Func '''
        yield from self.values

    def __getitem__(self, key) -> Array:
        ''' Get a return value with a name. Tuple is supported for multiple values '''
        if type(key) is tuple or type(key) is list:
            ret = []
            for k in key:
                ret.append(self[k])
            return ret
        for k, v in zip(self.keys, self.values):
            if k == key:
                return v
        raise ffi.DriverError("No such return value named " + key)


class Driver(ffi.Driver):

    def __init__(self,
                 func: ffi.Func,
                 src: str,
                 device: Optional[Device] = None,
                 host_device: Optional[Device] = None,
                 verbose: Optional[bool] = None):
        '''
        Compile a program using a backend compiler and load it into memory

        This class is for internal use. Please consider using `build_binary`

        Parameters
        ----------
        func : ffi.Func
            AST of the function, where the function signature is needed to
            determine the parameters and return values
        src : str
            Native code generated from codegen
        device : Device (Optional)
            The device to run the program. If omitted, use the default device
            in config
        verbose : bool (Optional)
            True to print extra infomation
        '''
        src = str(src)
        if device is None:
            device = config.default_device()
        if verbose is None:
            verbose = False
        if host_device is None:
            super(Driver, self).__init__(func, src, device, verbose)
        else:
            super(Driver, self).__init__(func, src, device, host_device,
                                         verbose)
        self.func = func

    def set_args(self, *args, **kws):
        ''' Set argument for an invocation '''
        args = list(args)
        kws = dict(kws)
        for i in range(len(args)):
            args[i] = array(args[i])
        for key in kws:
            kws[key] = array(kws[key])
        super(Driver, self).set_args(args, kws)

    def collect_returns(self):
        '''
        Collect return values from an invocation

        Return values must be collect. Otherwise there will be memory leaks

        If there is only one return value, it is returned directly. Otherwise,
        the return values are packed in a ReturnValuesPack
        '''
        values = super(Driver, self).collect_returns()
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return values[0]
        else:
            return ReturnValuesPack(
                map(
                    lambda r: r.name,
                    filter(lambda r: not r.is_in_closure or r.return_closure,
                           self.func.returns)), values)

    def __call__(self, *args, **kws):
        '''
        Set argument, execute the binary code, and collect the returns

        If there is only one return value, it is returned directly. Otherwise,
        the return values are packed in a ReturnValuesPack

        This function will introduce some overhaed handling arguments and return
        values. For an accurate execution time measurement, plase call
        `self.set_args` first, then `self.time`, and finally `self.collect_returns`
        '''
        self.set_args(*args, **kws)
        self.run()
        return self.collect_returns()


def build_binary(code: Optional[NativeCode] = None,
                 device: Optional[Device] = None,
                 host_device: Optional[Device] = None,
                 verbose: Optional[bool] = None):
    '''
    Compile a program using a backend compiler and load it into memory

    Parameters
    ----------
    code : NativeCode
        Native code generated by `codegen`. If not specified, a partial
        function is returned, which can be used as a decorator
    device : Device (Optional)
        The device to run the program. If omitted, use the default device
        in config
    '''

    if code is not None:
        if device is None:
            device = config.default_device()
        if device.target() != code.target:
            raise ffi.DriverError(
                f"Codegen target ({code.target}) is inconsistent with device target ({device.target()})"
            )
        return Driver(code.func, code.code, device, host_device, verbose)
    else:
        f = build_binary
        if device is not None:
            f = functools.partial(f, device=device)
        if host_device is not None:
            f = functools.partial(f, host_device=host_device)
        if verbose is not None:
            f = functools.partial(f, verbose=verbose)
        return f
