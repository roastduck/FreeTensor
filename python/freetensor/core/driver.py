__all__ = [
    'Array', 'array', 'move', 'TargetType', 'Target', 'Device', 'CPU', 'GPU',
    'ReturnValuesPack', 'Driver', 'build_binary'
]

import freetensor_ffi as ffi
import functools
import numpy as np

from typing import Optional, Sequence
from freetensor_ffi import TargetType, Target, Array

from . import config
from .codegen import NativeCode


def array(data, dont_drop_borrow: bool = False, moved: bool = False):
    '''
    Factory function for Array

    It converts more data format to Array

    Parameters
    ----------
    data : Numpy Array, PyTorch Tensor, or another FreeTensor Array
        Data to be copied to or borrowed by the new Array object
    dont_drop_borrow : bool
        If true, report an error if we have to drop a borrwed data. This flag is set
        to true when the Array is cunstructed IMPLICITLY (not by this function) from
        a user object by borrowing from it, where users may expect they are acutually
        manipulating the their user object, instead of this Array
    moved : bool
        If true, it means we do not care about data in this Array any more after the
        program runs. Variables with "input-mutable" access type may modify the Array
    '''

    if type(data) is Array:
        data.set_dont_drop_borrow(dont_drop_borrow)
        data.set_moved(moved)
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
        return Array(data, dont_drop_borrow, moved)

    if data.__class__.__module__ == 'torch':
        import torch
        if type(data) is torch.Tensor:
            if not config.with_pytorch():
                raise ffi.InvalidIO(
                    "FreeTensor should be built with WITH_PYTORCH to accept a PyTorch tensor"
                )
            if not data.is_contiguous():
                data = data.contiguous()
            return Array(data, dont_drop_borrow, moved)

    raise ffi.InvalidIO(f"Unsupported data type {type(data)} for Array")


def move(data):
    ''' Alias for array(data, dont_drop_borrow=False, moved=True) '''

    return array(data, dont_drop_borrow=False, moved=True)


_old_target_device_stack = []


def _register_target(cls):

    def __enter__(self: cls):
        '''
        A Target can be used as a "with" scope, then all the `lower`s and `codegen`s
        will use it by default. In this style, it also sets the default Device as the
        0-th device of the given Target. E.g:

        ```
        with Target(...):
            ast = lower(ast)  # Use the Target above by default
            a = Array(...)  # Use the 0-th device of the Target above by default
        ```
        '''

        _old_target_device_stack.append(
            (config.default_target(), config.default_device()))
        config.set_default_target(self)
        config.set_default_device(Device(self.type(), 0))
        return self

    def __exit__(self: cls, exc_type, exc_value, traceback):
        old_target, old_device = _old_target_device_stack.pop()
        config.set_default_target(old_target)
        config.set_default_device(old_device)

    cls.__enter__ = __enter__
    cls.__exit__ = __exit__


_register_target(ffi.CPUTarget)
if config.with_cuda():
    _register_target(ffi.GPUTarget)


class Device(ffi.Device):
    '''

    A computing device can be constructed from
         1. (TargetType, DeviceNumber)
         2. (TargetType, getDeviceByName): cuda uses best matches criteria.
         3. (TargetType, FullName, nth): get nth(from 0) device named `Fullname`.

    E.g. Device(TargetType::GPU, 0) means the 0-th GPU (device)
         Device(TargetType::GPU, "V100") means a GPU which best matches "V100"
         Device(TargetType::GPU, "NVIDIA GeForce RTX 3060 Laptop GPU", 0)

    A Device can be used as a "with" scope, then all the `Array`s and `Driver`s
    will use it by default. In this style, it also sets the default Target. E.g:

    ```
    with Device(...):
        ast = lower(ast)  # Use the Target of the Device above by default
        a = Array(...)  # Use the Device above by default
    ```
    '''

    def __enter__(self):
        _old_target_device_stack.append(
            (config.default_target(), config.default_device()))
        config.set_default_target(self.target())
        config.set_default_device(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        old_target, old_device = _old_target_device_stack.pop()
        config.set_default_target(old_target)
        config.set_default_device(old_device)


class CPU(Device):

    def __init__(self, *args):
        super().__init__(TargetType.CPU, *args)


class GPU(Device):

    def __init__(self, *args):
        super().__init__(TargetType.GPU, *args)


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

    def __contains__(self, key):
        ''' Test if a return value exists '''
        for k, v in zip(self.keys, self.values):
            if k == key:
                return True
        return False


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
        self.src = str(src)
        if device is None:
            device = config.default_device()
        if verbose is None:
            verbose = False
        if host_device is None:
            super(Driver, self).__init__(func, self.src, device, verbose)
        else:
            super(Driver, self).__init__(func, self.src, device, host_device,
                                         verbose)
        self.func = func

        # When we pass numpy or pytorch tensors to `set_args`, they are
        # converted to `Array` objects by reference. In `Array`'s FFI, we
        # keep these tensors alive whenever the `Array`'s PYTHON objects
        # alive. We need to also keep the `Array`'s PYTHON objects here.
        # Please note that we cannot hold the reference count in `Driver`'s
        # C++ implementation, where we can only hold the `Array`'s C++
        # objects alive.
        self.args_ref_cnt_holder = []

    def native_code(self):
        ''' Get native code compiled by backend compiler '''
        return self.src

    def set_args(self, *args, **kws):
        ''' Set argument for an invocation '''

        # No need to hold reference of the last run any more
        self.args_ref_cnt_holder = []

        args = list(args)
        kws = dict(kws)
        for i in range(len(args)):
            args[i] = array(args[i], not isinstance(args[i], Array))
        for key in kws:
            kws[key] = array(kws[key], not isinstance(kws[key], Array))

        for arg in args:
            self.args_ref_cnt_holder.append(arg)
        for key in kws:
            self.args_ref_cnt_holder.append(kws[key])

        super(Driver, self).set_args(args, kws)

    def collect_returns(self, always_return_pack: bool = False):
        '''
        Collect return values from an invocation

        Return values must be collect. Otherwise there will be memory leaks

        If there is only one return value, it is returned directly. Otherwise,
        or if `always_return_pack` is set, the return values are packed in a
        ReturnValuesPack
        '''
        values = super(Driver, self).collect_returns()
        if len(values) == 0 and not always_return_pack:
            return None
        elif len(values) == 1 and not always_return_pack:
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
