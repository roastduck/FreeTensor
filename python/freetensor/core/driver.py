__all__ = [
    'Array', 'array', 'move', 'TargetType', 'Target', 'Device', 'CPU', 'GPU',
    'Driver', 'build_binary'
]

from typing import Optional, Callable, Sequence
import functools
import numpy as np

from .. import ffi
from ..ffi import TargetType, Target, Device, Array

from . import config
from .codegen import NativeCode
from .enable_attach_backward import EnableAttachBackward
from .return_values_pack import ReturnValuesPack
from .jit import JITTemplate
from .meta import DataType, to_numpy_dtype, to_torch_dtype
from .utils import as_decorator


def array(data,
          dtype=None,
          dont_drop_borrow: bool = False,
          moved: bool = False):
    '''
    Factory function for Array

    This function is preferred over directly calling `Array`'s constructor, because
    it accepts more data format.

    - If `data` is another FreeTensor `Array`, the original object will be returned,
    with `dont_drop_borrow` and `moved` set to new values. If `dtype` is set and
    different from the original data type, the `Array` will be copied first to convert
    the data type.
    - If `data` is Numpy `Array` or PyTorch `Tensor`, it will be converted to FreeTensor
    `Array`. Memory copy will be avoided in most cases, but it is inevitable if the
    data is strided. If `dtype` is set and different from the original data type, the
    `Array` or `Tensor` will be copied first to convert the data type.
    - Otherwise, the data will be treated as an n-dimensional array-like object, and
    will be parsed according the rules in NumPy. The data type is also set accordingly,
    unless `dtype` is set.

    Parameters
    ----------
    data : FreeTensor Array, Numpy Array, PyTorch Tensor, or other array-like objects
        Data to be copied to or borrowed by the new Array object
    dtype : ft.DataType or str
        If `data` is not in `dtype`, convert it to `dtype` first before constructing
        the `Array`
    dont_drop_borrow : bool
        If true, report an error if we have to drop a borrwed data. This flag is set
        to true when the Array is cunstructed IMPLICITLY (not by this function) from
        a user object by borrowing from it, where users may expect they are acutually
        manipulating the their user object, instead of this Array
    moved : bool
        If true, it means we do not care about data in this Array any more after the
        program runs. Variables with "input-mutable" access type may modify the Array
    '''

    if dtype is not None:
        dtype = DataType(dtype)

    if type(data) is Array:
        if dtype is not None and dtype != data.dtype:
            # Must be contiguous
            data = Array(data.numpy().astype(to_numpy_dtype(dtype)))
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
        if dtype is not None and to_numpy_dtype(dtype) != data.dtype:
            data = data.astype(to_numpy_dtype(dtype), order='C')
        elif not data.flags['C_CONTIGUOUS']:
            data = data.copy(order='C')
        return Array(data, dont_drop_borrow, moved)

    if data.__class__.__module__ == 'torch':
        import torch
        if type(data) is torch.Tensor:
            if not config.with_pytorch():
                raise ffi.InvalidIO(
                    "FreeTensor should be built with WITH_PYTORCH to accept a PyTorch tensor"
                )
            if dtype is not None and to_torch_dtype(dtype) != data.dtype:
                data = data.to(to_torch_dtype(dtype),
                               memory_format=torch.contiguous_format)
            elif not data.is_contiguous():
                data = data.contiguous()
            return Array(data, dont_drop_borrow, moved)

    return array(np.array(
        data, dtype=None if dtype is None else to_numpy_dtype(dtype)),
                 dtype=dtype,
                 dont_drop_borrow=dont_drop_borrow,
                 moved=moved)


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


def _register_device(cls):

    def __enter__(self: cls):
        '''
        A Device can be used as a "with" scope, then all the `Array`s and `Driver`s
        will use it by default. In this style, it also sets the default Target. E.g:

        ```
        with Device(...):
            ast = lower(ast)  # Use the Target of the Device above by default
            a = Array(...)  # Use the Device above by default
        ```
        '''
        _old_target_device_stack.append(
            (config.default_target(), config.default_device()))
        config.set_default_target(self.target())
        config.set_default_device(self)
        return self

    def __exit__(self: cls, exc_type, exc_value, traceback):
        old_target, old_device = _old_target_device_stack.pop()
        config.set_default_target(old_target)
        config.set_default_device(old_device)

    cls.__enter__ = __enter__
    cls.__exit__ = __exit__


_register_device(Device)


class CPU(Device):

    def __init__(self, *args):
        super().__init__(TargetType.CPU, *args)


class GPU(Device):

    def __init__(self, *args):
        super().__init__(TargetType.GPU, *args)


class Driver(EnableAttachBackward, ffi.Driver):

    def __init__(self,
                 native_code: NativeCode,
                 device: Optional[Device] = None,
                 host_device: Optional[Device] = None,
                 cxx_flags: Sequence[str] = [],
                 verbose: bool = False):
        '''
        Compile a program using a backend compiler and load it into memory

        This class is for internal use. Please consider using `build_binary`

        Parameters
        ----------
        native_code : NativeCode
            Native code generated from codegen
        device : Device (Optional)
            The device to run the program. If omitted, use the default device
            in config
        cxx_flags : Sequence[str]
            Additional C++ flags passed to the backend compiler
        verbose : bool (Optional)
            True to print extra infomation
        '''
        self._native_code = native_code
        if device is None:
            device = config.default_device()
        if host_device is None:
            super(Driver, self).__init__(native_code, device, cxx_flags,
                                         verbose)
        else:
            super(Driver, self).__init__(native_code, device, host_device,
                                         cxx_flags, verbose)

        # When we pass numpy or pytorch tensors to `set_args`, they are
        # converted to `Array` objects by reference. In `Array`'s FFI, we
        # keep these tensors alive whenever the `Array`'s PYTHON objects
        # alive. We need to also keep the `Array`'s PYTHON objects here.
        # Please note that we cannot hold the reference count in `Driver`'s
        # C++ implementation, where we can only hold the `Array`'s C++
        # objects alive.
        self.args_ref_cnt_holder = []

    def native_code(self) -> NativeCode:
        ''' Get native code compiled by backend compiler '''
        return self._native_code

    def set_args(self, *args, **kws):
        ''' Set argument for an invocation '''

        # No need to hold reference of the last run any more
        self.args_ref_cnt_holder = []

        args = list(args)
        kws = dict(kws)
        for i in range(len(args)):
            args[i] = array(args[i],
                            dont_drop_borrow=not isinstance(args[i], Array))
        for key in kws:
            kws[key] = array(kws[key],
                             dont_drop_borrow=not isinstance(kws[key], Array))

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
                           self._native_code.returns)), values)

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
        try:
            self.run()
        finally:  # Always collect returns or there will be memory leak
            ret = self.collect_returns()
        return ret

    def time(self, *args, kws={}, rounds=10, warmups=3):
        '''
        Measure running time. The return is dropped.

        Returns
        -------
        Tuple[float, float]
            - [0] = average time, in ms
            - [1] = estimated standard deviation of the average time = sqrt(Var(X1 +
            X2 + ... + Xn))), in ms
        '''
        self.set_args(*args, **kws)
        t = super().time(rounds=rounds, warmups=warmups)
        self.collect_returns()  # Must collect. Then we drop the result
        return t


@as_decorator
def build_binary(code: Optional[NativeCode] = None,
                 device: Optional[Device] = None,
                 host_device: Optional[Device] = None,
                 jit_cache: Callable[Callable, Callable] = functools.cache,
                 cxx_flags: Sequence[str] = [],
                 verbose: bool = False):
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
    jit_cache : Callable[Callable, Callable]
        Function decorator used to cache JIT instances
    cxx_flags : Sequence[str]
        Additional C++ flags passed to the backend compiler
    verbose : int
        Verbosity level

    Returns
    -------
    Driver or JITTemplate
        Return a Driver for the executable if there is no JIT parameters.
        Return a JITTemplate that generates a Driver if there is at least one
    '''

    # Note for JIT: `build_binary` should respect the default target when it is called
    if device is None:
        device = config.default_device()

    if isinstance(code, JITTemplate):

        class BuildBinaryTemplate(JITTemplate):

            def __init__(self, *args, **kvs):
                super().__init__(*args, **kvs)
                self.args = None
                self.kvs = None

            @jit_cache
            def instantiate_by_only_jit_args(self, *jit_args):
                return build_binary(
                    code.instantiate_by_only_jit_args(*jit_args),
                    device=device,
                    host_device=host_device,
                    cxx_flags=cxx_flags,
                    verbose=verbose)

            def __call__(self, *args, **kvs):
                ''' Helper to directly run the template '''

                self.args = args
                self.kvs = kvs
                return self.instantiate_and_call(*args, **kvs)

            @property
            def backward(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .backward")
                # TODO: Here we re-instantiate. Change to another implementation if the overhead
                # is too much
                return self.instantiate(*self.args, **self.kvs).backward

            @property
            def input_name_to_gradient_name(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .input_name_to_gradient_name")
                return self.instantiate(*self.args,
                                        **self.kvs).input_name_to_gradient_name

            @property
            def output_name_to_gradient_name(self):
                ''' Helper to act as an EnableAttachBackward object '''

                if self.args is None or self.kvs is None:
                    raise TypeError(
                        "A JIT program requires first running the forward pass before getting"
                        " .output_name_to_gradient_name")
                return self.instantiate(*self.args,
                                        **self.kvs).output_name_to_gradient_name

        return BuildBinaryTemplate(code.params, code.jit_param_names)

    if device.target() != code.target:
        raise ffi.DriverError(
            f"Codegen target ({code.target}) is inconsistent with device target ({device.target()})"
        )
    ret = Driver(code, device, host_device, cxx_flags, verbose)

    if code.has_backward():
        ret.attach_backward(
            build_binary(code.backward,
                         device=device,
                         host_device=host_device,
                         jit_cache=jit_cache,
                         cxx_flags=cxx_flags,
                         verbose=verbose), code.input_name_to_gradient_name,
            code.output_name_to_gradient_name)
    return ret
