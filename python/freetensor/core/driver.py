import ffi

from typing import Optional
from ffi import CPU, GPU, Array

from . import config


class Target(ffi.Target):
    '''
    A target architecture

    A Target can be used as a "with" scope, then all the `lower`s and `codegen`s
    will use it by default. E.g:

    ```
    with Target(...):
        ast = lower(ast)  # Use the Target above by default
    ```
    '''

    def __init__(self, use_native_arch: bool = True):
        super(Target, self).__init__(use_native_arch)

    def __enter__(self):
        self.old_target = config.default_target()
        config.set_default_target(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        config.set_default_target(self.old_target)


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


class Driver(ffi.Driver):

    def __init__(self, func: ffi.Func, src: str, dev: Optional[Device] = None):
        '''
        Compile a program using a backend compiler and load it into memory

        Parameters
        ----------
        func : ffi.Func
            AST of the function, where the function signature is needed to
            determine the parameters and return values
        src : str
            Native code generated from codegen
        dev : Device (Optional)
            The device to run the program. If omitted, use the default device
            in config
        '''
        if dev is not None:
            super(Driver, self).__init__(func, src, dev)
        else:
            super(Driver, self).__init__(func, src)

    def set_params(self, *args, **kws):
        super(Driver, self).set_params(args, kws)

    def __call__(self, *args, **kws):
        self.set_params(*args, **kws)
        self.run()
        return self.collect_returns()
