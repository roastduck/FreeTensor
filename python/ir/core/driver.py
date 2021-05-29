import ffi

from ffi import Target, CPU, GPU, Device, Array


class Driver(ffi.Driver):

    def __init__(self, func, src, dev):
        super(Driver, self).__init__(func, src, dev)

    def set_params(self, **kws):
        super(Driver, self).set_params(kws)
