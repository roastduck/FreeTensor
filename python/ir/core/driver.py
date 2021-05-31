import ffi

from ffi import Target, CPU, GPU, Device, Array


class Driver(ffi.Driver):

    def __init__(self, func, src, dev):
        super(Driver, self).__init__(func, src, dev)

    def set_params(self, *args, **kws):
        super(Driver, self).set_params(args, kws)

    def __call__(self, *args, **kws):
        self.set_params(*args, **kws)
        self.run()
