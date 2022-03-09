import ctypes

_cudart = ctypes.CDLL('libcudart.so')


def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)
