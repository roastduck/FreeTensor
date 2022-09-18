import freetensor as ft
import pytest


def test_pluto_fuse():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(256):
            for j in range(255):
                x[i, j + 1] += x[i, j]
        #! label: L1
        for i in range(256):
            for j in range(255):
                x[i, 254 - j] += x[i, 255 - j]

    print(kernel)
    kernel = ft.schedule(kernel, lambda s: s.pluto_fuse("L0", "L1"))
    print(kernel)


def test_pluto_fuse_2():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(256):
            for j in range(255):
                x[i, j + 1] += x[i, j]
        #! label: L1
        for i in range(256):
            for j in range(255):
                x[255 - i, 254 - j] += x[255 - i, 255 - j]

    print(kernel)
    kernel = ft.schedule(kernel, lambda s: s.pluto_fuse("L0", "L1"))
    print(kernel)


def test_pluto_fuse_3():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(256):
            for j in range(255):
                x[i, j + 1] += x[i, j]
        #! label: L1
        for i in range(256):
            for j in range(254, -1, -1):
                x[255 - i, j] += x[255 - i, j + 1]

    print(kernel)
    kernel = ft.schedule(kernel, lambda s: s.pluto_fuse("L0", "L1"))
    print(kernel)
