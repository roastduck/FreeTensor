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
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    kernel = s.func()
    print(kernel)
    assert parallelism == 1


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
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    assert parallelism == 1
    kernel = s.func()
    print(kernel)


def test_pluto_fuse_3():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(256):
            for j in range(255):
                x[i, j + 1] += x[i, j]
        #! label: L1
        for i in range(255, -1, -1):
            for j in range(254, -1, -1):
                x[i, j] += x[i, j + 1]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    assert parallelism == 1
    kernel = s.func()
    print(kernel)


def test_pluto_fuse_reversed():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(255, -1, -1):
            for j in range(255):
                x[i, j + 1] += x[i, j]
        #! label: L1
        for i in range(255, -1, -1):
            for j in range(255):
                x[255 - i, 254 - j] += x[255 - i, 255 - j]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    assert parallelism == 1
    kernel = s.func()
    print(kernel)


def test_pluto_fuse_imbalanced_nest():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"],
               c: ft.Var[(256,), "float32", "inout"]):
        #! label: L0
        for j in range(1, 256):
            c[j] = j
        #! label: L1
        for i in range(256):
            for j in range(1, 256):
                x[i, j] *= c[j - 1] + c[j]

    @ft.transform
    def kernel_expected(x: ft.Var[(256, 256), "float32", "inout"],
                        c: ft.Var[(256,), "float32", "inout"]):
        for j in range(0, 255):
            c[j + 1] = j + 1
            for i in range(256):
                x[i, j + 1] *= c[j] + c[j + 1]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    kernel = s.func()
    print(kernel)
    assert parallelism == 0
    assert kernel.body.match(kernel_expected.body)


def test_pluto_fuse_modulo():

    @ft.transform
    def kernel(x: ft.Var[(256,), "float32", "inout"],
               xc: ft.Var[(128,), "float32", "output"]):
        #! label: L0
        for i in range(1, 256):
            x[i] += x[i - 1]
        #! label: L1
        for i in range(128):
            xc[i] = x[i * 2 + 1]

    @ft.transform
    def kernel_expected(x: ft.Var[(256,), "float32", "inout"],
                        xc: ft.Var[(128,), "float32", "output"]):
        #! label: L0
        for i in range(0, 255):
            x[i + 1] += x[i]
            if i % 2 == 0:
                xc[i // 2] = x[i + 1]

    print(kernel)
    s = ft.Schedule(kernel)
    s.pluto_fuse("L0", "L1")
    kernel = s.func()
    print(kernel)
    assert kernel.body.match(kernel_expected.body)


def test_pluto_permute_reorder():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        #! label: L0
        for i in range(1, 256):
            for j in range(256):
                x[i, j] += x[i - 1, j]

    @ft.transform
    def kernel_expected(x: ft.Var[(256, 256), "float32", "inout"]):
        for j in range(256):
            for i in range(1, 256):
                x[i, j] += x[i + -1, j]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_permute("L0")
    kernel = s.func()
    print(kernel)
    assert parallelism == 1
    assert kernel.body.match(kernel_expected.body)


def test_pluto_permute_fully_parallelizable():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "output"],
               y: ft.Var[(258, 258), "float32", "input"]):
        #! label: L0
        for i in range(256):
            for j in range(256):
                x[i, j] = y[i - 1, j - 1] + y[i + 1, j + 1]

    @ft.transform
    def kernel_expected(x: ft.Var[(256, 256), "float32", "output"],
                        y: ft.Var[(258, 258), "float32", "input"]):
        #! label: L0
        for i in range(256):
            for j in range(256):
                x[i, j] = y[i + -1, j + -1] + y[i + 1, j + 1]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_permute("L0")
    kernel = s.func()
    print(kernel)
    assert parallelism == 2
    assert kernel.body.match(kernel_expected.body)


def test_pluto_permute_inner_loop():

    @ft.transform
    def kernel(x: ft.Var[(256, 256, 5), "float32", "inout"]):
        #! label: L0
        for i in range(1, 256):
            for j in range(256):
                x[i, j] = x[i - 1, j] * 5

    @ft.simplify
    @ft.transform
    def kernel_expected(x: ft.Var[(256, 256, 5), "float32", "inout"]):
        #! label: L0
        for j in range(256):
            for i in range(1, 256):
                x[i, j] = x[i + -1, j] * 5

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_permute("L0")
    kernel = s.func()
    print(kernel)
    assert parallelism == 1
    assert kernel.body.match(kernel_expected.body)


def test_pluto_permute_outer_loop():

    @ft.transform
    def kernel(x: ft.Var[(256, 256), "float32", "inout"]):
        for t in range(100):
            #! label: L0
            for i in range(1, 256):
                for j in range(256):
                    x[i, j] += x[i - 1, j]

    @ft.transform
    def kernel_expected(x: ft.Var[(256, 256), "float32", "inout"]):
        for t in range(100):
            #! label: L0
            for j in range(256):
                for i in range(1, 256):
                    x[i, j] += x[i + -1, j]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_permute("L0")
    kernel = s.func()
    print(kernel)
    assert parallelism == 1
    assert kernel.body.match(kernel_expected.body)


def test_pluto_fuse_bloat():

    @ft.transform
    def kernel(Ls, Lsg, xc, xcg, Yg):
        Ls: ft.Var[(5, 2, 2), "float64"]
        Lsg: ft.Var[(5, 2, 2), "float64", "inout"]
        xc: ft.Var[(1000, 5, 2), "float64"]
        xcg: ft.Var[(1000, 5, 2), "float64", "inout"]
        Yg: ft.Var[(1000, 5, 2), "float64"]
        #! label: L1
        for i in range(999, -1, -1):
            for j in range(4, -1, -1):
                for k in range(1, -1, -1):
                    for p in range(1, -1, -1):
                        Lsg[j, k, p] += Yg[i, j, k] * xc[i, j, p]
        #! label: L2
        for i in range(999, -1, -1):
            for j in range(4, -1, -1):
                for k in range(1, -1, -1):
                    for p in range(1, -1, -1):
                        xcg[i, j, p] += Yg[i, j, k] * Ls[j, k, p]

    with pytest.raises(ft.InvalidSchedule):
        s = ft.Schedule(kernel)
        s.pluto_fuse("L1", "L2")


def test_pluto_fuse_external():

    @ft.transform
    def kernel(N: ft.Var[(), "int64", "input"], x):
        x: ft.Var[(N, N), "float32", "inout"]
        assert 0 < N < 2**30
        for t in range(100):
            #! label: L0
            for i in range(N):
                for j in range(N - 1):
                    x[i, j + 1] += x[i, j]
            #! label: L1
            for i in range(N):
                for j in range(N - 1):
                    x[i, N - 2 - j] += x[i, N - 1 - j]

    @ft.transform
    def kernel_expected(N: ft.Var[(), "int64", "input"], x):
        x: ft.Var[(N, N), "float32", "inout"]
        assert 0 < N < 2**30
        for t in range(100):
            for i in range(N):
                for j in range(N + -1):
                    x[i, j + 1] += x[i, j]
                for j in range(N + -1):
                    x[i, -1 * j + N + -2] += x[i, -1 * j + N + -1]

    print(kernel)
    s = ft.Schedule(kernel)
    _, parallelism = s.pluto_fuse("L0", "L1")
    kernel = s.func()
    print(kernel)
    assert parallelism == 1
    assert kernel.body.match(kernel_expected.body)


def test_pluto_fuse_bloat_external():

    @ft.transform
    def kernel(N: ft.Var[(), "int64", "input"], Ls, Lsg, xc, xcg, Yg):
        Ls: ft.Var[(5, 2, 2), "float64"]
        Lsg: ft.Var[(5, 2, 2), "float64", "inout"]
        xc: ft.Var[(N, 5, 2), "float64"]
        xcg: ft.Var[(N, 5, 2), "float64", "inout"]
        Yg: ft.Var[(N, 5, 2), "float64"]
        #! label: L1
        for i in range(N - 1, -1, -1):
            for j in range(4, -1, -1):
                for k in range(1, -1, -1):
                    for p in range(1, -1, -1):
                        Lsg[j, k, p] += Yg[i, j, k] * xc[i, j, p]
        #! label: L2
        for i in range(N - 1, -1, -1):
            for j in range(4, -1, -1):
                for k in range(1, -1, -1):
                    for p in range(1, -1, -1):
                        xcg[i, j, p] += Yg[i, j, k] * Ls[j, k, p]

    with pytest.raises(ft.InvalidSchedule):
        s = ft.Schedule(kernel)
        s.pluto_fuse("L1", "L2")
        print(s.ast())
