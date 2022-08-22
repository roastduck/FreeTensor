import freetensor as ft
import numpy as np

target = ft.CPU()
device = ft.Device(target.type())


def test_cache_write():
    a = 128
    b = 256
    m = 4
    # c = 64

    @ft.transform
    def test(w, x, y):
        w: ft.Var[(m, m, a, b), "int32", "input", "cpu"]
        x: ft.Var[(m, m, b, a), "int32", "input", "cpu"]
        y: ft.Var[(m, m, a, a), "int32", "output", "cpu"]
        #! label: L1
        for i in range(m):
            #! label: L2
            for j in range(m):
                #! label: L3
                for k in range(b):
                    #! label: L4
                    for p in range(a):
                        #! label: L5
                        for q in range(a):
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("y.c", (m, m, a, a), "int32", "cache", "cpu")]) as (w, x,
                                                                         y, yc):
        with ft.For("i0", 0, m) as i0:
            with ft.For("i1", 0, m) as i1:
                with ft.For("i2", 0, a) as i2:
                    with ft.For("i3", 0, a) as i3:
                        yc[i0, i1, i2, i3] = y[i0, i1, i2, i3]
        with ft.For("i", 0, m, label='L1') as i:
            with ft.For("j", 0, m, label='L2') as j:
                with ft.For("k", 0, b, label='L3') as k:
                    with ft.For("p", 0, a, label='L6') as p:
                        with ft.For("q", 0, a, label='L7') as q:
                            yc[i, j, p,
                               q] = yc[i, j, p,
                                       q] + w[i, j, p, k] * x[i, j, k, q]
        with ft.For("i0", 0, m) as i0:
            with ft.For("i1", 0, m) as i1:
                with ft.For("i2", 0, a) as i2:
                    with ft.For("i3", 0, a) as i3:
                        y[i0, i1, i2, i3] = yc[i0, i1, i2, i3]
    std = ft.pop_ast()
    std = ft.make_reduction(std)

    s = ft.Schedule(test)
    s = ft.AutoSchedule(s, target, device, rule_set={"cache_write"})
    ast = s.test_round().ast()
    assert std.match(ast)


def test_non_perfect_loop():
    a = 128
    b = 256
    m = 4
    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu")]) as (w, x, y, u):
        with ft.For("i", 0, m, label='L1') as i:
            with ft.For("j", 0, m, label='L2') as j:
                u[i, j] = y[i, j, 0, 0]
                with ft.For("k", 0, b, label='L3') as k:
                    with ft.For("p", 0, a, label='L6') as p:
                        with ft.For("q", 0, a, label='L7') as q:
                            y[i, j, p,
                              q] = y[i, j, p, q] + w[i, j, p, k] * x[i, j, k, q]

    s = ft.pop_ast()

    s = ft.Schedule(s)
    s = ft.AutoSchedule(s, target, device, rule_set={"cache_write"})
    ast = s.test_round().ast()
    print(ast)

    with ft.VarDef([("w", (m, m, a, b), "int32", "input", "cpu"),
                    ("x", (m, m, b, a), "int32", "input", "cpu"),
                    ("y", (m, m, a, a), "int32", "output", "cpu"),
                    ("u", (m, m), "int32", "output", "cpu")]) as (w, x, y, u):
        with ft.For("i", 0, m, label='L1') as i:
            with ft.For("j", 0, m, label='L2') as j:
                u[i, j] = y[i, j, 0, 0]
                with ft.VarDef("y.c", (1, 1, a, a), "int32", "cache",
                               "cpu") as yc:
                    with ft.For("i2", 0, a) as i2:
                        with ft.For("i3", 0, a) as i3:
                            yc[0, 0, i2, i3] = y[i, j, i2, i3]
                    with ft.For("k", 0, b, label='L3') as k:
                        with ft.For("p", 0, a, label='L6') as p:
                            with ft.For("q", 0, a, label='L7') as q:
                                yc[0, 0, p,
                                   q] = yc[0, 0, p,
                                           q] + w[i, j, p, k] * x[i, j, k, q]
                    with ft.For("i2", 0, a) as i2:
                        with ft.For("i3", 0, a) as i3:
                            y[i, j, i2, i3] = yc[0, 0, i2, i3]
    std = ft.pop_ast()
    std = ft.make_reduction(std)
    print(std)
    assert std.match(ast)

    s = ft.Schedule(std)
    s = ft.AutoSchedule(s, target, device, rule_set={"cache_write"})
    ast = s.test_round().ast()
    print(ast)
    assert std.match(ast)
