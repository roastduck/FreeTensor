import fnmatch

import freetensor as ft
import pytest


def fnmatch_list(strings, patterns):
    if len(patterns) != len(strings):
        return False
    for string, pattern in zip(strings, patterns):
        if not fnmatch.fnmatch(string, pattern):
            return False
    return True


def test_fission_then_pluto_fuse():
    with ft.VarDef([("x1", (1000,), "int32", "input", "cpu"),
                    ("x2", (1000,), "int32", "input", "cpu"),
                    ("y", (1000,), "int32", "inout", "cpu")]) as (x1, x2, y):
        with ft.For("i", 0, 999, label="Li") as i:
            ft.MarkLabel("S0")
            y[i] += x1[i]
            ft.MarkLabel("S1")
            y[i + 1] += x2[i]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_pluto(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs,
        ['fission(Li, before, S1, *)', 'pluto_fuse($fission.0{Li}, Li, *)'])


def test_fission_then_pluto_fuse_2d():
    with ft.VarDef([("x1", (1000, 1000), "int32", "input", "cpu"),
                    ("x2", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu")]) as (x1, x2,
                                                                       y):
        with ft.For("i", 0, 999, label="Li") as i:
            with ft.For("j", 0, 999, label="Lj") as j:
                ft.MarkLabel("S0")
                y[i, j] += x1[i, j]
                ft.MarkLabel("S1")
                y[i + 1, j + 1] += x2[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_pluto(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(
        logs,
        ['fission(Li, before, S1, *)', 'pluto_fuse($fission.0{Li}, Li, *)'])


def test_fission_then_pluto_fuse_2d_4way():
    with ft.VarDef([("x1", (1000, 1000), "int32", "input", "cpu"),
                    ("x2", (1000, 1000), "int32", "input", "cpu"),
                    ("x3", (1000, 1000), "int32", "input", "cpu"),
                    ("x4", (1000, 1000), "int32", "input", "cpu"),
                    ("y", (1000, 1000), "int32", "inout", "cpu")
                   ]) as (x1, x2, x3, x4, y):
        with ft.For("i", 0, 999, label="Li") as i:
            with ft.For("j", 0, 999, label="Lj") as j:
                ft.MarkLabel("S0")
                y[i, j] += x1[i, j]
                ft.MarkLabel("S1")
                y[i, j + 1] += x2[i, j]
                ft.MarkLabel("S2")
                y[i + 1, j] += x2[i, j]
                ft.MarkLabel("S3")
                y[i + 1, j + 1] += x2[i, j]

    ast = ft.pop_ast(verbose=True)
    s = ft.Schedule(ast)
    s.auto_pluto(ft.CPU())
    print(s.ast())
    logs = list(map(str, s.logs()))
    print(logs)
    assert fnmatch_list(logs, [
        'fission(Li, before, S1, *)', 'fission(Li, before, S2, *)',
        'fission(Li, before, S3, *)',
        'pluto_fuse($fission.0{Li}, $fission.2{Li}, *)',
        'pluto_fuse($pluto_fuse.fused.0{$fission.0{Li}, $fission.2{Li}}, $fission.3{Li}, *)',
        'pluto_fuse($pluto_fuse.fused.0{$pluto_fuse.fused.0{$fission.0{Li}, $fission.2{Li}}, $fission.3{Li}}, Li, *)'
    ])
