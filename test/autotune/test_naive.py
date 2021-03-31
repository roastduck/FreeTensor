import ir
import pytest
import autotune
import numpy as np

def test_naive_matmul():
    with ir.VarDef([
            ("a", (1024, 1024,), "int32", "input", "cpu"),
            ("b", (1024, 1024,), "int32", "input", "cpu"),
            ("c", (1024, 1024,), "int32", "output", "cpu")]) as (a, b, c):
        with ir.For("i", 0, 1024, nid = "Li") as i:
            with ir.For("j", 0, 1024, nid = "Lj") as j:
                with ir.For("k", 0, 1024, nid = "Lk") as k:
                    c[i, j] += a[i, k] * b[k, j]
                    
    ast = ir.pop_ast()
    s = ir.Schedule(ast)

    a_np = np.ones((1024, 1024), dtype = "int32")
    b_np = np.ones((1024, 1024), dtype = "int32")
    c_np = np.ones((1024, 1024), dtype = "int32")
    
    target = ir.CPU()
    a_arr = ir.Array(a_np, ir.Device(target))
    b_arr = ir.Array(b_np, ir.Device(target))
    c_arr = ir.Array(c_np, ir.Device(target))

    autotune.naive_search([("Li", 1024), ("Lj", 1024), ("Lk", 1024)], ast, {"a": a_arr, "b": b_arr, "c" : c_arr}, target, iters = 20)
