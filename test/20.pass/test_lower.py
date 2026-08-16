'''
This file tests whether all the passes are effectively combined
'''

import freetensor as ft
import pytest


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_remove_writes_then_prop_one_time_use(as_subprocess):

    @ft.lower(verbose=1, as_subprocess=as_subprocess)
    @ft.transform
    def f(x: ft.Var[(4,), "float32"]):
        a = ft.empty((4,), "float32")
        for i in range(4):
            a[i] = 0
        for i in range(4):
            a[i] += x[i]
        b = ft.empty((4,), "float32")
        for i in range(4):
            b[i] = 0
        for i in range(4):
            b[i] += a[i]
        return b

    @ft.transform
    def expect(x: ft.Var[(4,), "float32"]):
        b = ft.empty((4,), "float32")
        for i in range(4):
            b[i] = x[i]
        return b

    assert expect.body.match(f.body)


def test_timeout_rejects_disabled_subprocess():
    with ft.VarDef("y", (4,), "int32", "output", "cpu") as y:
        with ft.For("i", 0, 4) as i:
            y[i] = 0 * i
    ast = ft.pop_ast()

    with pytest.raises(ft.SubprocessError,
                       match="timeout implies as_subprocess"):
        ft.simplify(ast, as_subprocess=False, timeout=10)
