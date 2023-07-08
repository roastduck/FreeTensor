'''
This file tests whether all the passes are effectively combined
'''

import freetensor as ft


def test_remove_writes_then_prop_one_time_use():

    @ft.lower(verbose=1)
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
