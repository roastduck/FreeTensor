import numpy as np

import freetensor as ft
from freetensor import libop


def test_out_of_place():

    @ft.optimize(verbose=1)
    def f(x: ft.Var[(3, 5), "float32", "input", "cpu"]):
        #! label: reshape
        return libop.transpose(x)

    x_np = np.random.rand(3, 5).astype("float32")
    y_np = f(x_np).numpy()

    assert np.all(y_np == x_np.transpose())


def test_perm():

    @ft.optimize(verbose=1)
    def f(x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]):
        #! label: reshape
        return libop.transpose(x, perm=[0, 2, 1])

    x_np = np.random.rand(3, 4, 5).astype("float32")
    y_np = f(x_np).numpy()

    assert np.all(y_np == x_np.transpose(0, 2, 1))


def test_circular_axis():

    @ft.optimize(verbose=1)
    def f(x: ft.Var[(3, 4, 5), "float32", "input", "cpu"]):
        #! label: reshape
        return libop.transpose(x, perm=[0, 2, -2])

    x_np = np.random.rand(3, 4, 5).astype("float32")
    y_np = f(x_np).numpy()

    assert np.all(y_np == x_np.transpose(0, 2, 1))
