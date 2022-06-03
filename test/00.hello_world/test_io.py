import freetensor as ft
import numpy as np
import pytest


def test_numpy_strided():

    @ft.optimize
    def test(x: ft.Var[(2, 2), "int32"]):
        y = ft.empty((2, 2), "int32")
        for i in range(2):
            for j in range(2):
                y[i, j] = x[i, j] + 1
        return y

    x = np.array([[0, 1], [2, 3]], dtype="int32").transpose()
    assert x.strides[0] < x.strides[1]
    y = test(x).numpy()
    assert np.array_equal(y, np.array([[1, 3], [2, 4]], dtype="int32"))
