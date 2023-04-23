import freetensor as ft
import numpy as np
import pytest


def test_basic():

    @ft.optimize(verbose=1)
    def test(x: ft.Var[(4,), "int32"], v: ft.JIT[int]):
        y = x * v
        return y

    # Test final result
    x = ft.array([0, 1, 2, 3], "int32")
    y1 = test(x, 2).numpy()
    assert np.array_equal(y1, [0, 2, 4, 6])
    y2 = test(x, 3).numpy()
    assert np.array_equal(y2, [0, 3, 6, 9])

    # Test memoization
    assert test.instantiate(x, 2) is test.instantiate(x, 2)
    assert test.instantiate(x, 2) is not test.instantiate(x, 3)


def test_check_param_must_be_either_var_or_jit():

    with pytest.raises(ft.StagingError):

        @ft.transform(verbose=2)
        def test(x: ft.Var[(4,), "int32"], v):
            y = x * v
            return y
