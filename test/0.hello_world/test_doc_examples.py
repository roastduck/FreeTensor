def test_vector_add():
    import freetensor as ft
    import numpy as np

    # Change this line to ft.optimize(verbose=1)
    # to see the resulting native code
    @ft.optimize
    def test(a: ft.Var[(4,), "int32"], b: ft.Var[(4,), "int32"]):
        y = ft.empty((4,), "int32")
        for i in range(4):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])


def test_vector_add_dynamic_length():
    import freetensor as ft
    import numpy as np

    @ft.optimize
    def test(n: ft.Var[(), "int32"], a, b):
        a: ft.Var[(n,), "int32"]
        b: ft.Var[(n,), "int32"]
        y = ft.empty((n,), "int32")
        for i in range(n):
            y[i] = a[i] + b[i]
        return y

    y = test(np.array(4, dtype="int32"), np.array([1, 2, 3, 4], dtype="int32"),
             np.array([2, 3, 4, 5], dtype="int32")).numpy()
    print(y)

    assert np.array_equal(y, [3, 5, 7, 9])
