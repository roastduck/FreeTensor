import ir
import numpy as np

target = ir.CPU()
device = ir.Device(target)


def test_inline():
    @ir.inline
    def test_i(a, b):
        ir.declare_var(a, (2, 2), "int32", "cache", "cpu")
        ir.declare_var(b, (2, 2), "int32", "cache", "cpu")
        c = ir.create_var((2, 2), "int32", "cache", "cpu")
        d = ir.create_var((2, 2), "int32", "cache", "cpu")
        for i in range(2):
            for j in range(2):
                b[i, j] = a[i, j]
                c[i, j] = b[i, j] * a[i, j]
                d[i, j] = b[i, j] + a[i, j]
        return c, d

    @ir.transform
    def test(y, c, d):
        ir.declare_var(y, (2, 2), "int32", "output", "cpu")
        ir.declare_var(c, (2, 2), "int32", "output", "cpu")
        ir.declare_var(d, (2, 2), "int32", "output", "cpu")
        c1, d1 = test_i([[1, 2], [3, 4]], y)
        for i in range(2):
            for j in range(2):
                c[i, j] = c1[i, j]
                d[i, j] = d1[i, j]

    s = ir.Schedule(test)
    func = ir.lower(s.func(), target)
    print(func)
    code = ir.codegen(func, target)
    print(code)

    # x_np = np.array([1, 2, 3, 4], dtype="int32")
    y_np = np.zeros((2, 2), dtype="int32")
    c_np = np.zeros((2, 2), dtype="int32")
    d_np = np.zeros((2, 2), dtype="int32")
    y_arr = ir.Array(y_np, ir.Device(target))
    c_arr = ir.Array(c_np, ir.Device(target))
    d_arr = ir.Array(d_np, ir.Device(target))

    driver = ir.Driver(func, code, device)
    driver.set_params(y=y_arr, c=c_arr, d=d_arr)
    driver.run()
    y_np = y_arr.numpy().reshape(2, 2)
    c_np = c_arr.numpy().reshape(2, 2)
    d_np = d_arr.numpy().reshape(2, 2)

    y_std = np.array([[1, 2], [3, 4]], dtype="int32")
    c_std = np.array([[1, 4], [9, 16]], dtype="int32")
    d_std = np.array([[2, 4], [6, 8]], dtype="int32")
    assert np.array_equal(y_np, y_std) and np.array_equal(c_np, c_std) and np.array_equal(d_np, d_std)



