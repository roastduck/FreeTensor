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


def test_grad():

    @ft.optimize
    @ft.grad(requires=[ft.Parameter(1), ft.Parameter(2)],
             provides=[ft.Return(1)],
             attach_backward=True)
    def test(v: ft.JIT, a, b):
        a: ft.Var[(4,), "float32", "input", "cpu"]
        b: ft.Var[(4,), "float32", "input", "cpu"]
        x = a + b + v
        y = a * b * v
        return x, y

    a = ft.array([1, 2, 3, 4], dtype="float32")
    b = ft.array([5, 6, 7, 8], dtype="float32")
    one = ft.array([1, 1, 1, 1], dtype="float32")

    c, d = test(2, a, b)
    da, db = test.backward(
        **{test.output_name_to_gradient_name[ft.Return(1)]: one
          })[test.input_name_to_gradient_name['a'],
             test.input_name_to_gradient_name['b']]
    assert np.array_equal(da.numpy(), [10, 12, 14, 16])
    assert np.array_equal(db.numpy(), [2, 4, 6, 8])

    c, d = test(3, a, b)
    da, db = test.backward(
        **{test.output_name_to_gradient_name[ft.Return(1)]: one
          })[test.input_name_to_gradient_name['a'],
             test.input_name_to_gradient_name['b']]
    assert np.array_equal(da.numpy(), [15, 18, 21, 24])
    assert np.array_equal(db.numpy(), [3, 6, 9, 12])


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_jacrev():
    import torch

    @ft.optimize
    @ft.jacrev(inputs=[ft.Parameter(1), ft.Parameter(2)],
               output=ft.Return(),
               attach_backward=True,
               verbose=1)
    def f(v: ft.JIT, a, b):
        a: ft.Var[(4, 5), "float32", "input", "cpu"]
        b: ft.Var[(5, 6), "float32", "input", "cpu"]
        return ft.libop.matmul(a, b) * v

    a_torch = torch.rand(4, 5, dtype=torch.float32)
    a_arr = ft.Array(a_torch.numpy())
    b_torch = torch.rand(5, 6, dtype=torch.float32)
    b_arr = ft.Array(b_torch.numpy())
    y_arr = f(2, a_arr, b_arr)
    y_torch = torch.tensor(y_arr.numpy())
    jac = f.backward()
    a_jac_torch = torch.tensor(jac[f.input_name_to_gradient_name['a']].numpy())
    b_jac_torch = torch.tensor(jac[f.input_name_to_gradient_name['b']].numpy())

    y_std = a_torch @ b_torch * 2
    assert torch.all(torch.isclose(y_torch, y_std))

    a_jac_std, b_jac_std = torch.func.jacrev(lambda a, b: a @ b * 2,
                                             argnums=(0, 1))(a_torch, b_torch)
    assert torch.all(torch.isclose(a_jac_torch, a_jac_std))
    assert torch.all(torch.isclose(b_jac_torch, b_jac_std))


@pytest.mark.skipif(not ft.with_pytorch(), reason="requires PyTorch")
def test_pytorch_integration():
    import torch

    @ft.optimize_to_pytorch(verbose=1)
    def sinh(n: ft.JIT, x: ft.Var[(4,), "float64"]):
        y = ft.empty((n,), "float64")
        for i in range(n):
            y[i] = (ft.exp(x[i]) - ft.exp(-x[i])) / 2
        return y

    # Test inference without grad
    x = torch.rand(4, requires_grad=False, dtype=torch.double)
    assert torch.all(
        torch.isclose(sinh(4, x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test forward with grad
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.all(
        torch.isclose(sinh(4, x), (torch.exp(x) - torch.exp(-x)) / 2))

    # Test backward
    x = torch.rand(4, requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(sinh, (4, x))


def test_ast_inline():

    @ft.transform
    def g(v: ft.JIT[int], x: ft.Var[(4,), 'int32', 'inout']):
        x[v] = 2

    @ft.transform(verbose=2)
    def f(x: ft.Var[(8, 4), 'int32', 'inout']):
        g(1, x[3])

    @ft.transform
    def expect(x: ft.Var[(8, 4), 'int32', 'inout']):
        x[3, 1] = 2

    assert expect.body.match(f.body)


@pytest.mark.skipif(not ft.with_cuda(), reason="requires CUDA")
def test_default_device():

    with ft.GPU(0):

        @ft.optimize(verbose=1)
        def test(x: ft.Var[(4,), "int32"], v: ft.JIT[int]):
            y = x * v
            return y

    x = ft.array([0, 1, 2, 3], "int32")
    instance = test.instantiate(x, 2)

    @ft.lower
    @ft.transform
    def expect(x: ft.Var[(4,), "int32", "input", "gpu/global"]):
        y = x * 2
        return y

    assert expect.body.match(instance.func.body)

    assert instance.device == ft.GPU(0)
